import argparse
import dataclasses
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
import torch.multiprocessing as mp


@dataclasses.dataclass
class RuntimeWindowMsg:
    """Headless-safe copy of the visualization control message defaults."""

    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


def apply_rotation_prior(T_WC, quaternion_xyzw):
    delta_data = torch.zeros((1, 8), dtype=T_WC.data.dtype, device=T_WC.data.device)
    delta_data[:, 3:7] = torch.as_tensor(
        quaternion_xyzw, dtype=T_WC.data.dtype, device=T_WC.data.device
    )
    delta_data[:, 7] = 1.0
    return T_WC * lietorch.Sim3(delta_data)


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def confirmed_retrieval_indices(
    retrieval_inds,
    retrieval_history,
    required_frames,
    candidate_frame_ids=None,
    current_frame_id=None,
    immediate_frame_gap=0,
    frame_id_radius=0,
):
    """Keep local candidates or scene-consistent candidates across queries."""
    candidates = list(retrieval_inds)
    if candidate_frame_ids is None:
        candidate_frame_ids = {candidate: candidate for candidate in candidates}
    if required_frames <= 1:
        return candidates
    recent = retrieval_history[-(required_frames - 1) :]
    confirmed = []
    for candidate in candidates:
        candidate_frame_id = candidate_frame_ids[candidate]
        if (
            current_frame_id is not None
            and 0 < current_frame_id - candidate_frame_id <= immediate_frame_gap
        ):
            confirmed.append(candidate)
            continue
        if len(recent) < required_frames - 1:
            continue
        if all(
            any(
                abs(candidate_frame_id - previous_frame_id) <= frame_id_radius
                for previous_frame_id in previous
            )
            for previous in recent
        ):
            confirmed.append(candidate)
    return confirmed


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)
    retrieval_history = []
    confirmation_frames = int(config["retrieval"].get("confirmation_frames", 1))
    confirmation_frame_id_radius = int(
        config["retrieval"].get("confirmation_frame_id_radius", 0)
    )
    confirmation_immediate_frame_gap = int(
        config["retrieval"].get("confirmation_immediate_frame_gap", 0)
    )
    density_window = int(config["retrieval"].get("density_window", 10))
    dense_min_keyframes = int(
        config["retrieval"].get("dense_min_keyframes", density_window)
    )
    dense_max_mean_frame_gap = float(
        config["retrieval"].get("dense_max_mean_frame_gap", 0)
    )
    retrieval_enabled = bool(config["retrieval"].get("enabled", True))
    if confirmation_frames < 1:
        raise ValueError("retrieval.confirmation_frames must be positive")
    if confirmation_frame_id_radius < 0:
        raise ValueError("retrieval.confirmation_frame_id_radius must be non-negative")
    if confirmation_immediate_frame_gap < 0:
        raise ValueError(
            "retrieval.confirmation_immediate_frame_gap must be non-negative"
        )
    if density_window < 2:
        raise ValueError("retrieval.density_window must be at least two")
    if dense_min_keyframes < 3 or dense_min_keyframes > density_window:
        raise ValueError(
            "retrieval.dense_min_keyframes must be within [3, density_window]"
        )
    if dense_max_mean_frame_gap < 0:
        raise ValueError("retrieval.dense_max_mean_frame_gap must be non-negative")
    previous_confirmation_mode = None
    dense_confirmation_latched = False

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        raw_retrieval_inds = (
            list(
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
            )
            if retrieval_enabled
            else []
        )
        candidate_frame_ids = {
            candidate: int(keyframes[candidate].frame_id)
            for candidate in raw_retrieval_inds
        }
        current_frame_id = int(frame.frame_id)
        density_start = max(1, idx - density_window)
        density_frame_ids = [
            int(keyframes[keyframe_index].frame_id)
            for keyframe_index in range(density_start, idx + 1)
        ]
        dense_keyframes_observed = False
        if len(density_frame_ids) >= dense_min_keyframes:
            mean_frame_gap = (
                density_frame_ids[-1] - density_frame_ids[0]
            ) / (len(density_frame_ids) - 1)
            dense_keyframes_observed = mean_frame_gap <= dense_max_mean_frame_gap
        else:
            mean_frame_gap = float("inf")
        dense_confirmation_latched = (
            dense_confirmation_latched or dense_keyframes_observed
        )
        dense_keyframes = dense_confirmation_latched
        confirmation_mode = "dense_scene_cluster" if dense_keyframes else "sparse_exact"
        if confirmation_mode != previous_confirmation_mode:
            print(
                "Loop confirmation mode",
                idx,
                ":",
                confirmation_mode,
                "mean_source_frame_gap=",
                mean_frame_gap,
            )
            previous_confirmation_mode = confirmation_mode
        immediate_frame_gap = (
            confirmation_immediate_frame_gap if dense_keyframes else 0
        )
        frame_id_radius = confirmation_frame_id_radius if dense_keyframes else 0
        retrieval_inds = confirmed_retrieval_indices(
            raw_retrieval_inds,
            retrieval_history,
            confirmation_frames,
            candidate_frame_ids=candidate_frame_ids,
            current_frame_id=current_frame_id,
            immediate_frame_gap=immediate_frame_gap,
            frame_id_radius=frame_id_radius,
        )
        rejected_retrieval_inds = sorted(set(raw_retrieval_inds) - set(retrieval_inds))
        retrieval_history.append(set(candidate_frame_ids.values()))
        if rejected_retrieval_inds:
            print(
                "Retrieval pending multi-frame confirmation",
                idx,
                ": ",
                rejected_retrieval_inds,
            )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="optional MASt3R model checkpoint; defaults to the official metric model",
    )

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    use_imu_rotation_prior = bool(
        config["tracking"].get("imu_rotation_prior", False)
    )
    use_stereo_pointmap_scale = bool(
        config["tracking"].get("stereo_pointmap_scale_prior", False)
    )
    if use_imu_rotation_prior and dataset.rotation_priors is None:
        raise ValueError(
            "tracking.imu_rotation_prior requires imu_rotation_priors.csv"
        )
    if use_imu_rotation_prior:
        print("Using calibrated IMU rotation priors for frame initialization")
    h, w = dataset.get_img_shape()[0]
    if use_stereo_pointmap_scale and dataset.stereo_depth_provider is None:
        raise ValueError(
            "tracking.stereo_pointmap_scale_prior requires exported stereo-right images"
        )
    if use_stereo_pointmap_scale:
        print("Using synchronized D405 stereo pointmap scale priors")

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        from mast3r_slam.visualization import run_visualization

        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(path=args.checkpoint, device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = RuntimeWindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []
    tracked_poses = []
    online_poses = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        if use_imu_rotation_prior and i > 0:
            T_WC = apply_rotation_prior(T_WC, dataset.get_rotation_prior(i))
        metric_depth = (
            dataset.get_stereo_depth(i, (h, w)) if use_stereo_pointmap_scale else None
        )
        frame = create_frame(
            i,
            img,
            T_WC,
            img_size=dataset.img_size,
            device=device,
            metric_depth=metric_depth,
        )

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            (X_init,), stereo_scale_report = tracker.scale_pointmaps(
                (X_init,), C_init, frame.metric_depth
            )
            frame.metric_anchor_mask = tracker.last_metric_anchor_mask
            if use_stereo_pointmap_scale:
                print("Stereo pointmap scale", frame.frame_id, stereo_scale_report)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            tracked_poses.append(
                (frame.frame_id, 0, lietorch.Sim3.Identity(1).data.cpu())
            )
            online_poses.append((frame.frame_id, frame.T_WC.data.detach().cpu()))
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        add_new_kf = False
        tracked = False
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            else:
                tracked = True
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            (X,), stereo_scale_report = tracker.scale_pointmaps(
                (X,), C, frame.metric_depth
            )
            frame.metric_anchor_mask = tracker.last_metric_anchor_mask
            if use_stereo_pointmap_scale:
                print("Stereo pointmap scale", frame.frame_id, stereo_scale_report)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        if tracked:
            anchor_idx = len(keyframes) - 1
            if add_new_kf:
                relative_pose = lietorch.Sim3.Identity(1)
            else:
                relative_pose = keyframes[anchor_idx].T_WC.inv() * frame.T_WC
            tracked_poses.append(
                (frame.frame_id, anchor_idx, relative_pose.data.detach().cpu())
            )
            online_poses.append((frame.frame_id, frame.T_WC.data.detach().cpu()))
        elif mode == Mode.RELOC and states.get_mode() == Mode.TRACKING:
            anchor_idx = len(keyframes) - 1
            anchor = keyframes[anchor_idx]
            if anchor.frame_id == frame.frame_id:
                tracked_poses.append(
                    (frame.frame_id, anchor_idx, lietorch.Sim3.Identity(1).data.cpu())
                )
                online_poses.append((frame.frame_id, frame.T_WC.data.detach().cpu()))
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_full_traj(
            save_dir,
            f"{seq_name}_full.txt",
            dataset.timestamps,
            keyframes,
            tracked_poses,
        )
        eval.save_online_traj(
            save_dir,
            f"{seq_name}_online.txt",
            dataset.timestamps,
            keyframes,
            online_poses,
        )
        if not args.no_reconstruction:
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
