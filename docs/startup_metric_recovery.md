# Measured stereo startup output recovery

This candidate fills an output hole when normal visual tracking/relocalization
fails before the first usable VINS relative measurement. It does not relax the
learned pointmap gate, source coverage check or downstream precision acceptance.

Use original rectified grayscale frames, recording camera intrinsics/baseline,
existing D405 stereo depth, bidirectional LK and the existing metric PnP solver.
PnP uses 100 points / 0.4 inlier ratio / 2 px; its physical translation is capped
by the existing startup30mm gate and checked against incremental IMU at1.5deg.
The recorded quaternion row is an increment, not an absolute orientation;
invert it for source-camera -> current-camera PnP rotation consistency.

Emit only a measured `pose_only` output with `startup_visual_unverified` and
`recovery_method=stereo_lk_pnp_imu`. Existing learned-factor admission excludes
this state. Do not write keyframes/descriptor anchors/pointmaps or overwrite
the pending RELOC SharedStates payload. Normal relocalization still runs.
Consecutive missing startup outputs have a separate measured output anchor.
Physical translation is metres, not multiplied by visual Sim3 scale.
Without an established metric pointmap gauge, only the initial unit-scale origin
or a measured output chain may anchor the metric proposal. If an intervening
healthy visual frame is still unscaled, remeasure every adjacent stereo/IMU
pair from the last physical anchor to the missing frame. Reject the entire
proposal if any intermediate measurement fails; do not use or rewrite the
intervening visual output. Log each physical pair in `measured_pairs`.
Do not use this exception for later VINS gaps, recovery windows or diagnostic
graph probes. Healthy normal tracking remains first choice.

Validation so far: 137 CPU unit tests; real-image local proposals on dev2,
dev1 and heldout1. dev2 index1 PnP224/225 inliers, P950.4304px, IMU
disagreement0.00538deg. These are not full-frontend coverage or E2E precision
claims. The active22-case batch retains its old frozen source version.
Full visual replays and raw E2E dev2 plus passing controls must precede any
acceptance claim or production promotion.

Isolated-worktree packaging: run `.venv/bin/python scripts/bind_runtime_lietorch.py`
before the workflow. This validates the actually imported source repository,
clean tracked source and pinned commit `e7df86554156b36846008d8ddbcc4d8521a16554`,
then creates only a missing `thirdparty/lietorch` symlink to that source. It
refuses conflicting paths and prints source/native-extension hashes. The
existing run-manifest git query then reads the real dependency identity, not
the enclosing toolchain repository's commit. No dependency source is changed.
