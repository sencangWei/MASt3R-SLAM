# Experimental onboard VINS takeover (2026-10-10)

This is an **opt-in mechanism candidate**, not a new production default and not
a claim of <=10 mm SLAM accuracy. The official learned checkpoint is unchanged.

Enable `tracking.vins_visual_takeover: true` together with the existing stereo
metric pointmap and VINS camera-prior configuration. Supply contiguous,
camera-origin, synchronized onboard poses via `MAST3R_VINS_CAMERA_POSES`.
`MAST3R_VINS_TAKEOVER_LOG` records each decision as JSONL. No external Tracker,
robot, Lighthouse, or evaluation error is read by these decisions.

## Behavior

- Reuse existing stereo support/confidence checks for unreliable pointmaps.
- Reject nonfinite, uphill or exhausted calibrated visual solves when the
  onboard metric prior is available. The final pose must have an evaluated
  cost: an unevaluated terminal GN increment is not published.
- Compose the adjacent **camera-relative metric SE3** VINS increment onto the
  previous visual-world pose. Preserve its world origin, orientation gauge and
  Sim3 scale; do not concatenate absolute trajectories from different worlds.
- Require finite, valid adjacent poses, positive scale, nonzero quaternions,
  and <=30 mm adjacent onboard displacement (the existing short-frame physical
  sanity bound). When gyro initialization is enabled, check rotation consistency.
- A pose-only frame cannot update the canonical reference, become a graph
  keyframe, or seed the descriptor anchor cache. Stationary VINS still emits the
  same pose, not an absent frame.
- Return to visual tracking only after valid geometry and two distinct,
  consecutive internally consistent frames. The 3-sigma VINS agreement gate
  is **not** the external accuracy threshold or a trajectory correction limit.
- When the graph reference becomes stale, keep a recent independently
  stereo-scaled self pointmap as a **local recovery reference**, outside
  SharedKeyframes. Its pose/maps/intrinsics are copied, and its scale state
  is isolated from the old graph. After two consistent frames, visual pose
  estimation can resume against this recent reference. Neither rejected nor
  recovered local observations write the old graph or create graph edges.
- Experimental graph re-entry is still default-off. When
  `tracking.vins_visual_safe_reentry: true`, a shadow-recovered frame must
  first pass a read-only graph-anchor probe, a multi-frame VINS consistency
  window, and a mandatory bidirectional D405 stereo metric bridge before it can
  become a graph keyframe again. The window uses the existing
  `vins_translation_prior_sigma_m`, `vins_reentry_window_max_sigma` and a
  hard `vins_reentry_window_max_error_m` cap (default 12 mm), so same-direction
  adjacent errors cannot accumulate indefinitely. Rejected recovery frames keep
  predicting from the last trusted window anchor until validated re-entry, rather
  than re-anchoring on the rejected pose. The bridge uses temporary
  depth-valid masks when the global pointmap-depth-anchor mode is disabled; it
  does not enable that sealed parameter family or rewrite keyframe pointmaps.
- Re-entry frames are marked `requires_metric_bridge` in SharedKeyframes. The
  backend no longer grants the normal consecutive-edge exemption to those
  frames, so a bridge edge must pass the ordinary match gate and, when enabled,
  the metric loop gate too.
- Before VINS initialization, retain the existing healthy visual start; there
  is no valid VINS motion to fabricate. Unexpected software errors propagate
  instead of being silently converted to a successful fallback.

## Verification and known limits

Fresh frontend suite after local-reference recovery: 65 tests passed, including composition/gauge/scale,
invalid/absent priors, stationary output, no reference mutation, recovery
confirmation, expected numerical rejection and unexpected-error propagation,
reference-copy isolation, fresh-reference selection, and a confirmed visual
recovery path that cannot read/write/append SharedKeyframes.

First actual ind2 right-eye 581-frame prefix experiment (v1): frames 569--580
were covered by VINS fallback. The world-metric 576->577 step was 2.384 mm;
the prefix's largest final adjacent displacement was 8.653 mm. These are
**motion continuity**, not GT position errors. v1 missed 10 early frames before
VINS initialization; the v2 candidate restricts the solver rejection guard to
states with an actual metric prior. Fresh full replays of ind2, heldout1 and
heldout4 all emit 1199/1199 poses without missing indices. The ind2 573->574
step is now 3.200 mm, against 58.879 mm in the former metric-rescue run.

Before local-reference recovery, long pose-only streaks left the last visual
keyframe too old: the full ind2 replay's last 554 frames remained pose-only. Heldout1's maximum
streak is 83 frames, heldout4's is 203. This is an unresolved mechanism gap,
not an accepted production result. New full replays and independent fused
precision checks are required for the local-reference extension.
`degraded_frames`, `window_anchor_frame_id`, recovery decisions and re-entry
reasons are logged. Do not promote a coverage-only VINS stream as successful
learned-visual recovery. The unchanged independent evaluator remains required
before promotion.

Main integration configuration and reproducible diagnostic launcher live in
`ego_vio_humble/config/mast3r_slam_d405_vins_visual_takeover.yaml` and
`.planning/ind2_raw_diagnostics_20261010/run_vins_takeover_frontend.sh`.
The launcher deliberately labels reuse of frozen VINS priors; it is not a fresh
raw end-to-end VINS replay. Historical artifacts and formal defaults stay intact.
