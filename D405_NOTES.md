# D405 前端笔记

## 2026-09-20：`tight` 候选的运动关键帧被**静默关闭** —— 09-14 复原失败的根因

### 症状

用现役默认配置重跑 `20260914_validation_v10_batch/group1/fusion/tight` 的前端，
产出的 `dataset_full.txt` 与 09-14 基线**前 457 帧逐位相同，从帧 457 起分叉**，一路差到尾。

### 根因

`mast3r_slam/tracker.py` 的 `motion_keyframe_trigger()`：

```python
translation_limit  = float(cfg.get("motion_keyframe_translation", 0.0))
rotation_limit_deg = float(cfg.get("motion_keyframe_rotation_deg", 0.0))
...
triggered = (translation_limit > 0.0 and translation >= translation_limit) or \
            (rotation_limit_deg > 0.0 and rotation_deg >= rotation_limit_deg) or aged_motion
```

**缺键 → 默认 `0.0` → `> 0.0` 守卫恒假 → `triggered` 永远为 False。**
不报错、不警告，运动关键帧机制整个消失。

- 09-14 的 `tight` 候选用的是 `config/mast3r_slam_d405_offline_motion_kf_tight.yaml`
  （`motion_keyframe_translation: 0.15` / `motion_keyframe_rotation_deg: 5.0`），
  由 `scripts/mast3r_slam_adaptive_precision_workflow.sh:46` 通过 `MAST3R_SLAM_CONFIG` 传入；
- 现役 `scripts/mast3r_slam_precision_workflow.sh` 默认
  `config/mast3r_slam_d405_offline.yaml`，**不含这两个键**（它们是 sparse 候选的配置）；
- `mast3r_slam_adaptive_precision_workflow.sh` 已**无人调用**（死代码），
  靠它传配置的那条路径随之消失。

### 证据

`Motion keyframe` 的打印在 `tracker.py:529`。

| 运行 | 配置 | Motion keyframe 次数 | 首个触发帧 |
|---|---|---|---|
| tight 基线（09-14） | `..._motion_kf_tight.yaml` | **73** | **457** |
| sparse 基线（09-14） | `..._offline.yaml` | 0 | — |
| 现役默认配置跑 tight 数据集 | `..._offline.yaml` | 0 | — |

**基线首个触发帧 457 == 分叉首帧。** 帧 457 之前两者关键帧决策完全一致，故逐位相同。

### 判决性实验

用 `motion_kf_tight.yaml` 重跑同一数据集：

```
sha256(dataset_full.txt) = 036067f60dfc121889258fd2d8bc47c61aff6708c1159a3e01a09b250c6d208a
                         == 09-14 基线（逐字节相同）
Motion keyframe: 基线 73 / 本次 73
```

⇒ 三件事同时成立：

1. **09-14 前端可完整复原**（只差一个配置文件的传参）；
2. **分叉 100% 由配置造成**，与 09-19 的 `1484dd0`（`tracker.py` +420 行等 1386 行插入）**无关**
   —— 基线由 `e6f4e3d`+脏工作区产出，本次由 `1484dd0`+本文件产出，输出逐字节相同；
3. 该次运行**开着匹配埋点**仍逐字节相同 ⇒ **埋点零行为影响**。

### 匹配埋点

`mast3r_slam/tracker.py` 新增 env 门控埋点：`MAST3R_MATCH_LOG=<path>` 时逐帧追加

```
frame_id,n_match,n_match_Q,n_opt,n_total
```

分别对应 `valid_match_k.sum()`、`valid_kf.sum()`、`valid_opt.sum()`、`valid_opt.numel()`。
**不设该环境变量时零行为影响**（已用上面的逐字节对照证明）。

### 待办 / 风险

- 现役生产路径若重跑前端，`tight` 候选**不再有运动关键帧** ⇒ 与 09-14 不可复现。
  要复原必须显式 `MAST3R_SLAM_CONFIG=.../mast3r_slam_d405_offline_motion_kf_tight.yaml`。
- 历史 `fusion_current/*` 的评测全部建立在 09-14 的前端产物上
  （尾段重跑复用 `fusion/<subset>/mast3r/`，不重跑前端），评测结论仍有效，
  但**当前工具链的默认调用复现不出它们**。
- 这两个键的缺省值 `0.0` + `> 0.0` 守卫是一个易踩的静默陷阱，
  与 `--calib` 的循环门控（`use_calib` 默认 False 导致静默丢相机模型）属于同一类。
