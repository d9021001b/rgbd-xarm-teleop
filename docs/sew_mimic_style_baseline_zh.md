# SEW-Mimic-style Geometric Baseline 復刻記錄

建立日期：2026-05-13

## 目的

這個 baseline 參考 SEW-Mimic 的核心思想：不要只把人手位置丟給 IK，而是從 shoulder-elbow-wrist 三點取出上臂、下臂與肘部 swivel 幾何，讓機械臂構型盡量像人的手臂。

## Paper 核心

SEW-Mimic 使用 human shoulder、elbow、wrist keypoints：

```text
u = unit(elbow - shoulder)
l = unit(wrist - elbow)
```

然後依序對齊 robot upper arm、lower arm、wrist orientation。原 paper 對特定 7-DoF humanoid-style arm 使用 closed-form geometric subproblems，目標是非常快的 arm-pose mimicry。

## 本專案復刻方式

xArm7 的 link/axis 結構和 humanoid arm 不完全相同，所以本專案先復刻最可轉移的部分：

```text
SMPL-X shoulder/elbow/wrist
-> upper/lower arm direction
-> shoulder-wrist line
-> law-of-cosines reachable elbow circle
-> elbow swivel direction from human arm
-> xArm7 elbow target
-> lightweight TCP + elbow-target IK
```

也就是說，這版不是完整的 SEW-Mimic joint-axis closed-form solver，而是 **SEW-Mimic-style geometric elbow-swivel baseline**。它保留我們抓杯任務需要的 TCP 位置，但用 SEW 幾何決定 xArm7 elbow posture。

## 實作位置

- Solver / helper：`docker/retarget-smplx-hand-to-xarm.py`
- Config：`configs/xarm7_sew_mimic_baseline_no_moveit.json`
- 本次輸出：`recordings/sew_mimic_baseline_20260513_180938/`

## 本次 dry-run 指標

輸入資料：

```text
recordings/taichi_mediapipe3d_smplx_20260510_150729/smplx_fit_right_arm_trajectory.json
```

執行條件：

```text
18 seconds, 10 fps, 181 trajectory points, raw reconstructed SMPL-X trajectory
```

結果摘要：

| 指標 | SEW-Mimic-style baseline |
|---|---:|
| TCP raw position error mean | 0.00010 m |
| TCP raw position error max | 0.00414 m |
| included-angle absolute error mean | 19.89 deg |
| forearm projected mean-max error | 83.39 deg |
| upper-arm projected mean-max error | 49.33 deg |
| SEW elbow target error mean | 0.0626 m |
| SEW elbow target error max | 0.0793 m |
| reach-clamped frames | 2 |
| runtime | 9.06 s / 181 frames |

## 觀察

這個 baseline 很快，約 50 ms/frame，比 OCRA-style finite-difference baseline 快很多，也比目前 calibrated functional/null-space 方法更接近即時。

但它也很清楚暴露出 trade-off：

1. TCP 位置非常準，因為本專案仍保留 xArm7 抓杯任務的 TCP primary objective。
2. 上臂與 included-angle 比 OCRA baseline 好一些。
3. forearm projected angle error 很大，因為 xArm7 wrist/tool orientation 被刻意降權，避免它破壞 TCP 位置。
4. 這版還不是 paper 原始 closed-form axis solver；完整復刻要再把 xArm7 的 joint axis mapping 改成 Algorithm 1 / Algorithm 2 那種逐段 closed-form AlignAxis。

## 和主方法比較的意義

SEW-Mimic-style baseline 證明：

- 幾何 elbow swivel 很適合當 fast baseline。
- 若只靠 SEW 幾何，能大幅降低 runtime。
- 但對本專案的 xArm7 cup-grasp teleoperation 來說，仍需要 functional forearm / wrist objective 才能讓 gripper 姿態更像人的手部動作。

因此它可以作為本專案下一步即時化的基礎：把目前 calibrated functional objective 的部分項目，改寫成 SEW 幾何 seed + 少量 null-space correction。
