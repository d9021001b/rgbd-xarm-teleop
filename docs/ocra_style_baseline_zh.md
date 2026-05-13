# OCRA-style Baseline 復刻記錄

建立日期：2026-05-13

## 目的

這個 baseline 用來對照本專案目前的 xArm7 teleoperation retargeting 方法。它參考 OCRA 的核心想法：每一幀不是只追 TCP 位置，而是把「手臂骨架相似」與「手部姿態相似」放進同一個最佳化目標中。

## 復刻的 OCRA-style 目標函數

本專案的 OCRA-style baseline 使用：

```text
epsilon_R = alpha * skeleton_error^2 + beta * hand_orientation_error^2
```

其中：

- `skeleton_error`：比較 SMPL-X 右肩、右肘、右手腕形成的人類手臂鏈，和 xArm7 shoulder proxy、elbow proxy、TCP 形成的 robot chain。
- `hand_orientation_error`：比較由 SMPL-X 前臂方向推導出的 functional wrist orientation，和 xArm7 TCP/gripper orientation。
- `alpha = 0.67`、`beta = 0.33`：讓骨架相似度比手部姿態更重，符合 OCRA 以 skeleton similarity 為主的精神。
- `target_position_weight = 1.0`：這是本專案額外加的 task tie-breaker，因為我們不是單純比劃手臂，而是要讓 TCP 對應抓杯位置。

## 與目前主方法的差異

目前主方法是：

```text
TCP hard priority
-> forearm / upper-arm / included-angle 作 null-space secondary objective
-> anti-self-insertion / MoveIt2 safety 可接入
```

OCRA-style baseline 是：

```text
skeleton similarity + hand orientation weighted objective
-> 每一幀一起最佳化
-> 沒有 TCP hard priority
-> 沒有 hierarchical null-space
```

因此它比較像「整條手臂姿態看起來像」，而我們目前方法比較像「先確保手/TCP 到對的位置，再讓手臂構型盡量像」。

## 實作位置

- Solver：`docker/retarget-smplx-hand-to-xarm.py`
- Config：`configs/xarm7_ocra_baseline_no_moveit.json`
- 本次輸出：`recordings/ocra_baseline_20260513_171908/`

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

| 指標 | OCRA-style baseline |
|---|---:|
| TCP raw position error mean | 0.0644 m |
| TCP raw position error max | 0.1843 m |
| similarity-fit position error mean | 0.0441 m |
| included-angle absolute error mean | 30.56 deg |
| forearm projected mean-max error | 44.65 deg |
| upper-arm projected mean-max error | 68.72 deg |
| OCRA skeleton RMSE mean | 0.1581 m |
| OCRA orientation mean | 0.242 rad |
| runtime | 288.54 s / 181 frames |

## 觀察

OCRA-style baseline 能跑完整 trajectory，但在本任務中有三個明顯弱點：

1. TCP 不是 hard constraint，所以在 sweep/return 階段最大誤差到 18 cm 等級。
2. 上臂 projected angle error 偏大，代表 weighted objective 容易在 xArm7 結構限制下犧牲 elbow-swivel。
3. 每幀有限差分最佳化成本高，18 秒 10 fps 需要約 289 秒，距離即時 teleoperation 還很遠。

這正好形成對照：OCRA-style baseline 可以當 paper baseline，而本專案的創新點是把 TCP hard priority、functional arm correspondence、included-angle matching 與 safety context 整合成更適合桌面抓杯的 xArm7 retargeting。
