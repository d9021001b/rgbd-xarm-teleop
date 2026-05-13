# SEW Seed + Functional Null-space Hybrid 記錄

建立日期：2026-05-13

## 目的

這一版把 SEW-Mimic-style geometric baseline 和本專案原本的 functional/null-space retargeting 合起來：

```text
SEW-Mimic geometric elbow-swivel seed
-> TCP hard solve
-> functional forearm / upper-arm / included-angle null-space correction
```

目標是取得三者平衡：

- 接近 SEW-Mimic 的速度。
- 保留 xArm7 抓杯任務需要的 TCP 精度。
- 比 pure SEW baseline 更像 SMPL-X 右臂動作。

## 方法

每一幀分成兩層：

1. **SEW geometric seed**

   從 SMPL-X `right_shoulder -> right_elbow -> right_wrist` 取得人的 upper/lower arm direction，用 law-of-cosines 在 xArm7 shoulder-wrist reachable circle 上計算一個 elbow target。

2. **Functional correction**

   從 SEW seed 出發，先硬保 TCP position，再只在 TCP null-space 裡修：

   ```text
   forearm projected angles
   upper-arm projected angles
   included angle
   ```

   這比從上一幀姿態直接做完整最佳化更快，也比 pure SEW 更能控制手臂姿態。

## 實作位置

- Solver：`docker/retarget-smplx-hand-to-xarm.py`
- Config：`configs/xarm7_sew_functional_hybrid_no_moveit.json`
- 本次輸出：`recordings/sew_functional_hybrid_20260513_195310/`

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

| 指標 | Pure SEW baseline | SEW + functional hybrid |
|---|---:|---:|
| runtime | 9.06 s / 181 frames | 14.72 s / 181 frames |
| approx time per frame | 50 ms | 81 ms |
| TCP raw position error mean | 0.00010 m | 0.00154 m |
| TCP raw position error max | 0.00414 m | 0.02284 m |
| included-angle absolute error mean | 19.89 deg | 4.42 deg |
| forearm projected mean-max error | 83.39 deg | 25.14 deg |
| upper-arm projected mean-max error | 49.33 deg | 41.53 deg |

## 觀察

Hybrid 明顯比 pure SEW 更像人的右臂：

- included-angle mean error 從 `19.89 deg` 降到 `4.42 deg`。
- forearm projected mean-max error 從 `83.39 deg` 降到 `25.14 deg`。
- runtime 從 `9.06 s` 增加到 `14.72 s`，但仍遠快於重型逐幀最佳化 baseline。

代價是 TCP max error 從 `4.14 mm` 增加到 `22.84 mm`，但平均仍是 `1.54 mm`，對離線 retargeting 與 teleoperation preview 來說仍在可接受範圍內。

## 結論

這版比 pure SEW 更適合作為本專案的即時化方向：

```text
SEW = fast geometric posture seed
functional null-space = teleoperation posture correction
TCP hard priority = cup-grasp task accuracy
```

下一步可以把 hybrid 做成正式 real-time retarget loop，再加入 table/MoveIt2 collision guard。
