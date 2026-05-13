# RGB-D DLS Baseline 復刻與 Hybrid 比較

建立日期：2026-05-13

## 目的

這個 baseline 參考 `research_papers/human_motion_retargeting_rgbd_smplx_xarm/pdfs/2603_vision_based_hand_shadowing_rgbd_ik.pdf` 的核心流程：

```text
RGB-D frame
-> hand landmarks
-> depth deprojection
-> robot-base target
-> damped least-squares IK
-> joint-space EMA smoothing
```

它是「手部/TCP shadowing」baseline，不做 SMPL-X 全臂姿態對應，也不做 SEW elbow-swivel 或 functional null-space correction。

## 我們復刻到本專案的內容

Paper 裡的 RGB-D DLS 方法重點是：

- 用 RealSense D400/D435i 類 RGB-D 相機取得 RGB + depth。
- 用 MediaPipe Hands 取得 21 個 hand landmarks。
- 用 depth map 把 2D landmarks 反投影成 3D camera points。
- 轉到 robot base frame 後，產生 end-effector target。
- 用 Damped Least Squares IK 解每一幀 robot joint。
- 用 joint-space EMA smoothing，paper 使用 `alpha_IK = 0.5`。

本專案目前實作的是 solver baseline 層：

- Config：`configs/xarm7_rgbd_dls_baseline_no_moveit.json`
- Solver：`docker/retarget-smplx-hand-to-xarm.py`
- 模式：`orientation.mode = "rgbd_dls_baseline"`
- DLS 參數：`max_iterations = 100`、`damping = 0.045`、`position_tolerance = 0.001 m`
- Joint smoothing：`joint_ema_alpha = 0.5`

為了和 hybrid 公平比較，本次控制輸入使用同一份 fitted SMPL-X right-arm trajectory，但 solver 只讀 wrist/hand target，不使用 shoulder/elbow 參與控制。shoulder/elbow 只在事後用來算姿態誤差。

## 數學意義

DLS baseline 每一幀解的是：

```text
minimize || J(q) * dq - e_pos ||^2 + lambda^2 ||dq||^2
```

其中：

- `q` 是 xArm7 關節角。
- `J(q)` 是 TCP 對關節角的 Jacobian。
- `e_pos = target_tcp - current_tcp`。
- `lambda` 是 damping，避免 Jacobian 接近奇異點時解爆掉。

它只問一件事：「TCP 能不能到手的位置？」  
它不問：「前臂方向像不像？上臂 elbow-swivel 像不像？手肘彎曲角像不像？」

## 本次 dry-run

輸入資料：

```text
recordings/taichi_mediapipe3d_smplx_20260510_150729/smplx_fit_right_arm_trajectory.json
```

輸出資料：

```text
recordings/rgbd_dls_baseline_20260513_203320/
```

執行條件：

```text
18 seconds, 10 fps, 181 trajectory points, raw reconstructed trajectory timing
```

Solver log 摘要：

| 指標 | RGB-D DLS baseline |
|---|---:|
| runtime | 6.07 s / 181 frames |
| approx time per frame | 33.5 ms |
| DLS iterations mean | 1.2 |
| raw DLS TCP error mean before EMA | 0.0002 m |
| TCP error mean after EMA | 0.0099 m |
| TCP error max after EMA | 0.0487 m |

## 與 SEW + Functional Hybrid 比較

以下使用同一個 analyzer 產生的 `trajectory_comparison_summary.json`：

| 指標 | RGB-D DLS baseline | SEW + functional hybrid |
|---|---:|---:|
| runtime | 6.07 s / 181 frames | 14.72 s / 181 frames |
| approx time per frame | 33.5 ms | 81 ms |
| TCP raw position error mean | 0.00835 m | 0.00154 m |
| TCP raw position error max | 0.04870 m | 0.02284 m |
| included-angle abs error mean | 17.33 deg | 4.42 deg |
| included-angle abs error max | 81.39 deg | 15.99 deg |
| forearm projected mean-max error | 56.30 deg | 25.14 deg |
| upper-arm projected mean-max error | 57.63 deg | 41.53 deg |

## 觀察

RGB-D DLS baseline 很快，而且沒有 EMA 時 TCP 解算本身非常準；但加入 paper-style `alpha_IK = 0.5` 後，快速動作會產生明顯時間落後，所以 analyzer 看到的 TCP error 變大。

更重要的是，DLS baseline 沒有上臂/前臂 objective，所以它可以把 TCP 送到接近目標的位置，但 xArm7 手臂構型未必像人的右臂。這正是 hybrid 改善的地方：

- hybrid 比 DLS 慢，但仍在離線/preview 可接受範圍。
- hybrid 的 included-angle error 明顯較低。
- hybrid 的 forearm projected angle error 明顯較低。
- upper-arm 因為 xArm7 和人體肩肘結構天然不同，仍是比較困難的項目。

## 結論

RGB-D DLS baseline 是很好的「直接手部追蹤」對照組：

```text
RGB-D DLS = fast TCP shadowing baseline
SEW seed = fast elbow-swivel posture seed
functional null-space = teleoperation posture correction
```

因此目前最合理的設計不是用 DLS 取代 hybrid，而是把 DLS 的速度優勢當作 real-time seed，再疊加 SEW / functional correction。也就是：

```text
RGB-D target DLS seed
-> SEW elbow-swivel initialization
-> TCP-hard functional null-space correction
-> MoveIt2/table safety guard
```

這樣可以同時保留 paper baseline 的簡潔與速度，也保留本專案對「功能對應」的姿態一致性。
