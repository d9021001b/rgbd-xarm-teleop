# TelePreview Preview Gate 復刻與 Hybrid 比較

建立日期：2026-05-13

## 目的

這一版參考 `research_papers/human_motion_retargeting_rgbd_smplx_xarm/pdfs/2412_telepreview_virtual_arm_assistance_smplx.pdf` 的 preview-then-execute 思路。

TelePreview 的核心不是另一個 IK solver，而是一個 teleoperation 狀態機：

```text
Preview Mode
-> 使用者只控制虛擬 robot，先看結果
-> Gate / Align Mode
-> 擷取最後確認的 preview pose / trajectory
-> 規劃或 retime 後才允許實體 robot 執行
```

Paper 用 AR overlay、AprilTag calibration 與虛擬 robot 對齊真實 robot。本專案目前把這層復刻成 Gazebo digital-twin preview gate：先在模擬/FK 裡檢查 hybrid trajectory，通過 gate 後才建議送往 robot 或 MoveIt2。

## 復刻內容

實作位置：

- Script：`docker/telepreview-preview-gate.py`
- Auto repair：`docker/telepreview-auto-gate-retime.py`
- Retiming：`docker/telepreview-retime-trajectory.py`
- Config：`configs/telepreview_gate_hybrid.json`
- 本次輸出：`recordings/telepreview_gate_hybrid_20260513_211027/gate/`

Gate 分成兩類：

| 類型 | 意義 | 失敗時 |
|---|---|---|
| Hard gate | 直接安全條件，例如 TCP error、桌面 clearance、URDF joint limit、主要 joint step p95 | `BLOCK`，留在 preview，不執行 |
| Soft gate | 品質/平順性警告，例如接近 joint limit、最大單幀跳動、速度/加速度/jerk、功能姿態誤差 | `APPROVE_WITH_WARNINGS`，可執行但建議 retime 或重新求解 |

## Gate 指標

這次 gate 使用目前最佳的 SEW + functional hybrid 軌跡：

```text
recordings/sew_functional_hybrid_20260513_195310/xarm7_sew_functional_hybrid_trajectory.json
```

Gate 結果：

```text
decision = APPROVE_WITH_WARNINGS
hard_pass = true
soft_pass = false
```

Hard gates 全部通過：

| Gate | Value | Threshold | 結果 |
|---|---:|---:|---|
| TCP max error | 0.02284 m | <= 0.030 m | pass |
| Table clearance min | 0.01019 m | >= 0.000 m | pass |
| Joint limit margin min | 0.000 rad | >= 0.000 rad | pass |
| Joint step p95 | 0.5599 rad/frame | <= 0.650 rad/frame | pass |

Soft warnings：

| Warning | Value | Threshold | 意義 |
|---|---:|---:|---|
| Joint limit margin warning | 0.000 rad | >= 0.025 rad | 有幾幀貼到 URDF joint limit，建議重新求解或加 joint-center regularization |
| Joint step absolute max | 5.841 rad/frame | <= 0.900 rad/frame | 第 0 幀到第 1 幀有很大的初始化跳動，建議 retiming 或從 prepose 啟動 |

其餘 soft quality gates 通過：

| 指標 | Value | Threshold |
|---|---:|---:|
| Joint velocity p95 | 5.60 rad/s | <= 6.00 rad/s |
| Joint acceleration p95 | 76.12 rad/s² | <= 85.00 rad/s² |
| Joint jerk RMS | 426.45 rad/s³ | <= 500.00 rad/s³ |
| Included-angle mean error | 4.43 deg | <= 8.00 deg |
| Forearm mean-max error | 25.08 deg | <= 35.00 deg |
| Upper-arm mean-max error | 41.41 deg | <= 45.00 deg |

## 與 Hybrid 直接執行的比較

Hybrid 本身回答的是：「SMPL-X right arm 能不能 retarget 成 xArm7 joint trajectory？」

TelePreview preview gate 回答的是：「這條 trajectory 在送出去之前，有沒有安全或可執行性問題？」

| 項目 | Direct Hybrid | TelePreview-gated Hybrid |
|---|---|---|
| Retarget solver | SEW seed + functional null-space | 同一條 hybrid trajectory |
| TCP mean / max error | 1.54 mm / 22.84 mm | 同上，gate 驗證通過 |
| Included-angle mean error | 4.42 deg | 4.43 deg，gate 驗證通過 |
| Table clearance | 未形成執行 gate | min 10.19 mm，通過 |
| Joint limit | 未形成執行 gate | 有貼邊 warning |
| Initial jump | 不一定在 retarget metrics 中顯眼 | 明確 warning，建議 prepose/retime |
| 執行決策 | 直接送軌跡 | `APPROVE_WITH_WARNINGS` |

## 結論

TelePreview preview gate 沒有取代 hybrid solver，而是補上執行前的安全與品質檢查層：

```text
SMPL-X / RGB-D fitted trajectory
-> SEW + functional hybrid retargeting
-> TelePreview digital-twin preview gate
-> retime / MoveIt2 / execute
```

這次比較顯示 hybrid 的功能對應品質已經足夠好，hard safety 也通過；但 preview gate 抓到兩個 hybrid 指標本身不會強調的問題：關節貼近 limit、初始幀跳動太大。下一步若要接近 TelePreview paper 的完整精神，應該把 gate 後的 `APPROVE_WITH_WARNINGS` 自動接到 retiming 或 prepose planner，而不是直接執行原始 dense trajectory。

## Auto Prepose + Retiming

已完成自動流程：

```text
original trajectory
-> TelePreview gate
-> APPROVE_WITH_WARNINGS
-> prepose hold + joint branch smoothing + step-limited retiming
-> TelePreview gate again
-> APPROVE
```

一鍵腳本：

```bash
python3 docker/telepreview-auto-gate-retime.py \
  --xarm-json recordings/sew_functional_hybrid_20260513_195310/xarm7_sew_functional_hybrid_trajectory.json \
  --retarget-config configs/xarm7_sew_functional_hybrid_no_moveit.json \
  --gate-config configs/telepreview_gate_hybrid.json \
  --comparison-summary recordings/sew_functional_hybrid_20260513_195310/comparison/trajectory_comparison_summary.json \
  --out-dir recordings/telepreview_auto_gate_retime_hybrid
```

本次輸出：

```text
recordings/telepreview_auto_gate_retime_hybrid_20260513_214806/output/
```

結果摘要：

| 指標 | 原始 hybrid | Auto prepose + retimed |
|---|---:|---:|
| Gate decision | APPROVE_WITH_WARNINGS | APPROVE |
| Points | 181 | 209 |
| Seconds | 18.0 s | 20.8 s |
| TCP max error | 0.02284 m | 0.02925 m |
| Joint step abs max | 5.841 rad/frame | 0.449 rad/frame |
| Joint step p95 | 0.560 rad/frame | 0.393 rad/frame |
| Joint velocity p95 | 5.60 rad/s | 3.93 rad/s |
| Joint acceleration p95 | 76.12 rad/s² | 38.38 rad/s² |
| Joint jerk RMS | 426.45 rad/s³ | 107.03 rad/s³ |

這裡有一個很重要的設計取捨：retiming 插入的 transition frames 是「安全抵達」軌跡，不再宣稱逐幀對應原始人體動作時間軸；真正的人體 retarget fidelity 仍在原始 source keyframes 上檢查。這比較接近 TelePreview paper 的 align mode：使用者先 preview，確認後系統生成一條安全、平順的執行軌跡。
