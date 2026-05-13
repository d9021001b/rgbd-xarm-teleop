# 可復刻方法與 xArm7 Retargeting 比較設計

建立日期：2026-05-13

目標：從已下載 paper 中挑出能復刻到本專案的具體方法，形成可比較 baseline，讓我們的創新不是只和自己比較，而是和既有 teleoperation / retargeting / RGB-D HMR 方法比較。

## 最值得優先復刻的方法

| 優先級 | Paper / 方法 | 可復刻部分 | 接到我們系統的位置 | 預期價值 |
|---|---|---|---|---|
| A | SEW-Mimic closed-form geometric retargeting | 用 shoulder-elbow-wrist 三點，把人類上臂/前臂方向解析成 7-DoF robot arm 姿態 | 替換或輔助 `retarget-smplx-hand-to-xarm.py` 的每幀 optimization IK | 大幅降低每幀 IK 時間，作為 fast baseline |
| A | OCRA optimization-based customizable retargeting | skeleton error + hand orientation error 的加權最佳化 | 對照我們的 TCP hard + forearm/upper/included-angle objective | 形成最直接的 optimization baseline |
| A | Vision-Based Hand Shadowing RGB-D + DLS IK | MediaPipe landmarks + depth deprojection + robot frame transform + DLS IK + smoothing + sim preview | D455 perception -> right hand/arm trajectory -> xArm7 IK | 對照我們 D455/SMPL-X 中介層是否真的比較穩 |
| A | TelePreview virtual preview / SMPL-X joint mapping | 先在虛擬 robot preview，再允許實機/模擬執行；SMPL-X 作跨平台中介 | Gazebo/MoveIt2 preview gate + SMPL-X trajectory viewer | 提升安全性與可操作性，也是 demo 很有說服力的一層 |
| B | MIRROR parallel differential IK + continuation + self-collision CBF | 多個 IK candidate/seed 並行求解，選擇讓 task error 下降且安全的更新 | 替換現在較慢的 branch recovery / anti-self-insertion | 朝 real-time teleoperation 前進 |
| B | SMPLify-X | 用 2D body/hand/face features + pose prior + collision prior fit SMPL-X | 強化目前 MediaPipe -> SMPL-X fitting | 讓 SMPL-X fitting 更像正式 HMR/mesh fitting |
| B | RGB-D Human Mesh Recovery | RGB-D fusion、depth ranking consistency、SMPL constraint generator | 替代目前單純 MediaPipe 3D world landmark | 用 D455 depth 解決單目深度歧義 |
| C | BodyFusion / DoubleFusion / TexMesh | 單深度相機人體表面融合、double-layer human body prior | 長期強化 RGB-D mesh reconstruction | 太重，不適合短期直接復刻到 xArm7 控制 |
| C | Unsupervised Neural Motion Retargeting | 學習式 motion retargeting | 未來大量資料後訓練 policy/baseline | 需要資料集與訓練流程，短期不如 analytical baseline |

## 與我們現有方法的關係

我們目前的方法是：

```text
D455 / taichi video
-> MediaPipe / RGB-D skeleton
-> SMPL-X right-arm fitted trajectory
-> xArm7 functional retargeting
-> TCP hard constraint
-> forearm / upper-arm / included-angle / joint regularization soft objective
-> MoveIt2 table collision safety
```

這套方法最接近 OCRA，但比 OCRA 多了：

1. SMPL-X 作為人體動作中介。
2. TCP 是 hard priority，前臂/上臂姿態在 null-space 裡做 secondary objective。
3. 額外加入 included-angle matching，避免只看方向而忽略手肘彎曲。
4. 額外加入 table collision / MoveIt2 safety，面向真實機械臂 workspace。

## 建議 baseline 實驗

| Baseline | 實作內容 | 比較指標 |
|---|---|---|
| Current method | 目前 calibrated functional objective | 作為主方法 |
| OCRA-style baseline | hand orientation + arm skeleton weighted objective | 姿態相似度、TCP error、每幀時間 |
| SEW-Mimic-style baseline | shoulder-elbow-wrist closed-form / geometric solver | 每幀時間、姿態相似度、是否可達 |
| RGB-D DLS baseline | MediaPipe + depth deprojection + DLS IK，不經 SMPL-X | 與 SMPL-X 中介比較穩定度 |
| MIRROR-style candidate IK | 多 seed / continuation / safety candidate selection | local minima、self-insertion、解算時間 |
| TelePreview mode | preview-only -> execute gate | 安全性、demo 可解釋性 |

## 我們應該主打的創新比較語句

我們不是單純提出一個 IK solver，而是提出一個面向 xArm7 teleoperation 的跨層 pipeline：

```text
RGB-D / SMPL-X reconstructed human motion
-> functional arm correspondence
-> xArm7 TCP-prioritized null-space retargeting
-> table-aware safety execution
```

和 SEW-Mimic 比：我們更重視 task-space TCP 與桌面任務安全，但目前速度較慢。

和 OCRA 比：我們也做 customizable weighted retargeting，但新增 SMPL-X 中介、included-angle objective、MoveIt2 collision context。

和 MIRROR 比：我們目前還不是 real-time GPU-parallel IK，但可以借它的 parallel candidate / continuation / CBF 思路來改善 branch recovery。

和 RGB-D hand shadowing 比：它直接從手部 keypoints 到 robot IK；我們多了一層 SMPL-X/right-arm functional fitting，可以比較是否提升遮擋與肢體姿態一致性。

和 BodyFusion/DoubleFusion 比：它們主攻人體重建；我們主攻「人體重建之後如何讓 xArm7 功能性模仿」。

## 下一步復刻順序

1. OCRA-style baseline：最容易，因為我們現在已經有 optimization objective。
2. SEW-Mimic-style solver：最有價值，因為能直接對付目前每幀約秒級的瓶頸。
3. RGB-D DLS baseline：建立不經 SMPL-X 的對照組，證明 SMPL-X 中介是否值得。
4. TelePreview mode：把 Gazebo preview gate 做成正式展示流程。
5. MIRROR-style parallel candidate IK：作為高階即時化改造。
