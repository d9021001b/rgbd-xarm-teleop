# Human Motion Retargeting / RGB-D SMPL-X / xArm7 Paper Pack

建立日期：2026-05-13

目的：蒐集與目前設計相近的論文，作為判斷創新性與撰寫技術報告的對照組。核心關鍵字包括 RGB-D、SMPL-X/HMR、human mesh fitting、human-to-robot retargeting、teleoperation、optimization IK、null-space IK、MoveIt/避障。

## 已下載 PDF

| 類別 | 檔案 | 來源 | 與我們設計的關係 |
|---|---|---|---|
| Robot retargeting / teleoperation | `pdfs/2603_mirror_parallel_differential_ik.pdf` | https://arxiv.org/pdf/2603.23995 | MIRROR 使用 parallel differential IK 做即時動作模仿，是我們 optimization IK / teleoperation 的直接對照。 |
| Robot retargeting / teleoperation | `pdfs/2602_sew_mimic_closed_form_geometric_retargeting.pdf` | https://arxiv.org/pdf/2602.01632 | SEW-Mimic 強調 closed-form geometric retargeting，可對照我們目前每幀最佳化 IK 的速度瓶頸。 |
| Robot hand / RGB-D / IK | `pdfs/2603_vision_based_hand_shadowing_rgbd_ik.pdf` | https://arxiv.org/pdf/2603.11383 | 使用 RGB-D 視覺輸入和 IK 做機器手 shadowing，和我們 D455 + xArm7 teleoperation 很接近。 |
| SMPL-X / teleoperation assistance | `pdfs/2412_telepreview_virtual_arm_assistance_smplx.pdf` | https://arxiv.org/pdf/2412.13548 | TelePreview 用 SMPL-X/虛擬手臂協助 teleoperation，是我們用 SMPL-X 作中介動作表示的近似參考。 |
| Motion retargeting learning | `pdfs/2406_unsupervised_neural_motion_retargeting.pdf` | https://arxiv.org/pdf/2406.00727 | neural retargeting 對照我們目前 analytical/optimization-based 方法。 |
| Optimization retargeting | `pdfs/2023_ocra_optimization_based_customizable_retargeting.pdf` | https://www.ais.uni-bonn.de/ICRA2023AvatarWS/contributions/ICRA_2023_Avatar_WS_Mohan.pdf | OCRA 是 optimization-based customizable retargeting，很適合比較我們的多目標權重設計。 |
| Robot arm mimicry | `pdfs/2017_mimicry_based_teleoperation_robot_arms.pdf` | https://graphics.cs.wisc.edu/Papers/2017/RMG17/hri17-preprint.pdf | 直接探討 robot arms mimicry-based teleoperation，可比較人臂到機械臂的功能映射。 |
| Geometric retargeting | `pdfs/1909_whole_body_geometric_retargeting_humanoids.pdf` | https://arxiv.org/pdf/1909.10080 | whole-body geometric retargeting；可對照我們只針對右臂/TCP/前臂/上臂做 functional mapping。 |
| Dynamical IK | `pdfs/1909_dynamical_inverse_kinematics_motion_tracking.pdf` | https://arxiv.org/pdf/1909.07669 | 動態 IK motion tracking，可對照我們每幀 DLS/null-space IK 是否可改成更快的動態追蹤。 |
| RGB-D SMPL-X | `pdfs/2103_realtime_rgbd_extended_body_pose_smplx.pdf` | https://arxiv.org/pdf/2103.03663 | RGB-D extended body pose estimation with SMPL-X，對應我們 D455 RGB-D -> SMPL-X fitting 這層。 |
| RGB-D HMR | `pdfs/1911_robust_rgbd_human_mesh_recovery.pdf` | https://arxiv.org/pdf/1911.07383 | RGB-D human mesh recovery，對應用深度資訊改善 HMR 的核心想法。 |
| SMPL-X baseline | `pdfs/2019_smplx_smplifyx_expressive_body_capture.pdf` | https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf | SMPL-X / SMPLify-X 基礎論文，是我們人體 mesh 表示與 fitting 的基準。 |
| Depth human capture | `pdfs/1804_doublefusion_single_depth_human_performance_capture.pdf` | https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_DoubleFusion_Real-Time_Capture_CVPR_2018_paper.pdf | 單深度感測器人體 performance capture，對照 D455 depth/point cloud fitting。 |
| Depth human capture | `pdfs/2017_bodyfusion_single_depth_human_motion_geometry.pdf` | https://openaccess.thecvf.com/content_ICCV_2017/papers/Yu_BodyFusion_Real-Time_Capture_ICCV_2017_paper.pdf | 單 depth camera 人體 motion/geometry capture，對照 RGB-D 逐幀重建。 |
| RGB-D human video mesh | `pdfs/2008_texmesh_rgbd_video_human_texture_geometry.pdf` | https://arxiv.org/pdf/2008.00158 | RGB-D video 中的人體 mesh/texture reconstruction，對照我們從 RGB-D 產生可用軌跡。 |

## 找到但未能直接下載 PDF

以下連結在搜尋中相關，但下載端回傳權限或防爬限制；已記錄，後續可用瀏覽器或學校/機構網路手動補抓。

| 類別 | 來源 | 狀態 |
|---|---|---|
| Service robot teleoperation / D455 / MediaPipe | https://www.mdpi.com/1424-8220/26/2/471/pdf | HTTP 403，未下載。 |
| Multimodal teleoperation / motion mimic | https://www.mdpi.com/2072-666X/14/2/461/pdf | HTTP 403，未下載。 |
| Biomimetic human arm motion generation review | https://www.mdpi.com/1424-8220/23/8/3912/pdf | HTTP 403，未下載。 |
| Human-robot kinematic mapping based on index constraint | https://www.sciencedirect.com/science/article/pii/S0957415824000485/pdf | 可能需要權限或動態 token，未下載。 |
| Toward optimal mapping of human dual-arm motion to humanoid motion | https://journals.sagepub.com/doi/pdf/10.1177/1729881418757377 | 可能需要權限，未下載。 |

## 初步創新性觀察

1. 既有 retargeting 論文多半聚焦於 humanoid/robot hand/whole-body，或是單純 robot arm mimicry；我們的重點是 xArm7 這種 7 軸協作臂的 functional teleoperation mapping。
2. 既有 SMPL-X/RGB-D 論文多半停在人體 mesh recovery；我們把 SMPL-X right arm trajectory 接到 xArm7 IK objective，是感知重建到機械臂動作的跨層設計。
3. TelePreview/MIRROR/SEW-Mimic 是最接近的三篇，需要重點研讀。若要凸顯創新，應強調「D455/SMPL-X reconstructed trajectory -> xArm7 functional objective -> TCP hard constraint + forearm/upper-arm/null-space soft matching + table collision/MoveIt2 safety」這條完整 pipeline。
4. 目前我們的瓶頸是每幀最佳化 IK 約秒級；SEW-Mimic 和 MIRROR 可作為改進方向，考慮 closed-form 或 differential IK caching，讓離線校正方法往即時 teleoperation 靠近。
