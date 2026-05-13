# SMPL-X 右臂到 xArm7 Teleoperation 對應

## 目標

本專案的 retargeting 不是讓 xArm7 長得像 SMPL-X 右臂，而是讓 xArm7 在 teleoperation 中保留操作者右手的任務意圖：

- 右手/手掌決定柔爪 TCP 的位置與朝向。
- 前臂方向決定柔爪接近物體時的姿態。
- 上臂與肘部只作為冗餘構型偏好，讓 xArm7 的 elbow-swivel 看起來接近人體，但不能破壞 TCP 追蹤。

## URDF 確認

目前 Gazebo 中 xArm7 的主要 chain 為：

```text
link_base
  joint1 -> link1
  joint2 -> link2
  joint3 -> link3
  joint4 -> link4
  joint5 -> link5
  joint6 -> link6
  joint7 -> link7
  joint_eef -> link_eef
  soft_gripper_mount_joint -> soft_gripper_palm
  joint_tcp -> link_tcp
```

`joint_tcp` 是 fixed joint，從 `soft_gripper_palm` 沿本地 z 方向偏移 0.17m 到 `link_tcp`。因此柔爪的任務端點應使用 `link_tcp`，柔爪本體/掌心參考可使用 `soft_gripper_palm` 或 `link7/link_eef`。

目前 `smplx_fit_right_arm_trajectory.json` 已匯出的穩定欄位是 `right_shoulder_world`、`right_elbow_world`、`right_wrist_world`。因此現階段先用 `right_wrist_world` 當手/掌心 proxy；等 SMPL-X hand vertices 或 hand joint 穩定輸出後，應改用 palm center 取代單純 wrist point。

## 建議對應

| SMPL-X 人體量 | xArm7 對應 | 用途 |
| --- | --- | --- |
| `right_wrist` 或手掌中心 | `link_tcp` | Primary objective。硬保 TCP 位置，這是 teleoperation 的主任務。 |
| 右手掌/手腕朝向 | `link_tcp` rotation 或 `soft_gripper_palm` rotation | Primary pose objective。讓柔爪開口/掌心方向對應人手接近方向。 |
| `right_elbow -> right_wrist` 前臂向量 | xArm7 elbow proxy 到 `link_tcp`，目前取 `joint5` origin -> `link_tcp` | Secondary direction objective。讓 distal arm 和柔爪方向跟人體前臂一致。 |
| `right_shoulder -> right_elbow` 上臂向量 | xArm7 shoulder proxy 到 elbow proxy，目前取 `joint3` origin -> `joint5` origin | Null-space posture objective。只調 elbow-swivel，不可犧牲 TCP。 |
| `right_elbow` 點 | `joint5` origin | Secondary elbow proxy。只做構型偏好，不做硬位置約束。 |

## IK 優先順序

合理的 hierarchical IK 應該是：

1. Primary: 保住 `link_tcp` 位置，並保住柔爪/手腕姿態。
2. Secondary: 在 primary null-space 中優化 `joint3 -> joint5 -> link_tcp` 的功能鏈姿態，使它接近 SMPL-X shoulder/elbow/wrist 的三投影角。
3. Safety: 若啟用 MoveIt2，桌面/相機/自碰撞只負責規劃安全路徑，不應改變 SMPL-X 到 TCP 的任務語意。

## 動作一致性角度定義

本專案的「角度接近」不是只看單一 3D 向量夾角，而是看空間姿態向量在三個座標平面的投影角：

- XY 投影角：俯視平面，用來檢查水平橫向/前後方向是否一致。
- XZ 投影角：側視平面，用來檢查伸手高度、下探、上提方向是否一致。
- YZ 投影角：正視平面，用來檢查左右側向與垂直方向是否一致。

對任一肢段向量 `v = p_end - p_start`，先投影到指定平面，再用 `atan2` 取得該平面的姿態角。例如：

```text
angle_xy = atan2(v_y, v_x)
angle_xz = atan2(v_z, v_x)
angle_yz = atan2(v_z, v_y)
```

比較 SMPL-X 和 xArm7 時，應對下列功能向量各自計算三個投影角差：

| 功能向量 | SMPL-X | xArm7 |
| --- | --- | --- |
| 手/TCP 位移 | `right_wrist` 軌跡切線或相鄰幀位移 | `link_tcp` 軌跡切線或相鄰幀位移 |
| 前臂姿態 | `right_elbow -> right_wrist` | `joint5` origin -> `link_tcp` |
| 上臂 / elbow-swivel | `right_shoulder -> right_elbow` | `joint3` origin -> `joint5` origin |

因此動作一致的判斷應是：

```text
TCP position error is reasonable
AND forearm projected angles [XY, XZ, YZ] are close
AND upper-arm/elbow-swivel projected angles [XY, XZ, YZ] are close
AND the time series is continuous
```

這種三投影角定義比單一 3D 夾角更容易定位錯誤：如果只有 XZ 角差很大，通常是高度/下探/上提映射錯；如果 XY 角差很大，通常是桌面平面方向或左右鏡像錯；如果 YZ 角差很大，通常是側向與垂直姿態耦合錯。

## 重要限制

xArm7 的 `joint1..joint3` 不是人體肩膀，`joint5..joint7` 也不是人體手腕骨骼；它們是產生相同末端任務的 7-DOF 機構。因此前臂/上臂只能做「方向與構型相似」，不能要求骨段一一同構。若 elbow objective 太硬，會把 TCP 從手的位置拉走，這不符合 teleoperation。
