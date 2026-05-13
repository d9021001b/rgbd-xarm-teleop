# MIRROR-style Parallel Candidate IK 復刻與 Hybrid 比較

建立日期：2026-05-13

## 目的

這一版參考 `research_papers/human_motion_retargeting_rgbd_smplx_xarm/pdfs/2603_mirror_parallel_differential_ik.pdf` 的核心思路：

```text
多個 differential IK candidate 並行求解
-> 用 task-space continuation alpha 控制每個候選步幅
-> 用 progress certificate 過濾沒有讓目標函數下降的候選
-> 在安全候選中選最大 alpha，讓動作既穩又能往目標前進
```

原 MIRROR paper 是 GPU 加速的 distributed / batched constrained IK，並包含 self-collision D-CBF。這裡復刻的是可在本專案 CPU / Docker 環境穩定重跑的 **MIRROR-style candidate selection baseline**，不是完整 GPU QP/CBF 實作。

## 對應到本專案的方法

輸入仍然是同一條 fitted SMPL-X right-arm trajectory：

```text
SMPL-X shoulder / elbow / wrist per frame
-> xArm7 base frame workspace mapping
-> MIRROR-style candidate IK
-> xArm7 joint trajectory
-> TelePreview gate
```

每幀在 controller blend-in 後會做：

1. 從上一幀 `q_t` 和目前 TCP 位置開始。
2. 建立 alpha grid：

   ```text
   [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
   ```

3. 對每個 alpha 產生 continuation target：

   ```text
   x_alpha = x_current + alpha * (x_target - x_current)
   ```

4. 每個 candidate 使用同一個 hierarchical functional IK：

   ```text
   TCP position hard priority
   -> forearm projected-angle correction
   -> upper-arm projected-angle correction
   -> upper/lower included-angle correction
   ```

5. candidate 解完後，不只看 continuation target，而是回頭用 final target 評分：

   ```text
   J =
     w_p * TCP_error / tolerance
   + w_f * forearm_error
   + w_u * upper_arm_error
   + w_a * included_angle_error
   + w_d * ||q_candidate - q_previous||
   + w_s * anti_self_insertion_penalty
   ```

6. 若 candidate 滿足安全條件，且相對上一幀讓 `J` 至少下降 `eta`，就視為 accepted。
7. 若有 accepted candidate，選 alpha 最大者；若沒有，退回選安全且 score 最低者。

這保留 MIRROR 的精神：不是每幀只相信單一路徑，而是同時試「保守小步」和「大步追目標」，再用進展與安全性選解。

## 實作位置

- Solver：`docker/retarget-mirror-parallel-candidate.py`
- Config：`configs/mirror_parallel_candidate_hybrid.json`
- 本次輸出：`recordings/mirror_parallel_candidate_20260513_220521/`

設定重點：

| 參數 | 值 | 意義 |
|---|---:|---|
| `parallel_workers` | 6 | 同幀並行評估 6 個 alpha candidate |
| `candidate_iterations` | 10 | 每個 candidate 的局部 IK 迭代數 |
| `lyapunov_eta` | 0.002 | candidate 至少要讓 score 下降的幅度 |
| `position_score_weight` | 2.0 | final TCP error 的評分權重 |
| `forearm_score_weight` | 1.0 | 前臂功能姿態權重 |
| `upper_arm_score_weight` | 0.8 | 上臂 elbow-swivel 權重 |
| `included_angle_score_weight` | 0.8 | 上臂/下臂夾角權重 |
| `anti_self_score_weight` | 0.25 | 避免 gripper 自插入的懲罰 |

## 本次 dry-run 指標

執行條件：

```text
18 seconds, 10 fps, 181 trajectory points
parallel_workers = 6
seed = SEW + functional hybrid first pose
```

MIRROR-style runtime：

| 指標 | 數值 |
|---|---:|
| runtime | 5.18 s / 181 frames |
| approx time per frame | 28.59 ms |
| mean candidate solve time | 33.98 ms |
| mean selected alpha | 0.598 |
| mean accepted candidates / frame | 1.50 |

Retargeting error：

| 指標 | MIRROR-style parallel candidate |
|---|---:|
| TCP raw position error mean | 5.00 mm |
| TCP raw position error max | 13.99 mm |
| included-angle absolute error mean | 15.20 deg |
| forearm projected mean-max error | 33.42 deg |
| upper-arm projected mean-max error | 51.45 deg |
| TelePreview decision | APPROVE_WITH_WARNINGS |

TelePreview hard gates 全部通過：

| Gate | 數值 | 結果 |
|---|---:|---|
| TCP max error | 13.99 mm <= 30 mm | pass |
| table clearance min | 13.28 mm >= 0 mm | pass |
| joint limit margin min | 0.160 rad >= 0 rad | pass |
| joint step p95 | 0.085 rad <= 0.65 rad | pass |

Soft warnings 來自姿態品質：included-angle mean `15.23 deg` 高於 `8 deg`，upper-arm mean-max `51.33 deg` 高於 `45 deg`。

## 與 SEW + Functional Hybrid 比較

| 指標 | MIRROR-style parallel candidate | SEW + functional hybrid |
|---|---:|---:|
| runtime / 181 frames | 5.18 s | 14.72 s |
| approx time per frame | 28.59 ms | 81 ms |
| TCP mean / max | 5.00 mm / 13.99 mm | 1.54 mm / 22.84 mm |
| included-angle mean | 15.20 deg | 4.42 deg |
| forearm mean-max | 33.42 deg | 25.14 deg |
| upper-arm mean-max | 51.45 deg | 41.53 deg |
| TelePreview decision | APPROVE_WITH_WARNINGS | APPROVE_WITH_WARNINGS |

MIRROR-style 版本的優點：

- 比 hybrid 快很多。
- TCP max error 比 hybrid 小。
- joint step / velocity / acceleration / jerk 都很平順。
- 多 alpha candidate 讓每幀可以在保守與積極步幅間選擇。

MIRROR-style 版本的弱點：

- 姿態相似度比 hybrid 差，尤其 included-angle 與 upper-arm elbow-swivel。
- 目前安全約束只是 z clearance、joint margin、anti-self score，還不是 paper 的完整 D-CBF self-collision QP。
- Python thread pool 在 Docker/CPU 環境有開銷；paper 的真正速度優勢需要 GPU batched QP 或 C++/JAX/Torch vectorization。

## 結論

這個 baseline 的定位很清楚：

```text
MIRROR-style = real-time candidate-selection / branch-recovery seed
Hybrid = functional posture fidelity reference
TelePreview = execution-before-preview safety gate
```

因此最合理的下一步不是用 MIRROR-style 直接取代 hybrid，而是把它當成 real-time seed：

```text
MIRROR-style candidate IK
-> selected safe fast seed
-> lightweight functional/null-space correction
-> TelePreview gate / MoveIt2 collision guard
```

這樣能保留 MIRROR 的快速與 local-minima recovery 優勢，同時補回本專案最重視的「TCP、前臂方向、上臂 elbow-swivel、上/下臂夾角」功能對應一致性。
