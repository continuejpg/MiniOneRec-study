# docs/experiments/E4_grpo_negative_results.md — GRPO / R1 / R2 负向实验

> **本文件由 Phase M 新增，位于影子树中。** 它是 RL 阶段的**单一入口说明**：真实训练流程、
> reward 与 advantage 的精确算术、五个实验版本的差异、以及 R1/R2 的研究逻辑链与边界。
> 复用既有台账（`docs/notes/experiment_summary.md` §4.4/§6/§7/§12 与
> `docs/provenance/rl_posttraining_results.md`），**不复制**其中的历史记录。
> **本文件不引入任何新数值、不新增命令、不修改任何脚本。**
> **阅读时间：约 8–12 分钟。** 只看结论请跳到 §5 与 §7。
>
> **姊妹文档：** [`E2_shuffled_sid.md`](E2_shuffled_sid.md) · [`E3_seq_only_ablation.md`](E3_seq_only_ablation.md)

---

## 0. 一句话结论

> **在固定的 17.5k GRPO 子集、`ranking` reward 与当前训练配置下，GRPO 没有超过 SFT 的 Top-K 质量。**
> **R1 把零优势组比例从 ~70 % 降到 0，质量却**崩溃**；R2 把零优势比降到 0.475，质量**仍**低于 SFT 与旧 GRPO。**
> **⇒ 「零优势比」**既不是充分条件、也不是必要条件**：把它降到 0 反而崩溃（R1），把它降下来也不足以提升质量（R2）。**
> **⇒ 但它**并非无贡献**：零优势组的 advantage 恒为 0，**对梯度恰好贡献 0** —— 见 §6.5 的精确表述。**
> **⇒ R1 崩溃的**因果**未建立；R2 退化的原因 **UNKNOWN**。两处都是**假设**，不是结论。**

---

## 1. GRPO 实际训练流程

```
patches/grpo_*.sh
  └─ nohup accelerate launch --num_processes 1 rl.py … > "$OUT/train.log" &
        │   一律 --model_path = runs/industrial_sft/final_checkpoint（P0 clean SFT）
        ▼
rl.py  train()   (L35-625)
  ├─ L90-94    读 info_file → item_name / item2id
  ├─ L97-113   构造 3 个数据集：SidDataset(G1) + RLTitle2SidDataset(G2) + RLSeqTitle2SidDataset(G3)
  ├─ L119-154  按 sample_id 冻结 17 516 行子集（G2 永远全量）
  ├─ L156-161  ConcatDataset → HF Dataset → shuffle(seed)
  ├─ L171-258  【仅 R2】route cache → 给 HARD 样本的 prompt 追加第一个 GT SID token
  ├─ L274-318  按稳定 sample_id 绑定 reward 目标（id2target / id2task）
  ├─ L368-370  加载 policy + tokenizer
  ├─ L388-518  定义 reward 函数
  ├─ L522-566  按 --reward_type 选择
  ├─ L575-600  GRPOConfig
  └─ L601-624  ReReTrainer(...).train() → save_pretrained
        │
        ▼
minionerec_trainer.py  ReReTrainer  (L124-1306)
  ├─ L486-511  GenerationConfig(num_beams=G, num_return_sequences=G)   ← beam_search=True
  ├─ L633-747  prefix trie 约束 + ConstrainedLogitsProcessor + TemperatureLogitsWarper
  ├─ L827-840  rollout：dedup prompt → generate（G = 16 条候选/样本）
  ├─ L946-952  reference 模型 per-token logps
  ├─ L1017-1045 逐 reward 函数求值 → 等权求和
  ├─ L1048-1113 group 内 mean/std → advantage
  ├─ L1179-1213 policy loss + KL +（R2）prefix CE
  └─ L1223-1247 指标聚合 → 日志
```

**★ 数据结构（每个样本）：** `{prompt, completion(=GT SID), sample_id, task_type}`。
**★ 每个 prompt 采样 16 条输出（G = 16），beam 搜索 + 温度 1.0。**
**★ rollout **也**受 SID 前缀 trie 约束 ⇒ 输出永远是合法 SID（台账记 `LegalRate 100%`）。**

---

## 2. 关键源码位置

| 主题 | 位置 |
|---|---|
| GRPO 主入口 | `rl.py:35-625`（`train()`） |
| 数据集构造 | `rl.py:97-113`；`SidDataset` = `data.py:364-420` |
| 冻结子集 | `rl.py:119-154` |
| 目标绑定（防错配） | `rl.py:274-318`、`_resolve_targets` `:347-363` |
| reward 权重向量 | `rl.py:388-389` |
| `ndcg_rule_reward` | `rl.py:393-416` |
| `rule_reward` | `rl.py:418-428` |
| `rere_rank_reward` | `rl.py:450-470` |
| `exact_match_reward` | `rl.py:539-562` |
| reward 选择 | `rl.py:522-566` |
| `GRPOConfig` | `rl.py:575-600` |
| ReRe 纯算术 | `rere_reward.py:39-94` |
| R2 精确匹配算术 | `rl_reward.py:27-63` |
| R2 prefix CE | `rl_prefix_ce.py`；调用 `minionerec_trainer.py:981-1000, 1199-1204` |
| R2 路由生成 | `rl_router.py` |
| **advantage 计算** | `minionerec_trainer.py:1048-1113`（公式在 **`:1105`**） |
| 零优势诊断 | `minionerec_trainer.py:1051-1101` |
| **KL** | `minionerec_trainer.py:1182` |
| **policy loss** | `minionerec_trainer.py:1186-1197` |
| 约束解码 | `LogitProcessor.py:26-93`（**掩码赋值在 `:88`**） |
| 启动器 | `patches/grpo_{baseline,short025,fast025,rere_rank025,r21}.sh` |

---

## 3. Reward 与 Advantage 的计算示例

### 3.1 上游 `ranking` = `rule_reward` + `ndcg_rule_reward`（两函数**求和**）

**权重向量（`rl.py:388-389`，G = 16，`Σ 1/log2(i+2) = 6.105999`）：**

| 组内名次 r | `ndcg_rewards[r]` |
|---|---|
| **0** | **−0.163773**（惩罚最大） |
| 1 | −0.103329 |
| 2 | −0.081887 |
| … | … |
| 15 | **−0.040067** |

**组逻辑（`rl.py:408-414`）：**

| 组内情况 | 合计 reward | 组内 std | 后果 |
|---|---|---|---|
| **有 GT 命中** | 命中 **+1.0**，其余 **−0.1638 … −0.0401** | 0.267174 | ✅ 有学习信号 |
| ★ **无 GT 命中** | ★ **全 0.0** | ★ **0.000000** | ★ **全组 advantage = 0 ⇒ 该步空转** |

**⇒ 这就是"零优势组"的字面来源。台账实测：orig 69.856 % / opt 70.153 % / 2.0ep 69.277 %。**

### 3.2 Advantage（`minionerec_trainer.py:1105`）
```
A_i = ( r_i − mean(r over group) ) / ( std(r over group) + 1e-4 )
```
* `std` 为**无偏**样本标准差（除以 G−1）；
* 分母加 `1e-4` ⇒ 组内 reward 完全相同时 `A_i = 0`（**不会 NaN**）；
* `A_i = 0` ⇒ 该候选对 policy loss 的贡献为 0。

### 3.3 R1 的 ReRe reward（`rere_reward.py:39-79`）
```
Eq.7  R_rule_i     = +1 if GT else 0
Eq.8  R_hat_rank_i = 0 if GT, else −1/log2(r+2)
Eq.9  R_rank_i     = −R_hat_rank_i / Σ_j R_hat_rank_j    ← 除以【本组自己的和】
      R_i          = R_rule_i + R_rank_i
```

**★ 关键算术（本轮复算）：当**无 GT 命中**时，`Σ raw = −6.105999`，于是**

| 名次 r | `R_rank`（无 GT 时） |
|---|---|
| **0** | ★ **−0.163773** |
| 1 | −0.103329 |
| 15 | −0.040067 |

**组内 std = 0.031992 ≠ 0 ⇒ advantage 非零。**
**★ 这就是"零优势比 = 0"的确切原因：Eq.9 的自归一化把"全零"变成了一个**按名次单调**的向量。**

**★★ 两个总体是**分离**的（Phase N 补算）：**

| 组类型 | 组内 std |
|---|---|
| **无 GT 命中** | **恒为 0.031992**（与样本无关的常数） |
| **有 GT 命中**（GT 在名次 0…15 任一处） | **0.267454 … 0.268875** |

**⇒ 因为 `min(有 GT) = 0.267454 > 0.031992`，任何 std < 0.267454 的组**必然**是无 GT 组。**
**⇒ 因此台账记录的 R1 `group_reward_std_mean` **中位数 = 0.031992** 在数学上**强制**"至少一半的组无 GT 命中"——这是一条**严格的**推断，不是类比。**

**★ 与另一条独立证据互相印证：** 台账同时直接测得 **GT 命中组 = 340 / 4 379 = 7.76 %**（由 `sum(R_total) = n_hits` 反推）⇒ **无 GT 占 92.24 %**。
**★ 两处口径不同（一个是 std 分布的中位数推断，一个是命中计数），结论同向。第二轮复算还给出第三个一致性检查：** 用台账的均值 0.0503 与两个总体的 std 值反解 GT 命中比例：
`0.031992·(1−p) + ≈0.2686·p = 0.0503 ⇒ p ≈ 7.7 %` —— 与 7.76 % 吻合。

**★ 但必须限定：`group_reward_std_mean` 是**单组内** reward 的样本标准差；每个优化步只处理**一个** prompt 组（见 §4.3），所以它是**逐步**记录的。**
**（R1/R2 **没有**逐 step 的 `_EXTRACTED_metrics` 文件——该目录只有 `grpo_{short025,fast025,baseline,smoke}_metrics.txt`。因此上述 R1 数值是**台账转述**，本轮**未从原始逐步日志独立复算**。）**

### 3.4 ★ 这个向量把梯度推向哪个方向（**Phase N 更正**）

> 原因是（**这是假设**）Eq.9 把 rank 项**在本组内归一化**，所以在没有正确答案的组里，advantage 变成「**名次越靠前惩罚越重**」⇒ 梯度**压低模型自己最自信的候选、抬高最不自信的**——和「哪个是对的」无关。

**① importance 因子恒等于 1**（`minionerec_trainer.py:1186`）：
`torch.exp(per_token_logps − per_token_logps.detach()) ≡ 1` ⇒ 每个 token 的 loss 就是 `−A_i`。

**② advantage 与 reward 是**同一组内**的正仿射变换**（`:1105`）：
`A_i = (r_i − mean)/(std + 1e-4)` ⇒ **`argmax(advantage) == argmax(reward)`**（本轮数值验证：相关系数 = 1.0000000000）。

**③ 无 GT 组（G = 16）的完整符号表：**

| 名次 r | reward | advantage `A` | loss 项 `−A` | 概率被推向 |
|---|---|---|---|---|
| **0**（模型最自信） | −0.163773 | **−3.155704** | **+3.155704** | ★ **降低** |
| 1 | −0.103329 | −1.272257 | +1.272257 | 降低 |
| 2 | −0.081887 | −0.604094 | +0.604094 | 降低 |
| 3 | −0.070533 | −0.250321 | +0.250321 | 降低 |
| 4 | −0.063356 | −0.026679 | +0.026679 | 降低 |
| 5 | −0.058337 | +0.129712 | −0.129712 | 提高 |
| … | … | … | … | … |
| **15**（模型最不自信） | −0.040067 | **+0.699010** | **−0.699010** | ★ **提高** |

（11 个候选被提高、5 个被降低。）

**④ 结论：** 在无 GT 的组里，梯度并不"强化模型的 top-1"。它**压低模型自己排在最前的候选、抬高排在最后的候选** —— **方向与排序目标相反**。
**★ 因此正确的表述是：** Eq.9 在无 GT 组产生的是**对模型自身排序的系统性反惩罚（flattening / inverse-ranking）**，而不是"自我强化"。
**★ 有 GT 的组则相反：** GT 的 advantage 在**全部 16 个可能名次上都是最大值**（本轮验证 16/16），所以 GT 总是被**推得最多**——这一条才符合预期。

**★ 与坍缩现象的关系（仍为**假设**）：** 92.24 % 的组在反惩罚自身 top-1，这与"不同 top-1 预测从 1 059 种塌到 99 种、最高频占 33.6 %"方向一致（输出分布被推离原策略的高分区）。**但台账明确：无对照 run ⇒ **因果未建立**。**

### 3.5 Policy loss 与 KL（`minionerec_trainer.py:1182-1197`）
```python
per_token_kl   = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1      # 恒非负，d=0 时为 0
per_token_loss = torch.exp(logp - logp.detach()) * advantages            # ★ 该因子恒等于 1
loss           = -(per_token_loss - beta * per_token_kl) 按 completion 内 token 平均
```
* **`exp(logp − logp.detach()) ≡ 1`** ⇒ **没有 importance ratio、没有 clip** ⇒ 这是**无 ratio 的 REINFORCE-with-KL 变体**；
* `β = 1e-3`，且**全部启动器传 `--sync_ref_model True`** ⇒ 参考模型**周期性同步**到当前策略；
* KL 项写成 `e^d − d − 1`：**恒非负**、对 `d<0` 也惩罚、且**指数放大正尾部** ⇒ 与台账观测到的极端 KL（max 1e8–1e11）方向一致。

---

## 4. 实验版本对照

| # | 名称 | 训练轮数 | 步数 | `--reward_type` | 评测 split | 证据 |
|---|---|---|---|---|---|---|
| **V1** | GRPO 2.0 ep | 2 | **35 032** | `ranking` | test | **E3** |
| V1′ | GRPO 1.5 ep | 2（中间 ckpt） | 26 274 | `ranking` | test | E3 ⚠ 与 V1 **不独立** |
| **V2** | GRPO 0.25 ep **原始** | 0.25 | 4 379 | `ranking` | test | ★ **E4（不可复算）** |
| **V3** | GRPO 0.25 ep **优化** | 0.25 | 4 379 | `ranking` | test | **E1′** |
| **V4** | **R1**（ReRe rank） | 0.25 | 4 379 | **`rere_rank`** | test | **E1′** |
| **V5** | **R2**（reachability） | 0.25 | 4 379 | **`r21_exact`** | ★ **validation** | **E1′** |

**五者**共同**：同一 P0 SFT 起点、同一 17.5k 冻结子集、G = 16、beam_search True、`lr 1e-5`、`β 1e-3`、`train_batch_size 16`、`grad_accum 1`。**
**五者**不同**：轮数（V1 vs 其余）、reward 类型（V4/V5）、prompt 内容（V5）、额外损失项（V5）、评测 split（V5）、评测间隔（V2 vs V3）。**

**★ 唯一干净的**单变量对比**是 **V3 → V4**（仅 `--reward_type` + `--output_dir` 不同）。**
**★ 训练预算**不可比**：V1 = 35 032 步，V2–V5 = 4 379 步（1/8）。**
**★ `test`（4 533）与 `validation`（4 532）**不可混用**。**

### 4.1 ★ 步数口径：4 379 与 35 032 到底是什么（Phase N 补算）

**★ 必须先区分四个不同的量：**

| 量 | 0.25 ep（V2–V5） | 2.0 ep（V1） |
|---|---|---|
| 冻结子集中的 **unique prompt 数** | 17 516 | 17 516 |
| 每个优化步处理的 **prompt 数** | **1** | 1 |
| 每步生成的 **序列数** | **16**（= G） | 16 |
| **optimizer 更新次数 = `global_step`** | **4 379** | **35 032** |
| 累计生成的序列数 | 70 064 | 560 512 |
| **epoch 数** | **0.25** | **2.0** |

**★ 为什么每步只有 1 个 prompt：** GRPO 里 `per_device_train_batch_size` 计的是 **prompt 数**（每个 prompt 展开成 G 条序列），而 `num_generations = 16 = train_batch_size`。
⇒ **每步恰好 1 个 prompt**，因此 `prompt 数 = optimizer 步数`，`epoch = steps / 17 516`。

**★ 三条独立交叉验证（全部精确成立）：**
```
检查点/provenance 的 epoch 标注           步数 / 17 516
  0.25 epoch  →  4 379                  4 379 / 17 516 = 0.2500  ✅
  1.50 epoch  →  26 274                26 274 / 17 516 = 1.5000  ✅
  2.00 epoch  →  35 032                35 032 / 17 516 = 2.0000  ✅
```
**且与 `checkpoint-4379/trainer_state.json` 的记录一致：`global_step=4379`、`epoch=0.25`、`max_steps=4379`（本轮实读）。**

**★ 与 SFT 的口径差异（不可混用公式）：** SFT 用 `ceil(ceil(N/micro_batch)/grad_accum) × epochs`；GRPO **不能**用该式（它的 batch 单位是 prompt，且每步展开 G 条序列）。
**★ 一个自洽性检查：** 台账记 R2 的 `GT-hit candidate fraction = 3.2813 % = 2 299 / 70 064`，其中分母 **70 064 = 4 379 × 16** —— 与本表"累计生成序列数"**精确吻合**。

**★ 另一条量级警示（Phase N 实读）：** 全部 GRPO 运行的 `num_groups = 1.0`（每一步只有一个组），因此 `zero_advantage_group_ratio` **逐步是 0/1 指示量**，其**均值**才等于"零优势组的占比"。

**⇒ 正确表述："4 379 次 optimizer 更新 / 4 379 个不同 prompt / 0.25 epoch"，不是"4 379 个训练样本"。**

### 4.2 代码固定、启动器不覆盖的关键超参（只能读 `rl.py:575-600`）
| 参数 | 值 |
|---|---|
| `max_completion_length` | **128** |
| `warmup_ratio` | 0.03 |
| `max_grad_norm` | 0.3 |
| `optim` | `paged_adamw_32bit`（★ 与 SFT 的 `adamw_torch` **不同**） |
| `lr_scheduler_type` | `cosine` |
| `save_total_limit` / `logging_steps` | 2 / 1 |

**★ `--mask_all_zero` 是**死参数**：`rl.py:63` 只在签名里出现，全仓库无使用点（但每个启动器都传它）。**

---

## 5. 主要负向结果

### 5.1 test split（V1–V4）
| 臂 | **HR@20** | **NDCG@20** |
|---|---|---|
| Clean SFT（对照） | **0.19832341** | **0.11786798** |
| V2 0.25 ep 原始 | 0.17626296 | 0.10885475 |
| V3 0.25 ep 优化 | 0.17538054 | 0.10824989 |
| V1 1.5 ep | 0.16170307 | 0.10470676 |
| V1 2.0 ep | 0.16236488 | 0.10447064 |
| **V4 R1** | ★ **0.05912199** | 0.05203455 |

**⇒ **每一个** GRPO 臂都低于 SFT。R1 是最差的（**−70.19 %**）。**

### 5.2 validation split（★ 仅此三行可互比）
| 臂 | **HR@20** | **NDCG@20** |
|---|---|---|
| P0 SFT | **0.18314210** | **0.12011306** |
| V3 optimized GRPO | 0.17122683 | 0.11443492 |
| **V5 R2** | **0.15600177** | **0.10630817** |

**⇒ 排序 SFT > optimized GRPO > R2；R2 在**每一个 K**（1/3/5/10/20）上都落后。**

### 5.3 效率优化的正面结果（唯一明确的正向结论）
| 指标 | V2 原始 | V3 优化 | 变化 |
|---|---|---|---|
| wall-clock | 6 111.57 s | 1 598.19 s | **−73.8 %** |
| throughput | 0.717 step/s | 2.74 step/s | **3.82×** |
| peak VRAM | 7.628 GB | 7.628 GB | 不变 |
| HR@20 | 0.17626296 | 0.17538054 | −0.088 pp |

**⇒ 「**约**保持质量」而非无损；单 seed；台账明确**不假设**两次运行的优化轨迹相同。**

### 5.4 坍缩与集中度
| model | distinct top-1 | 最高频占比 |
|---|---|---|
| P0 SFT（test） | 1 059 | — |
| V3 optimized GRPO（test） | 1 252 | — |
| ★ **V4 R1（test）** | ★ **99** | ★ **1 522/4 533 = 33.6 %** |
| P0 SFT（validation） | 1 134 | 498（10.99 %） |
| V3（validation） | 1 339 | 664（14.65 %） |
| **V5 R2（validation）** | 1 223 | **952（21.01 %）** |

**⇒ R1 严重坍缩；R2 无严重坍缩，但 top-1 集中度升到 SFT 的 1.91 倍。**
**★ 两臂 `LegalRate` 均 100 % ⇒ 坍缩**不是**非法输出造成的。**

---

## 6. R1 / R2 的研究逻辑链

### 6.1 起点：为什么怀疑零优势比
GRPO 各臂均低于 SFT，且诊断显示 **≈70 % 的组 std < 1e-6** ⇒ 该步无学习信号。
**⇒ 猜想："七成步空转 ⇒ 学不到东西"。**
**★ 台账明确把它标为 observation，**不用来解释推荐质量**。**

### 6.2 R1：换掉 reward
**改动：** 仅 `--reward_type` → `rere_rank`（**唯一的干净单变量对比**）。
**结果：** 零优势比 **→ 0.0**（4379/4379 步），**HR@20 却崩到 0.05912**。
**★ 机制假设（台账自述 data-consistent, NOT proven causal）：** Eq.9 组内自归一化 ⇒ 无 GT 组（**92.24 %**）落到一个**按名次单调的负 reward 向量** ⇒ 经 advantage 与 loss 符号后**反惩罚模型自身排序**（精确方向见 §3.4）⇒ 与后续观察到的坍缩方向一致。
**⇒ 本轮补充的算术证据：** 无 GT 组的 std = **0.031992**，与台账的 `group_reward_std_mean` **中位数逐位相同** ⇒ 佐证"中位数组 = 无 GT 组"。
**★ 但**因果未建立**：无对照 run（如 GT-only rank term）。**

### 6.3 R2：换路由 + 换 reward
**改动：** `--reward_type r21_exact`（0/1 精确匹配）+ `--r21_enable True` + `--route_cache`。
**机制：** 离线 route cache 把样本分为 NORMAL（27.51 %）/ HARD（72.49 %）；HARD 的 prompt 追加**第一个 GT SID token**；reward 用 `hint + 采样后缀` 与 GT 精确比较；并对 hint token 加**固定系数 0.1** 的 prefix CE。
**★ 必须记录的前置修正 `count_0`：** `ConstrainedLogitsProcessor` 原本首键恒为 `sent[-3:]`，带 hint 后该窗口滑到 hint 上 ⇒ **trie MISS ⇒ 强制 EOS**（探针实测 h=1 → 0 个合法候选）。加 `count_0` 后修正；**默认 0 ⇒ NORMAL 路径逐位不变**。
**结果：** 零优势比 **0.7015 → 0.4750**，**HR@20（validation）仍低于 SFT 与旧 GRPO**。
> 所以最重要的结论是：**降低零优势比既不充分、也不自动有益**——旧 GRPO 带着 70 % 的零优势组，质量却比零优势只有 47.5 % 的 R2 更好；而 R1 把零优势降到 0 反而崩溃。**但它是否构成长训练的瓶颈，我没有做能回答这个问题的对照实验，所以标成 UNKNOWN，而不是「已否定」。**

### 6.4 逻辑链小结
```
GRPO 全线 < SFT  ──► 怀疑零优势比 ~70 %
        │
        ├─ R1：把零优势比降到 0  ──► 质量崩溃（−70.19 %）
        │        ⇒ 假设：Eq.9 在无 GT 组产生"反惩罚自身排序"的梯度（未证因果）
        │
        └─ R2：把零优势比降到 0.475 ──► 质量仍低
                 ⇒ ★ 反例：降零优势比不足以提升质量
                 ⇒ 但两个变化混杂 ⇒ 原因 UNKNOWN
```

### 6.5 ★ 零优势比到底"贡献"了什么（Phase N 收紧表述）

**★ 必须先区分两句话，它们真假不同：**

| 表述 | 判定 | 依据 |
|---|---|---|
| "**零优势比高**导致**质量差**"（把它当**充分**原因） | ★ **已被证据否定** | R1 把零优势比降到 **0.0**，质量反而**崩溃**；R2 降到 **0.475**，质量仍低于旧 GRPO。**⇒ 它不是充分条件。** |
| "把零优势比降下来**就**能提升质量"（把它当**充分**手段） | ★ **已被证据否定** | 同上 |
| "零优势比高**是**质量差的原因之一"（**必要**性方向） | ⚠ **本实验不支持，也未排除** | 全部实验都是**单 seed、无预算匹配**；且**没有任何一个臂**在"零优势比高 + 其他条件等同"下与低零优势臂对照过（R1/R2 **同时**改了 reward） |
| "零优势组**对梯度没有贡献**" | ✅ **严格成立** | `A_i = (r_i − mean)/(std + 1e-4)`；组内 reward 全相等 ⇒ `A_i = 0` ⇒ policy loss 项 `−A_i = 0`。**⇒ 逐步、精确地为 0，不是"近似为 0"。** |
| "零优势比**不是瓶颈**" | ⚠ **表述过强，本报告不再使用** | 只能说"**降低它既不充分、也不自动有益**"。是否构成长训练的**限制因素**，需要**预算匹配**的对照实验才能回答，**本项目的实验不构成该证据**。 |

**★★ 因此本报告的结论收紧为：**
> **「零优势比」在本项目中**既不是提升质量的充分条件、也不是充分手段**；把它降到 0 甚至伴随崩溃。**
> **零优势组对当步梯度的贡献**严格为 0**（数学上精确）。**
> **但它是否构成长期训练的**瓶颈/限制因素**，**本项目没有做能回答该问题的对照实验**（所有 GRPO 臂都是单 seed，且 R1/R2 同时改动了 reward 形状）⇒ **标记为 UNKNOWN，不是"已否定"。**

---

## 7. 实验局限与证据等级

### 7.1 ★ 证据等级（只衡量**预测产物可复算性**）
| 臂 | **预测产物** | 等级 | 可复算历史指标 | 权重/检查点本地可得 | 可重跑 |
|---|---|---|---|---|---|
| V1 / V1′ | `checkpoints/grpo_evidence/eval_grpo_{baseline,step26274}/test_beam20.json` | **E3** | ✅ | `grpo_baseline/final_checkpoint/model.safetensors` ✅ | ⚠ 需 GPU |
| **V2** | ★ **无** | ★ **E4** | ★ **❌ 不可** | ★ **✅ 在**（`grpo_short025/checkpoint-4379/model.safetensors`，1 977 199 248 B） | ⚠ 需 GPU |
| V3 | `freeze_v1_records.tar.gz` → `predictions/optimizedGRPO_test_eval_beam20.json` | **E1′** | ✅（需解压归档） | `grpo_fast025/checkpoint-4379/model.safetensors` ✅ | ⚠ 需 GPU |
| V4 / V5 | `predictions/{R1_test_beam20,R2_valid_predictions}.json`（归档内） | **E1′** | ✅（需解压归档） | ❌ **不在本地**（仅 SHA256 `7ebdf9c1…` / `a983f46b…`） | ❌ |

**★ 三处必须与结论同引：**
1. **V2 缺的是【预测产物】而不是【权重】。** 该 run 的 **checkpoint 与逐步日志都在本地**：
   * `checkpoints/grpo_evidence/grpo_short025/checkpoint-4379/`（含 `model.safetensors` 1 977 199 248 B、`trainer_state.json`、`training_args.bin`、`scheduler.pt`、`rng_state.pth`）
   * `checkpoints/grpo_evidence/grpo_short025/train.log`（4 789 286 B，本轮实读）
   * `checkpoints/grpo_evidence/_EXTRACTED_metrics/grpo_short025_metrics.txt`（4 379 行逐步诊断，本轮实读）
   **⇒ 因此"V2 无 checkpoint"是**错的**；正确说法是"**缺预测产物 `eval_beam20.json`，故 HR@20 = 0.17626296 无法**复算**"。**
   **⇒ 这也意味着 V2 **原则上可以被重新评测**（权重在、`evaluate.py` 在），但那是**新运行**，不是恢复历史产物；且需要 GPU（本机 8 GB，见 §7.5）。
2. **R1/R2 权重不在本地**（仅哈希钉住）⇒ 指标可复算（预测产物在归档内），但**不能重跑**。
3. **R2 用 validation split** ⇒ 与所有 test-split 表**不可混用**。

### 7.2 ★ 证据边界（禁止越界表述）
| ✅ 已验证 | ⚠ 仅假设 | ⛔ UNKNOWN |
|---|---|---|
| 全部 GRPO 臂 < SFT | R1 崩溃的机制（Eq.9 自我强化） | R2 退化的原因 |
| R1 零优势比 = 0.0 且质量崩溃 | R2 退化源于 hint 白送第一层 | R1 崩溃的**因果** |
| R1 严重 Top-1 坍缩（99 / 33.6 %） | 零优势比高反映"模型能力不足" | V2 的数字（不可复算） |
| R2 零优势比 0.4750 且质量仍低 | —— | V2 与 V3 的 LR 轨迹是否相同 |
| **零优势比不是瓶颈**（有反例） | —— | —— |
| R2 无 NaN 但 grad_norm/KL 极端离群 | —— | —— |
| **单 seed** ⇒ **不做任何显著性主张** | —— | —— |

### 7.3 未完成的实验
* **匹配零优势比的对照**（R2 的两个变化分离消融）；
* **GT-only rank reward 对照**（检验 R1 的机制假设）；
* **多 seed**（方差与显著性）；
* **`max_completion_length` / `reward_weights` / prefix CE 系数（0.1）的扫描**；
* R2 的 test-split 评测（**故意未做**，理由：validation 劣势在所有 K 上一致）。

### 7.4 ⛔ 本文件**不**声称的
* ❌ 不声称 GRPO 对推荐无效（**未**改变子集、reward 权重、算法、数据集）；
* ❌ 不声称"零优势比高反而更好"（正确表述见 §6.5）；
* ❌ ★ **不声称"零优势比不是瓶颈"** —— 本项目的实验只能否定它的**充分性**，**没有**做能判定**必要性/瓶颈**的预算匹配对照（Phase N 收紧，见 §6.5）；
* ❌ 不声称 R1 证明了 rank reward 不可用（只测了**一个**具体形式）；
* ❌ 不声称 R2 证明了 reachability hint 有害（**未分离**）；
* ❌ ★ **不声称 Eq.9 在无 GT 组"强化了模型自身的排序"** —— 精确的梯度方向是**反惩罚**（见 §3.4）；
* ❌ 不用通用 GRPO 理论替代本项目的真实归因。

### 7.5 运行环境与本机验证能力（★ 必须与结论同引）
| 项 | 事实 |
|---|---|
| 历史训练硬件 | 项目声明**单卡 RTX 4090（24 GB）**（`README.md`） |
| **本机（审查环境）GPU** | **NVIDIA GeForce RTX 5070 Laptop GPU，8 151 MiB** —— **不是 4090，也不是 24 GB** |
| 本机能否重跑 GRPO | ⚠ **不能**：V2/V3 权重虽在本地，但 8 GB 显存不足以承载该配置的历史峰值显存（台账记 V2/V3 peak **7.628 GB**、R2 peak **16.74 GB**） |
| 本机能否复算历史指标 | ⚠ **仅当预测产物可得**；V2 **无**预测产物 ⇒ 不可；V3/V4/V5 产物在归档内 |
| 本机是否有模型权重 | ✅ **部分**：`grpo_baseline/final_checkpoint`、`grpo_{short,fast}025/checkpoint-4379`、`industrial_sft/final_checkpoint` 均在 `checkpoints/`（各 `model.safetensors` 约 1.98 GB / 0.99 GB） |
| **本轮是否执行训练/推理** | ⛔ **未执行** |
| **本轮是否复算 GRPO 指标** | ⛔ **未执行**（E1′ 产物在冻结归档内，按治理规则未解压） |

**★ 结论：本地具备**读取**历史证据的能力，**不具备**重跑这些 GRPO 配置的硬件条件。**

---

## 8. 一分钟与三分钟面试介绍

### 8.1 一分钟版
> 这个项目里我做了完整的 **GRPO 后训练**，而且**全部是负向结果**，我把它完整报告了出来。
>
> 设定是：从 SFT 的 checkpoint 出发，在一个 **17 516 条**的固定子集上做 GRPO，每个 prompt 采样 **16 条**输出，reward 是"命中给 +1、否则按**组内名次**给一个负分"。
>
> **第一个发现**：所有 GRPO 臂都**低于** SFT —— HR@20 从 0.198 掉到 0.176 / 0.162。同时诊断指标显示 **约 70 % 的组**reward 标准差为 0，也就是 advantage 全 0、**这一步完全没学**。
>
> **第一个尝试 R1**：我换成 ReRe 的 rank reward（Eq.7–9），**零优势比例直接从 ~70 % 降到 0**。但 HR@20 **崩到 0.059**，而且 top-1 预测从 1 059 种**塌到 99 种**，单个 SID 占了 33.6 %。原因是（**这是假设**）Eq.9 把 rank 项**在本组内归一化**，所以在没有正确答案的组里，advantage 变成「**名次越靠前惩罚越重**」⇒ 梯度**压低模型自己最自信的候选、抬高最不自信的**—— 和「哪个是对的」无关。
>
> **第二个尝试 R2**：改成 0/1 精确匹配 reward，并给 72 % 的样本在 prompt 里**提示第一个 SID token**。零优势比降到 0.475，**但质量还是低于 SFT 和旧 GRPO**。
>
> **最重要的结论**：**"零优势比过高"不是瓶颈** —— 旧 GRPO 带着 70 % 的零优势却比 R2 好。这个是**有反例支持的**。
>
> 而 R2 为什么退化，**我不知道** —— 因为我同时改了两件事（reward 形状 + hint 路由），**没有做消融把她们分开**。R1 崩溃的因果也**没有建立**，因为我没有做 GT-only 的对照。

### 8.2 三分钟版

**0:00–0:40　设定与主结果**
> 我在这个项目里做了 GRPO 后训练，从 P0 SFT 的 checkpoint 出发，数据是一个**冻结的 17 516 条**子集，为了严格可比：`seq_rec` 10 000、`title2sid/description2sid` 6 516、`seqtitle2sid` 1 000。每个 prompt 用 beam 采样 **16 条**输出，而且 rollout 也受 SID 前缀 trie 约束，所以输出永远是合法 SID。
>
> 主结果是：**所有 GRPO 臂都低于 SFT**。0.25 epoch 是 0.176、2 epoch 反而更差 0.162，SFT 是 0.198。**唯一明确的正向结果是工程性的**：通过去掉训练中的重复验证和冗余存档，wall-clock 快了 **73.8 %**、吞吐 **3.82 倍**，质量只动 −0.088 个百分点。

**0:40–1:20　诊断：七成步在空转**
> 我加了两个诊断指标。一个是组内 reward 标准差，一个是"零优势组比例"。结果显示**大约 70 % 的组标准差小于 1e-6** —— 也就是 advantage 全 0，**这一步对参数没有任何更新**。
>
> 我看了 reward 代码，找到确切原因：reward 在"这一组 16 条候选**一条都没命中**"的时候，会**整组返回 0**（`rl.py` 里那个 `flag` 分支）。所以零优势比高，本质上是"**这个样本模型完全答不对**"的比例。

**1:20–2:10　R1：把零优势降到 0，但崩了**
> 于是第一个尝试 R1：换成 ReRe 的 rank reward。这个改动**非常干净** —— 和优化版启动器只差 `--reward_type` 一项，是全部实验里**唯一一个严格单变量对比**。
>
> 结果：**零优势比从 ~70 % 直接降到 0.0**，全程 4 379 步都是 0。但 **HR@20 崩到 0.059**，比 SFT 低 **70 %**；而且**输出多样性塌了** —— test split 上不同的 top-1 预测从 1 059 种掉到 **99 种**，最高频的一个 SID 占了 **33.6 %**。
>
> 机制上我给的解释是**假设，不是结论**：Eq.9 把 rank 项**除以本组自己的和**，所以在没有正确答案的组里，reward 是一个只跟**名次**有关的固定负向量、且随名次单调。我把它一路推到梯度符号：advantage 在名次 0 最负（−3.1557），而 loss 是 `−A`，所以**模型最自信的那条候选被压低**，最不自信的反而被抬高——**方向与排序目标相反**。这跟「哪个 SID 正确」无关。**92 % 的组都属于这种情况。**
>
> 我还有一个**独立的算术佐证**：我从公式手算"无 GT 组"的标准差，得到 **0.031992**，而训练日志里 `group_reward_std_mean` 的**中位数正好是 0.031992** —— 逐位相同。这说明中位数组确实就是无 GT 组。

**2:10–2:50　R2：换了 reward 和路由，还是不行**
> 第二个尝试 R2 换了思路：先离线问"正确答案在约束解码下**到得到吗**"，把样本分成 NORMAL 27.5 % 和 HARD 72.5 %；HARD 的 prompt 里**直接给出第一个 GT SID token**，reward 改成 **0/1 精确匹配**，并对 hint token 加一个系数 0.1 的监督 CE。
>
> 这里有一个必须修的前置 bug：约束解码器原本第一次查询 trie 的键是固定的 `sent[-3:]`，加了 hint 之后这个窗口滑到 hint 上，**查不到键就返回空、直接强制 EOS** —— 探针实测 h=1 时合法候选是 0 个。加 `count_0` 把 hint 计为"已生成"就修好了，而且默认 0 让 NORMAL 路径逐位不变。
>
> 结果：**零优势比从 0.70 降到 0.475**，但 HR@20 在 validation 上还是**低于 SFT 和旧 GRPO**，而且是**每一个 K 都低**。

**2:50–3:10　结论与边界**
> 所以最重要的结论是：**"零优势比过高"不是瓶颈**。这是一个**有反例的结论** —— 旧 GRPO 带着 70 % 的零优势组，质量却比零优势只有 47.5 % 的 R2 更好。
>
> 而 R2 为什么退化，我**明确说不知道**：我同时改了两件事（reward 形状 + hint 路由），**没有做消融把它们分开**。R1 崩溃的因果也**没有建立**，因为缺一个 GT-only 的对照臂。
>
> 另外四条证据边界必须一起说：**原始 0.25 epoch 那次运行缺的是预测产物，所以它的指标不可复算**（但 checkpoint 与逐步日志都在本地）；**R1/R2 的权重不在本地**，只能复算指标不能重跑；**R2 用的是 validation split，和所有 test-split 的数字不能混**；而且全部都是**单 seed**，所以我不做任何显著性主张。

---

## 9. 可能的连续追问及回答

| # | 追问 | 回答要点 |
|---|---|---|
| **1** | **什么是"零优势组"？怎么产生的？** | 组内 reward 标准差 < 1e-6 的组。在 `ranking` reward 下，一组 16 条候选**一条都没命中 GT** 时，`ndcg_rule_reward` 的 `flag` 分支会把整组 reward 覆盖成 0 ⇒ std = 0 ⇒ `A = 0/(0+1e-4) = 0`。 |
| **2** | **零优势比 70 % 说明什么？** | 它衡量"这个样本的 16 条候选一条都没答对"的比例。**所以它同时反映模型能力和 reward 设计** —— 不只是 reward 的问题。 |
| **3** | **R1 为什么把零优势降到 0？** | 因为 Eq.9 除以**本组自己的和**。无 GT 时 `Σ raw = −6.105999`，出向量是 `(1/log2(r+2))/6.105999`，随名次单调、std = 0.031992 ≠ 0 ⇒ advantage 非零。 |
| **4** | **那 R1 的 reward 到底在鼓励什么？** | 在无 GT 的组里，advantage 最小的正是**名次 0**（−3.1557），loss 项 `−A` 因此为**正** ⇒ **模型最自信的那条候选被压低**；名次 15 的 advantage 最大（+0.6990）⇒ **最不自信的那条被抬高**。所以它**不是「锐化自身排序」，而是反惩罚自身排序**（11 个被抬高、5 个被压低）。与「哪个 SID 正确」无关。**注意这是假设，不是已证因果。** |
| **5** | **你怎么知道 R1 的 reward 真的按名次生效？** | 依赖一个前提：rollout 用 `num_beams=16, num_return_sequences=16` ⇒ transformers 判为 **BEAM_SAMPLE** 并**已按序列分数降序返回**，所以**组内位置就是名次**，不需要额外排序。**这个前提写在 `rere_reward.py` 的 docstring 里，作者说已在 4.57.1 上验证过 —— 但我没有独立复验。** |
| **6** | **`group_reward_std_mean` 中位数 0.031992 有什么意义？** | 我从 ReRe 公式**手工算出**"无 GT 组"的 std 正好是 **0.031992**，与日志中位数**逐位相同** ⇒ 说明 R1 下**超过一半的组没有 GT 命中**，其 advantage 完全由 Eq.9 的自指涉分量驱动。这是**独立佐证**，但它**不证明**这些梯度造成了坍缩。 |
| **7** | **R1 的坍缩有多严重？** | test split 上不同的 top-1 预测从 SFT 的 1 059 种掉到 **99 种**（−90.7 %），最高频的一个 SID 占 **33.6 %**。但 `LegalRate` 仍是 **100 %**、`DuplicateRate` 0 % ⇒ **不是非法输出或重复候选造成的**。 |
| **8** | **R2 的 `count_0` 是什么？为什么必须修？** | 约束解码器第一次查 trie 的键原本固定是 `sent[-3:]`。加了 hint 后这个窗口滑到 hint 上，得到 trie 里不存在的键 ⇒ `allowed=[]` ⇒ **强制 EOS**。探针实测：h=0 命中、48 个候选；h=1 / h=2 都 MISS。`count_0` = prompt 末尾已有的 SID token 数，把它们计为"已生成"即可对齐。**默认 0 ⇒ NORMAL 路径逐位不变（G=1 和 G=16 掩码相同）。** |
| **9** | **R2 改动了几件事？** | 命令行层面 4 个 flag（`--reward_type`、`--r21_enable`、`--route_cache`、`--output_dir`）。但**实际生效的变化至少三个**：reward 从排序型变成 0/1、prompt 被注入 hint（72.49 % 的样本）、以及隐含启用的 **prefix CE（系数 0.1）**。 |
| **10** | **零优势比是瓶颈吗？** | ★ **不能这么说。** 本项目只能否定它的**充分性**：R1 把零优势比降到 **0.0** 反而崩溃；R2 降到 **0.475** 仍低于旧 GRPO（0.171 → 0.156）。而它是否构成长训练的**瓶颈/必要因素**，**没有做预算匹配的对照实验** ⇒ 标记为 **UNKNOWN，而不是「已否定」**。同时有一条严格成立的表述：零优势组的 advantage 恒为 0，**对当步梯度的贡献精确为 0**。 |
| **11** | **那 R2 退化的原因是什么？** | **UNKNOWN。** 台账原文就是这么写的：两个变化同时施加、**没有消融分离**，所以不能归因。我的**假设**是 hint 在 72.49 % 的样本上白送了第一个 SID 层级，把学习压力转移到残余后缀 —— 但这是**与数据一致的假设，不是已证原因**。 |
| **12** | **Policy loss 是标准 GRPO 吗？** | **不完全是。** `per_token_loss = exp(logp − logp.detach()) * advantages` 里的指数因子**恒等于 1**（`logp − logp.detach()` 数值上为 0），而且**没有 clip / epsilon**。所以实际是**无 importance ratio 的 REINFORCE-with-KL 变体**，不是带 ratio 裁剪的 PPO/GRPO。这一点在代码里**没有注释**。 |
| **13** | **KL 是怎么算的？为什么会有 1e8 这种值？** | `per_token_kl = exp(d) − d − 1`，其中 `d = ref_logp − logp`。这个函数**恒非负**、`d=0` 时为 0、对 `d<0` 也惩罚，而且**指数放大正尾部**。所以极端的 `d` 会产生极大的 KL。台账实测 median ≈ 0.041、p99 ≈ 36.8、**max = 1.047e8**（2.0 ep 到 1.761e11）。 |
| **14** | **参考模型是固定的吗？** | **不是。** 全部启动器传 `--sync_ref_model True`，所以参考模型会**周期性同步**到当前策略。`β = 1e-3`。 |
| **15** | **优化版为什么快 3.82 倍？质量没代价吗？** | 只改了 **`--eval_step`** 和 **`--save_steps_frac`**（都设成 999999）⇒ 去掉训练中的重复全量验证和冗余存档。wall-clock −73.8 %、吞吐 3.82×、显存不变；HR@20 变化 **−0.088 pp**。台账明确写"**约**保持"而非无损，且**不假设**两次运行的优化轨迹相同。 |
| **16** | **V2（原始 0.25 ep）的数字可靠吗？** | ★ **指标不可复算。** 该 run **缺的是预测产物** `eval_beam20.json`（台账写的 `runs/grpo_short025/eval_beam20.json` 不可得），证据等级 **E4**。但★ **它的 checkpoint 与逐步日志都在本地**：`checkpoints/grpo_evidence/grpo_short025/checkpoint-4379/model.safetensors`（1 977 199 248 B）、`trainer_state.json`、`train.log`、`_EXTRACTED_metrics/grpo_short025_metrics.txt`（4 379 行逐步诊断）。**⇒ 不能说「没有 checkpoint」**——只是没有那一次的评测产物。 |
| **17** | **R1/R2 你能重跑吗？** | **不能。** 预测产物在冻结归档 `freeze_v1_records.tar.gz` 内、被 SHA256 钉住，**指标可复算**；但**权重不在本地**（只有 SHA256 前缀 `7ebdf9c1…` / `a983f46b…`）⇒ 不能重跑、不能做权重级探查。另外按治理规则该归档**不可解压**，所以本轮**没有独立复算任何 GRPO 指标**。 |
| **18** | **R2 的数字为什么不能和别的表放一起？** | 因为它用的是 **validation split（4 532）**，而 R2 之前所有 run 用的是 **test split（4 533）**。台账明文：*"The two splits are not interchangeable and their numbers must never be mixed."* R2 的 validation 三方对比是**内部一致**的，但**与 test-split 表不可比**。 |
| **19** | **为什么 R2 不跑 test？** | **故意的。** validation 上的劣势在**每一个 K**（1/3/5/10/20）上都一致，所以再花 test 算力不会改变结论。 |
| **20** | **如果给你一次机会再做一个实验，做什么？** | **把 R2 的两个变化分开。** 具体是补两个臂：① 只换 0/1 exact-match reward、**不加 hint**；② 只加 hint、**保留原 ranking reward**。这样就能判定退化来自 reward 形状还是 hint 路由。**这是台账自己列的"最高价值的下一个实验"。** 第二优先是给 R1 补 **GT-only rank reward** 对照，以检验那个自我强化的机制假设。 |
| **21** | **有做显著性检验吗？** | **没有。** 全部 **单 seed**，所以本文件不做任何显著性主张，所有数字都是**点估计**。 |
| **22** | **`--mask_all_zero` 是不是就是我用来屏蔽零优势组的开关？** | ★ **不是。** 实测它在 `rl.py` 和 `minionerec_trainer.py` 里**只出现 1 次**（就是签名本身），**全仓库没有任何使用点**。每个启动器都传它，但**它什么也不做**。这也是"我们试过屏蔽零优势组"这种说法会出错的地方。 |
