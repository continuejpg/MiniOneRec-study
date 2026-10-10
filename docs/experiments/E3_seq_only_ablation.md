# docs/experiments/E3_seq_only_ablation.md — SFT 多任务监督与 seq-only 消融

> **本文件由 Phase L 新增，位于影子树中。** 它是 E3 的**单一入口说明**：把训练数据构造、
> 三类任务的真实形态、Loss 实现、Full 与 seq-only 的完整差异、以及**结论边界**串成一条线。
> 复用已有台账（`docs/notes/experiment_summary.md` §14）与逐参数 diff
> （`tools/analysis/results/clean_vs_shuffled_sft_recipe.md`），**不复制**其中的历史内容。
> **本文件不引入任何新数值、不新增命令、不修改任何脚本。**
> **阅读时间：约 6–10 分钟。** 只看结论请跳到 §5 与 §6。
>
> **姊妹文档：** [`E2_shuffled_sid.md`](E2_shuffled_sid.md)（shuffled-SID 干预）

---

## 1. SFT 多任务数据流

`--sft_mode full` 下，`sft.py` 构造 **3 个数据集**，共 **79 834** 条样本：

```
                      data/Amazon/train/<cat>_5_*.csv
                      data/Amazon/index/<cat>.item.json      ← 商品标题/描述
                      data/Amazon/index/<cat>.index.json     ← SID 码本
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  T1 SidSFTDataset            T2 SidItemFeatDataset        T3 FusionSeqRecDataset
  history SID → target SID    SID ↔ 文本对齐                history SID → 目标标题
     data.py:423                 data.py:703                   data.py:1168
     36 259 条                   7 316 条                      36 259 条
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     ▼
                     sft.py:252  ConcatDataset([T1, T2, T3])
                                     ▼
                     sft.py:267-268  HFDataset.from_dict(...)
                                     .shuffle(seed=42)          ← 全局打乱，三类任务混合
                                     .select(range(int(1 * N)))  ← sample_frac=1，恒等
                                     ▼
                     sft.py:287-321  transformers.Trainer
                                     DataCollatorForSeq2Seq(pad_to_multiple_of=8)
                                     ▼
                     labels = [-100]*prompt_len + tokens[prompt_len:]
                     ⇒ Response-only 交叉熵
```

**★ 恒等式（可从产物精确重算）：**
```
36 259  (T1) + 7 316  (T2) + 36 259  (T3) = 79 834
```
**`7 316` 的来历**（容易被误算成 `2 × 3 686 = 7 372`）：

| 量 | 值 | 原因 |
|---|---|---|
| index 条目 | 3 686 | —— |
| `len(sid2title)` | **3 670** | **SID 碰撞**：15 组把 31 个商品压到 15 个 SID，折叠 16 个 |
| `len(title2sid)` | **3 646** | **标题重复**：15 个标题被多个商品共用，折叠 40 个 |
| **T2 样本数** | **7 316** | = 3 670 + 3 646 |

**★ `--sft_mode seq_only` 时：T2 与 T3 **根本不构造**（`sft.py:240`），只剩 T1 = 36 259 条。**

---

## 2. 核心源码定位

| 主题 | 位置 |
|---|---|
| **模式开关（唯一控制点）** | `sft.py:240` `if sft_mode == "full":` |
| 模式白名单校验 | `sft.py:222-225` |
| 签名默认值 | `sft.py:145` `sft_mode: str = "full"` |
| 数据集构造 | `sft.py:228-244` |
| 分量日志（**运行期证据**） | `sft.py:250-251` |
| `ConcatDataset` | `sft.py:252` |
| 打乱与前缀 | `sft.py:266-268` |
| `Trainer` / `TrainingArguments` | `sft.py:287-321` |
| **T1** 主任务 | `data.py:423-492`（`get_history` L429-445、**`pre` L447-492**） |
| **T2** 文本对齐 | `data.py:703-816`（`generate_prompt` L763-774、**`pre` L776-816**） |
| **T3** history→标题 | `data.py:1168-1371`（`get_history` L1277-1311、**`pre` L1319-1371**） |
| prompt 模板 | `data.py:90-94`；统一函数 `data.py:37-44` |
| tokenizer 包装 | `data.py:13-32` |
| **标签屏蔽（三处同构）** | `data.py:483`、`data.py:807`、`data.py:1362` |
| SID 深度推导 | `sid_utils.infer_prefix_index`（评测侧） |

**★ 三处 `labels` 构造**逐字符同构**：**
```python
input_prompt_len = len(tokens)
tokens = tokens + golden_tokens
attention_mask = [1] * len(tokens)
labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
```

---

## 3. Full 与 Seq-only 对照表

| 维度 | **Full SFT** | **Seq-only** |
|---|---|---|
| T1 `SidSFTDataset` | ✅ | ✅ |
| T2 `SidItemFeatDataset` | ✅ | ⛔ **不构造** |
| T3 `FusionSeqRecDataset` | ✅ | ⛔ **不构造** |
| `--sft_mode` | 不传（默认 `full`） | **`seq_only`** |
| 分量日志 | 三个分量 | **`36259 = 36259`** |
| **训练样本** | **79 834** | **36 259** |
| microbatches/epoch | `ceil(79834/16)` = 4 990 | `ceil(36259/16)` = 2 267 |
| **optimizer steps/epoch** | 1 248 | 567 |
| **max optimizer steps** | **2 496** | **1 134** |
| 实际最终 `global_step` | 2 496 | 1 134 |
| `epochs` | 2 | 2 |
| `per_device_train_batch_size` | 16 | 16 |
| `gradient_accumulation_steps` | 4 | 4 |
| 有效 batch | 64 | 64 |
| `learning_rate` | 3e-4 | 3e-4 |
| `cutoff_len` | 512 | 512 |
| `seed` | 42 | 42 |
| `bf16` / `optim` | True / adamw_torch | True / adamw_torch |
| `warmup_steps` | **20（绝对）** | **20（绝对）** |
| ↳ **占调度比例** | **0.801 %** | **1.764 %（×2.20）** |
| `--item_meta_path` | 原文件 | 原文件 |
| 评测协议 / `calc.py` / info 文件 | —— **相同** —— | |

**★ `--output_dir` 也从 `runs/industrial_sft` 变为 `runs/seq_only_clean_sft`，但那是写入目标，不是配方参数。**

### 3.1 ★ 为什么这不是单变量实验
两臂除**任务集合**外，还同时差了 **4 个派生量**：

| # | 同时变化的量 | 幅度 |
|---|---|---|
| 1 | **任务集合** | 3 → 1 |
| 2 | **训练样本数** | −54.58 % |
| 3 | **optimizer 更新次数** | −54.57 % |
| 4 | **学习率调度相对形状** | warmup 占比 ×2.20 |
| 5 | **总 token 数**（非等比，因被移除的 T2/T3 response 更长） | 幅度未量化 |

**⇒ Seq-only 是"配方可比、但非单变量"的对照臂。**

---

## 4. 训练目标与 Loss

### 4.1 三类任务的 prompt / response

| 任务 | prompt 主体 | response |
|---|---|---|
| **T1** | `The user has interacted with items {历史 SID 串} in chronological order. Can you predict the next possible item that the user may expect?` | 目标 `SID` |
| **T2-a** `title2sid` | `Which item has the title: {title}?` | `SID` |
| **T2-b** `sid2title` | `What is the title of item "{SID}"?` | `title` |
| **T3** | `The user has sequentially interacted with items {历史 SID 串}. Can you recommend the next item for him? Tell me the title of the item` | 目标 `title` |

三者的 instruction 前缀不同（`data.py:448-453` / `L780-785` / `L1320-1325`），**prompt 包装统一**为
`### User Input: \n{...}\n\n### Response:\n`（`data.py:90-94`）。

### 4.2 Response-only 交叉熵
* `labels = [-100] * prompt_len + tokens[prompt_len:]`
* `-100` 是 `CrossEntropyLoss` 的默认 `ignore_index` ⇒ **prompt 位置不产生梯度**
* `labels` 由 `tokens` 切片得到 ⇒ **因果错位自动成立**
* **`sft.py:287-321` 未传自定义 loss** ⇒ 用模型自带 causal-LM 交叉熵
* `attention_mask` 全 1 ⇒ response **参与注意力**，被屏蔽的只是 **loss 贡献**
* **Token 权重未做任何归一化**（不等长 response 产生的 token 数差异**未补偿**）

### 4.3 数据混合
* `ConcatDataset([T1, T2, T3])` ⇒ **按顺序串联**，不是交错、不是加权采样
* `.shuffle(seed=42)` ⇒ **全局打乱**，三类任务混合
* `sample_frac = 1` ⇒ `.select()` 是**恒等操作**
* `--sample -1` ⇒ **不下采样**
* `num_train_epochs = 2` ⇒ 全部数据恰好遍历 **2 次**
* **T3 的 `dedup` 未传入** ⇒ 默认 `False` ⇒ 训练集中 **3 069 / 36 259 = 8.46 %** 满足 `target_sid == history[-1]` 的样本**全部保留**

---

## 5. 主要实验结果

**正式协议：原始 test split、4 533 样本、beam 20、约束 SID 解码、原始 INFO、`calc.py` 未修改。**

### 5.1 四条曲线（HR@20 / NDCG@20）

| 臂 | HR@20 | NDCG@20 |
|---|---|---|
| **Clean full SFT** | 0.19832341 | 0.11786798 |
| **Shuffled full SFT** | 0.10765497 | 0.08157358 |
| **Clean seq-only** | **0.17802780** | **0.10842814** |
| **Shuffled seq-only** | **0.10125745** | **0.07683363** |

### 5.2 两个方向的相对变化

**① shuffled-SID 效应（在每个配方内）**
| 配方 | HR@20 clean → shuffled | 相对 Δ |
|---|---|---|
| Full | 0.19832341 → 0.10765497 | **−45.7175 %** |
| **Seq-only** | 0.17802780 → 0.10125745 | **−43.1227 %** |

**绝对下降量之比 = 0.8467（HR@20）、0.8705（NDCG@20）。**

**② 移除文本辅助监督的合计效应**
| 指标 | clean-full → clean-seq-only | 相对 Δ |
|---|---|---|
| HR@20 | 0.19832341 → 0.17802780 | **−10.23 %** |
| NDCG@20 | 0.11786798 → 0.10842814 | **−8.01 %** |

### 5.3 过程性证据（两臂）
`legal_rate = 1.0`、`duplicate_rate = 0.0`、source-target 对齐 **4533/4533**、无 early stopping、跑满 **1 134** 步。

**⇒ 指标差异不是实现差异造成的。**

---

## 6. 实验局限

### 6.1 ★ 结论边界（必须与结果同引）

**✅ 可以支持：**
> 移除两条显式文本辅助监督后，shuffled-SID 效应仍然很大（HR@20 −43.12 %，为 full 配方的 **84.67 %**）
> ⇒ **该效应并非主要依赖这两条辅助路径。**

**❌ 不可以支持：**
| 不可支持的表述 | 为什么 |
|---|---|
| "全部下降源于文本语义信息" | **43.12 % 是 shuffled vs clean 的效应**，不是 full vs seq-only 的效应 |
| "文本辅助监督的独立贡献是 X %" | `−10.23 %` 是"移除辅助监督 **+ 少 54.6 % 训练量 + 更新次数减半 + 调度形状改变**"的**合计**，不能拆给任何单项 |
| "移除了系统的全部文本信息" | SID 本身仍是**文本派生的**且两臂完全相同；移除的只是**显式对齐目标**。★ 注意：本项目**没有**独立验证 P0 目录的具体 SID 构造算法（见 §6.4） |
| "证明了语义层次结构" | 打乱**同时**破坏语义与任意对应，未做分离实验（同 E2） |
| "这是单变量实验" | 见 §3.1，四项同时变化 |
| "结论具有普适性" | **单类目、单 seed**，无方差估计与显著性检验 |

### 6.2 未完成的对照（未来工作）
| 缺失的对照 | 作用 |
|---|---|
| **匹配训练量的 full 对照**（如 `T1-only` 但跑到 2 496 步） | ★ **最关键** —— 检验"full 的优势是否只来自更多更新次数" |
| **只移除 T2 / 只移除 T3** | 分离两条辅助监督各自的贡献 |
| **恢复 T3 的描述变体** | `data.py:1335-1343` 已注释 ⇒ **描述监督从未被真正测试** |
| 多 seed / 多类目 | 方差与泛化 |

### 6.3 其他已知局限
* **T3 名义上是"title + description 双任务"，但描述分支被注释掉**（`data.py:1335-1343`）⇒ 实际只运行标题变体；`sid2description` 与 `_process_description()`（47 行）是**死代码**。
* **T2 的两个子任务一起移除**，未分离。
* **`warmup_steps=20` 是绝对步数**，不随调度总长缩放 ⇒ 两臂相对形状不同。
* **T2/T3 的 response 更长**（标题）⇒ 移除它们造成的 token 削减**大于** 54.58 %（幅度未量化）。

### 6.4 ★ P0 目录的 SID 构造算法：**未独立验证**（证据缺口）

上表引用了"SID 是文本派生的"。**这一点在本项目中无法从代码或产物独立确认**，必须如实标注：

| 事实 | 证据 |
|---|---|
| 本树**没有** `*.npz` / codebook / quantizer checkpoint | `deprecated/self_audit/MINIONEREC_PROJECT_STATUS_REPORT.md` 的清单缺口条目 |
| P0 目录的 SID 构造过程**在本 clone 中不可复现**；本项目也**明确选择不重建**量化器（"official SID artifacts are shipped and verified"） | 同上；`MINIONEREC_CODE_AUDIT.md` 的 "Not needed" 清单第 9 条 |
| `data/Amazon/index/Industrial_and_Scientific.index.json` 在本树中**只是一个输入产物**：3 686 item / 3 670 unique SID / 15 碰撞组 / 31 item / 560 个 SID token | 本轮只读复算 |
| `rq/` 下**存在** RQ-VAE / RQ-KMeans / RQ-KMeans+ / constrained 的多份实现，`README.md` 也指向该路径 | 源码确认 |

**⇒ 因此正确的表述是："P0 目录的 SID 由上游 `rq/` 路径（RQ-VAE / RQ-KMeans 家族）产生"是一个**与仓库内容一致的归属**，但**不是本项目独立验证过的事实**。**
**⇒ 本 E3 的结论**不依赖**具体是哪一个量化器** —— 因为**两臂使用完全相同的 SID 码本**，所以无论码本由哪种算法产生，"移除了显式对齐目标"这一论述都成立。**
**（登记为证据缺口 L-9。）**

---

## 7. 一分钟与三分钟面试介绍

### 7.1 一分钟版
> 这个项目里 SFT 不是单任务，而是**三个任务一起训**：主任务是"历史 SID → 下一个商品的 SID"，另外两条是**显式文本辅助监督** —— 一条做"SID ↔ 标题"的双向对齐，一条做"历史 SID → 目标标题"。合起来 **79 834** 条样本。
>
> 我做的消融叫 **seq-only**：把 `sft.py` 的 `--sft_mode` 设成 `seq_only`，**那两条辅助任务就根本不构造**，只剩主任务的 **36 259** 条。
>
> 目的是检验一个稳健性问题：**shuffled-SID 干预的效应，是不是主要靠那两条文本辅助监督撑起来的？** 结果是**不是** —— 去掉辅助监督后，打乱 SID 仍然让 HR@20 从 0.178 掉到 0.101，**下降 43.12 %**，是完整配方效应（45.72 %）的 **84.67 %**。
>
> 但我要说清楚**这不是单变量实验**：两臂除了任务集合，还同时差了训练样本数（少 54.6%）、optimizer 更新次数（少 54.6%）和学习率调度的相对形状。所以它证明的是**效应稳健**，**不能**用来测量"文本辅助监督贡献了多少"。

### 7.2 三分钟版

**0:00–0:40　背景：SFT 是多任务的**
> MiniOneRec 的 SFT 阶段不是只训一个任务。`sft.py` 在 `full` 模式下构造**三个数据集**：主任务是"用户历史 SID → 下一个商品的 SID"；第二个是 **SID 和文本的双向对齐**，包含"给标题问 SID"和"给 SID 问标题"两个子任务；第三个是"历史 SID → 目标商品的**标题**"，也就是让模型直接生成文本。三个数据集用 `ConcatDataset` 串起来，一共 **79 834** 条样本，然后全局 shuffle 打散，所以每个 batch 里三类任务混合。
>
> 有个细节值得说：第二条任务的样本数不是简单的"商品数 × 2"。因为存在 **SID 碰撞**（15 组把 31 个商品压到 15 个 SID），以及**标题重复**（3 686 个商品只有 3 646 个不同标题），两个映射字典都会折叠，最后是 **3 670 + 3 646 = 7 316**。这个数我能从 `index.json` 和 `item.json` **精确重算出来**，和训练日志里的 `79 834` 完全对得上。

**0:40–1:20　Loss：response-only**
> 三个任务的监督形式是**完全一致**的：把 instruction + prompt 拼成 token 序列，然后接上 response 的 token，最后 `labels = [-100] * prompt_len + tokens[prompt_len:]`。`-100` 是 `CrossEntropyLoss` 的默认 `ignore_index`，所以**只有 response 段产生梯度**，prompt 段被完全屏蔽。`labels` 直接从 `tokens` 切片得到，因果错位自动成立。`Trainer` 那边没有传自定义 loss，用的是模型自带的 causal LM 交叉熵。

**1:20–2:00　消融的设计**
> 消融的实现非常干净：`sft.py` 里**只有一个控制点**，就是 `if sft_mode == "full"`。设成 `seq_only` 时，第二、第三个数据集**根本不构造**，不是给空数据。
>
> 而且这个有**运行期证据**：训练日志会打印 `train dataset components`，full 臂是三个分量，seq-only 臂是 **`36259 = 36259`** —— 一个分量。这行日志本身就是"辅助任务确实没被实例化"的证明。

**2:00–2:40　结果**
> 结果分两个方向看。**第一个方向**：在 seq-only 配方内打乱 SID，HR@20 从 **0.17803** 掉到 **0.10126**，相对下降 **43.12 %**；完整配方是 45.72 %，**绝对下降量的比值是 0.8467**。所以**移除那两条文本辅助监督之后，shuffled-SID 的效应基本没变**。
>
> **第二个方向**：从 clean-full 到 clean-seq-only，HR@20 从 0.19832 到 0.17803，也就是 **−10.23 %**。
>
> 另外两臂的 `legal_rate` 都是 1.0、`duplicate_rate` 都是 0、样本对齐 4533/4533，说明**指标差异不是实现问题**。

**2:40–3:00　边界（这部分我一定会说）**
> 但这个实验**不是单变量**。两臂除了任务集合，还同时差了四样东西：训练样本数少了 54.6%、optimizer 更新次数少了 54.6%、学习率调度的相对形状也变了（`warmup_steps=20` 是绝对步数，在 1134 步的调度里占 1.76%，在 2496 步里只占 0.80%，差 2.2 倍）。
>
> 所以：**43.12 % 只能说明"效应稳健"，不能说明"文本辅助监督的贡献"**；而那个 **−10.23 %** 是四项变化的**合计**，不能拆给任何单项。
>
> 要真正隔离文本辅助监督的影响，最关键的一个对照是：**只保留主任务、但把更新次数跑到和 full 一样多**。如果那样也能接近 full 的水平，就说明 full 的优势主要来自**训练量**而不是辅助监督的内容。**这个对照我还没有做。**

---

## 8. 可能的连续追问及回答

| 追问 | 回答要点 |
|---|---|
| **为什么第二个任务的样本数是 7 316 而不是 7 372？** | 因为样本数是**两个字典的键数**。`sid2title` 被 SID 碰撞折叠到 3 670，`title2sid` 被标题重复折叠到 3 646，合计 7 316。可从 `index.json` + `item.json` 精确复算。 |
| **`-100` 是怎么起作用的？** | 它是 `CrossEntropyLoss` 的默认 `ignore_index`，那些位置的 loss 被忽略、不产生梯度。因为 `labels` 由 `tokens` 切片而来，因果错位（`labels[t] == input_ids[t+1]`）自动成立。 |
| **为什么 response 更长会让"训练量减少"超过 54.6%？** | 被移除的 T2/T3 的 response 是**商品标题**，比 T1 的"3 个 SID token"长得多。所以按 **token** 算的削减幅度大于按**样本**算的 54.58%。**这个数我还没量化**，因为需要跑一遍 tokenizer 预处理。 |
| **`attention_mask` 全是 1，那 prompt 不是也被算进去了？** | 不是。`attention_mask` 控制**注意力可见性**，`labels=-100` 控制 **loss 贡献**。prompt 需要被 response attend 到（否则模型看不到历史），但不需要被预测。两者正交。 |
| **`ConcatDataset` 会不会导致某类任务被过采样？** | 会**隐式**加权：T1 和 T3 各 36 259 条、T2 只有 7 316 条，所以按样本数加权，T2 约占 9.2%。`shuffle(seed=42)` 只是打散顺序，不改变比例。**没有显式的任务配比控制。** |
| **T3 的 `dedup` 为什么没生效？** | `sft.py:243` 构造 `FusionSeqRecDataset` 时**没有传 `dedup`**，默认 `False`。训练集里有 3 069 条（8.46%）满足 `target_sid == history[-1]`，这些样本全部保留。 |
| **T3 不是还有 description 任务吗？** | 代码里**有**描述分支，但被注释掉了（`data.py:1335-1343`），实际只跑标题变体。所以 `sid2description` 和那个 47 行的 `_process_description()` 在当前配置下是**死代码** —— 描述监督**从未被真正测试**。 |
| **`warmup_steps=20` 相同，为什么说调度不一样？** | 因为它是**绝对步数**。20/2496 = 0.80%，20/1134 = 1.76%，相对占比差 2.2 倍。相同策略 ≠ 相同轨迹。 |
| **那 full 的优势到底来自任务还是训练量？** | **目前无法区分**，这正是最大的局限。需要补一个"只保留主任务但跑到 2 496 步"的对照臂。 |
| **有没有做显著性检验？** | 没有。**单 seed、单类目**，没有方差估计。所以所有数字都是**点估计**。 |
| **`--sft_mode` 会不会在别处也有分支？** | 只有一个控制点：`sft.py:240`。另有 `sft.py:222-225` 做白名单校验，非法值直接 `ValueError`。 |
| **怎么证明辅助任务真的没被构造？** | 训练日志的 `train dataset components` 行：seq-only 臂打印 **`36259 = 36259`**（一个分量）。代码上它位于 `sft.py:250-251`，在 `ConcatDataset` 之前。 |
