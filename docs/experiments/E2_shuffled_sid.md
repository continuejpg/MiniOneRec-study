# docs/experiments/E2_shuffled_sid.md — shuffled-SID 受控干预（核心实验）

> **本文件由 Phase K 新增，位于影子树中。** 它是 E2 的**单一入口说明**：把此前散落在
> `docs/notes/experiment_summary.md` §10/§14、`tools/analysis/results/clean_vs_shuffled_sft_recipe.md`、
> `tools/analysis/results/clean_vs_shuffled_eval_recipe.md` 三处的材料串成一条线，并补齐**代码可追溯性**。
> **本文件不引入任何新数值、不新增命令、不修改任何脚本。** 一切数值与证据等级以被引用文件与原始产物为准。
> **阅读时间：约 5–10 分钟。** 只想看结论请跳到 §5 与 §6。

---

## 1. 为什么做 shuffled-SID

MiniOneRec 把每个商品压成 3 个 token 的 Semantic ID（`<a_x><b_y><c_z>`），让 LLM 生成下一个商品的 SID。
SID 由上游文本量化得到，**看起来**承载语义层次。

**但 SID 是任意指派的**：3 686 个商品被映射到 3 670 个唯一码（15 组碰撞、31 个商品受影响）。

于是有一个绕不开的问题：

> **SFT 的高 HR@20，来自"SID 的语义层次结构"，还是仅仅来自"模型记住了 item ↔ SID 这张对应表"？**

这个问题之所以值得做实验，是因为**如果只是背表**，那么"语义 ID"这一说法就没有内容支撑。
E2 就是为了区分这两种解释。

---

## 2. 为什么"简单随机替换 SID"不够严谨

朴素做法是"把 SID 随机重排再训练一次，看掉多少"。**但这样会同时改变很多东西**：

| 朴素替换会改动的量 | 为什么这会毁掉归因 |
|---|---|
| SID 的**码本集合**（哪些码存在） | 解码时的合法候选空间变了 |
| SID 的**多重集**（每个码被几个商品占用） | 碰撞结构变了 |
| **trie 前缀结构**（约束解码的掩码） | 生成难度本身变了 |
| **SID token 词表** | 词表大小/新 token 数变了，嵌入布局变了 |
| **流行度分层** | 热门商品可能被分到冷门码 |

一旦这些量变了，**HR@20 的下降就无法归因给"item↔SID 对应关系"** —— 任何一项都能造成下降。

**⇒ E2 的设计目标：只改"哪个商品拥有哪个 SID"，其他全部锁死。**

---

## 3. 具体如何实施干预

### 3.1 干预算法（`analysis/build_shuffled_sid_strict.py`）

```
1. 从 index 读 item→SID；统计 sid→items           (L124-129)
2. 划分：
     collision SIDs  = 被 ≥2 个商品占用的 SID        (L131)
     collision items = 这 15 组涉及的 31 个商品      (L132)
     singleton items = 其余 3 655 个商品             (L133)
3. 冻结：31 个碰撞商品 → 保留原 SID，不参与置换      (L141-142)
4. 对每个【训练频次桶】内的 singleton：
     a. 取出该桶 singleton 的 SID 列表               (L152)
     b. rng.shuffle(slots)   ← 桶内置换，seed=42     (L154-155)
     c. 修复不动点：若 slots[i] 仍是自己的原 SID，
        与另一个 j 交换，要求交换后两者都不落在原值上  (L157-164)
5. 写出新的 index（同名 <Category>.index.json）      (L179-180)
6. 用同一套映射重写 train/valid/test 的 SID 列       (L191-206)
7. 跑 18 项不变量检查 + 干预强度统计                 (L209-319)
```

**★ 关键点：置换是"桶内错排（derangement）"，不是全局随机置换。** 每个 singleton 一定拿到**同频次桶**里的另一个 SID。

### 3.2 流行度分桶（`analysis/build_shuffled_sid_strict.py:68-69,146-147`）

桶的定义基于**训练集**中该商品作为 target 出现的次数：

| 桶 | 频次范围 |
|---|---|
| `f=0` | 0 |
| `f=1` | 1 |
| `f=2` | 2 |
| `f=3-5` | 3–5 |
| `f=6-20` | 6–20 |
| `f=>20` | ≥21 |

频次来源：`Counter(r["item_id"] for r in csv.DictReader(train.csv))`（`L137-138`）—— **只用训练集，不看 valid/test**（避免泄漏）。

### 3.3 为什么必须冻结那 31 个碰撞商品（`L7-26` 的 docstring 直接说明）

早期版本 `build_shuffled_sid.py` **把碰撞组整体置换**，导致**大小相同的碰撞组互换 SID 字符串** ⇒ **精确 SID 多重集没有被保留**。
严格版改为：**所有碰撞涉及的商品原地冻结**，只置换 singleton。

**为什么这样就安全：** singleton 的 SID 构成一个**闭集**（没有 singleton 的 SID 被碰撞商品持有，且每个 singleton SID 唯一）⇒ **桶内错排天然保持**：
* 精确 SID 集合 与 **精确 SID 多重集**；
* **碰撞组及其成员**；
* **每一层的 trie 结构**；
* **SID token 词表**。

### 3.4 干预强度（实测，来自 `analysis/results/shuffled_sid_preflight.json`）

| 指标 | 值 |
|---|---|
| singleton 精确 SID 保留率 | **0.0000 %**（`L268-270` 断言 fixed points = 0） |
| 全体精确 SID 保留率 | **0.8410 %**（= 31 / 3686，即被冻结的那些） |
| prompt 改变比例 | **98.81 %** |
| target 改变比例 | **93.01 %** |

**⇒ 99.16 % 的商品失去了原有的 item↔SID 对应关系。**

---

## 4. 如何控制其他变量

### 4.1 18 项不变量检查（`build_shuffled_sid_strict.py` L219-273）

| # | 检查 | 行号 |
|---|---|---|
| 1 | item 数量不变 | L219 |
| 2 | item 身份集合不变 | L220 |
| 3 | **精确 SID 集合相同** | L222-224 |
| 4 | **精确 SID 多重集相同**（逐字符串计数） | L226-229 |
| 5 | **15 个碰撞组及其成员完全相同** | L231-234 |
| 6-8 | **trie 结构在深度 1/2/3 完全相同** | L236-238 |
| 9 | **SID token 词表相同** | L240-242 |
| 10-15 | train/valid/test：行数不变；`user_id`/`item_id`/`item_title`/`history_item_id` 不变 | L244-253 |
| 16 | 所有 CSV 的 history/target SID 与新映射一致 | L255-266 |
| 17 | **singleton 不动点 = 0** | L268-270 |
| 18 | **只有 31 个碰撞商品保留原 SID** | L271-273 |

### 4.2 两臂之间**只换 3 个路径**（训练侧）

`scripts/sft_shuffled_sid.sh` 的头部注释（`L8-24`）明确声明：它是 `sft_full.sh` 的**功能克隆**，**只改**：

| 参数 | clean | shuffled |
|---|---|---|
| `--train_file` | `data/Amazon/train/…csv` | `analysis/shuffled_sid/train.csv` |
| `--eval_file` | `data/Amazon/valid/…csv` | `analysis/shuffled_sid/valid.csv` |
| `--sid_index_path` | `data/Amazon/index/<cat>.index.json` | `analysis/shuffled_sid/<cat>.index.json` |
| `--output_dir` | `runs/industrial_sft` | `runs/industrial_sft_shuffled_sid` |（写入目标，不改配方）

其余全部相同：`--base_model`、`--sample -1`、`--num_epochs 2`、`--batch_size 64`、`--micro_batch_size 16`、`--learning_rate 3e-4`、`--cutoff_len 512`、`--seed 42`、`--train_from_scratch False`、`--freeze_LLM False`。

**★ 两个刻意的设计决定：**
1. **`--item_meta_path` 保持原文件**（`scripts/sft_shuffled_sid.sh:22-23`）：标题/描述是**商品属性**，不是 SID 指派；干预不应触碰它们。
2. **`--info_file` 评测时保持原文件**（见 §4.3）：早期版本曾换成 shuffled info 并声称"否则约束解码会还原映射"——**该主张已被撤回**（`scripts/eval_shuffled_sid.sh:25-43`）。审计结论：`evaluate.py` 读 info 文件**只用字段 0（SID 串）** 来建 trie，**item_id 字段从不被读**；trie **只是 SID 码本的函数**，而严格置换**保持码本不变** ⇒ 两个 info 文件产生**同一个 trie**（9684 个前缀键、键集合相同、允许 token 集合 0 处不同；121 个键仅**集合迭代顺序**不同，而 `LogitProcessor.py:68` 的先进索引赋值**忽略顺序**）。

### 4.3 评测侧的变量控制

`scripts/eval_shuffled_sid.sh` 声明它**复现正式 clean 协议**，只替换三处（模型 / test 数据 / 输出），其余协议常量逐字相同：
```
BATCH_SIZE=8  NUM_BEAMS=20  MAX_NEW_TOKENS=256  LENGTH_PENALTY=0  SEED=42  K=0
```
（`eval_shuffled_sid.sh:85-90`）

**preflight 还额外断言**（`L122-177`）：
* 两个 info 文件的**行数、SID token 词表、合法 SID 码本**完全相同；
* shuffled test **恰好 4 533 行**；
* 模型 tokenizer **词表 = 152 225**；
* shuffled test 的所有 SID 都在 shuffled index 中（trie 一致性）。

---

## 5. 实验结果说明了什么

| 模型 | HR@20 | NDCG@20 |
|---|---|---|
| Clean SFT | **0.19832341** | **0.11786798** |
| **Shuffled-SID SFT** | **0.10765497** | **0.08157358** |
| **相对变化** | **−45.72 %** | **−30.79 %** |

**★ 姊妹文档：** [`E3_seq_only_ablation.md`](E3_seq_only_ablation.md)（SFT 多任务监督与 seq-only 消融）

**★ 稳健性延伸（`docs/notes/experiment_summary.md` §14、§20）：**

| 臂 | 打乱后 HR@20 相对下降 |
|---|---|
| 完整配方（0.5B） | −45.72 % |
| **去掉两条显式文本辅助监督**（seq-only，0.5B） | **−43.12 %** |
| 完整配方，骨干放大到 1.5B（3.124×） | **−41.77 %** |

**⇒ 在锁死码本、词表、trie、碰撞结构、流行度分层与配方之后，仅打乱 item↔SID 就使 HR@20 近乎腰斩；且该效应在移除显式文本辅助监督、以及放大骨干后依然巨大。**

---

## 6. ★ 实验**不能**证明什么（必须与结论同引）

**不要**把"HR@20 下降"直接读成"模型学到了完整的 SID 语义层次结构"。本实验能支持的最强表述是：

> **在完整的 MiniOneRec SFT 配方下，item↔SID 的对应关系对推荐质量有实质性贡献。**

| # | 不能证明的推论 | 原因 |
|---|---|---|
| **1** | **不能证明 SID 是唯一原因** | 干预只破坏了对应关系，未做"只破坏层次、保留对应"的分离实验 |
| **2** | **不能证明模型学到了"语义层次"** | 打乱同时破坏了**语义**与**任意对应**两者；本实验**没有把二者分开** |
| **3** | **31 个碰撞商品未受处理** | 它们保留原 SID，既不能被用来解释下降，也不构成对照 |
| **4** | **seq-only 臂不是单变量隔离** | 它同时把训练行数从 **79 834 降到 36 259**（−54.6 %） |
| **5** | **单 seed、单类目** | 无方差估计、无显著性检验 |
| **6** | **beam-restricted 指标** | HR@20 受 beam 宽度 20 限制，**不是全目录召回** |
| **7** | **主干并非严格单变量** | `seq-only` 移除的是**显式文本辅助监督**，不是"所有文本信息"（SID 本身仍由上游文本量化产生） |

**⇒ E2 是受控干预证据，不是因果唯一性的证明。**

---

## 7. 怎样定位相关源码和证据

### 7.1 源码（按执行顺序）

| 阶段 | 文件 | 关键位置 |
|---|---|---|
| **① 构造干预数据** | `analysis/build_shuffled_sid_strict.py` | 分桶 `L68-69`；冻结 `L141-142`；桶内错排+修不动点 `L152-172`；写 index/mapping `L179-189`；CSV 重写 `L191-206`；不变量 `L219-273`；强度 `L275-319` |
| **② 训练** | `scripts/sft_shuffled_sid.sh` | 三路径替换 `L47-58`；preflight `L66-137`（含内嵌 Python 断言 TokenExtender 路径，`L105-131`）；`torchrun` 调用 `L145-163` |
| | `sft.py` | `TokenExtender` 类 `L59-84`；调用点 `L180-190`；`sft_mode` 校验 `L222-225`；数据集构造 `L228-244`；Labels 掩码见下 |
| | `data.py` | `SidSFTDataset` `L423-492`；prompt 模板 `L90-94`；统一 prompt 函数 `build_recommendation_prompt` `L37-44`；tokenizer 包装 `L13-32` |
| **③ 评测** | `scripts/eval_shuffled_sid.sh` | 协议常量 `L85-90`；preflight `L92-182`；`evaluate.py` 调用 `L191-203`；`calc.py` 调用 `L210-213` |
| | `evaluate.py` | trie/hash_dict 构建 `L103-136`；`ConstrainedLogitsProcessor` 使用 `L201-208`；`generate()` `L225-232` |
| | `calc.py` | `gao()` 全函数 `L12-81`；HR/NDCG 累加 `L70-75`；口径提示见下 |

### 7.2 结果来源（`calc.py` 的精确口径）

```
L47   candidates = [predict] 逐条 strip 引号
L55-58  target 取 output 字段（list 时取 [0]）
L60-68  从 rank 0 起找【第一个】精确匹配 → minID = 该 rank；找不到则 minID 保持 1_000_000
L66-68  命中即 break（只取首次命中）
L73-75  if minID < topk:  HR += 1;  NDCG += 1 / log(minID + 2)
L79     NDCG 打印时再除以 (1/log 2)
L52,71  topk 过滤：只统计 k <= n_beam（beam=20 ⇒ 报 1/3/5/10/20）
```

**★ 两个容易误读的点（阅读结果时务必注意）：**
1. **NDCG 累加用的是自然对数** `1/log(minID+2)`，**打印时才除以 `1/log(2)`** 换算到 log2 尺度。因此**中间累加器不是最终指标**，直接取用会大幅高估 NDCG。
2. **一条样本只匹配一次**（`break`）；若同一 SID 在多个名次出现，只算最靠前的那个。

### 7.3 证据位置

| 层 | 位置 | 证据等级 |
|---|---|---|
| 干预数据 + 18 项不变量 + 强度 | `analysis/results/shuffled_sid_preflight.json` | 构建产物 |
| 训练/评测的配方差异 | `tools/analysis/results/clean_vs_shuffled_sft_recipe.md`、`clean_vs_shuffled_eval_recipe.md` | 逐参数 diff |
| 预测与指标 | `minionerec_final_artifacts.tar.gz` → `runs/eval_shuffled_sid/{test_beam20.json, metrics.txt}` | **E2**（该归档**无 manifest**） |
| 完整实验记录 | `docs/notes/experiment_summary.md` §10、§14 | 权威台账 |
| 登记与溯源 | `docs/provenance/reproducibility.md` | 复现与溯源说明 |

**★ 干预数据集的位置：** `analysis/shuffled_sid/`（train/valid/test.csv + index + mapping，共 28.6 MB）。**注意：该目录在主仓库 T-root，不在本树内**；在本树中运行时需把 `PROJECT_ROOT` 指向 T-root（或相应覆盖 `_paths.py` 的变量）。

---

## 8. 一分钟速览

1. SID 是**任意指派**的 → 需要区分"学到语义结构"与"背下对应表"。
2. 朴素随机换 SID 会同时改码本/trie/碰撞/分层 → **无法归因**。
3. E2 的做法：**冻结 31 个碰撞商品，把 3 655 个 singleton 在训练频次桶内错排（seed 42）**。
4. 结果：**18/18 不变量通过**，精确 SID 多重集、trie、token 词表、碰撞组**全部不变**；singleton 精确保留率 **0.0000 %**。
5. 结果：HR@20 **0.19832341 → 0.10765497（−45.72 %）**；seq-only 臂 **−43.12 %**；1.5B **−41.77 %**。
6. 边界：**这不证明 SID 是唯一原因，也不证明模型学到了完整语义层次**；单 seed、单类目、beam-restricted。
