# docs/ENTRYPOINTS.md — 核心实验入口索引

> **本文件由 Phase H（P3-A1）新增，位于影子树中。**
> **它只做定位与转述，不引入任何新实验、新数值或新命令。**
> 所有事实均来自**对影子树内文件的静态审计**（见 §9 的方法与限制）。
> **本文件不修改任何脚本；不改动 `sft.py` / `rl.py` / `evaluate.py` / `calc.py` 的语义。**

---

## 0. 先读这一节：两种世代的启动器

本仓库的 `.sh` 启动器分为**两个世代**，写法与可移植性不同。**判断某个脚本是否"正式入口"时，先看它属于哪一代。**

| | **世代 A：上游 / 早期（legacy 写法）** | **世代 B：本项目加固（hardened 写法）** |
|---|---|---|
| Python 解释器 | 字面量 `python` | `"$PY"`（由 `scripts/common.sh` 提供，默认 `python`） |
| 脚本路径 | 相对 `./evaluate.py`（**依赖 CWD**） | `"$PROJECT_ROOT/evaluate.py"`（**绝对，不依赖 CWD**） |
| 路径来源 | 硬编码 `D=/root/autodl-tmp` | `source scripts/common.sh`（可用环境变量覆盖） |
| 前置检查 | 无 | `chk` / `check_exists` 断言目标文件存在，缺失即拒绝启动 |
| GPU 指定 | `CUDA_VISIBLE_DEVICES=$i`（**对 `evaluate.py` 无效，见 §7**） | `CUDA_VISIBLE_DEVICES=0` |
| 代表脚本 | `sft.sh`、`rl.sh`、`evaluate.sh`、`eval_industrial.sh`、`eval_letter.sh` | `sft_full.sh`、`scripts/*.sh`、`patches/grpo_*.sh` |

**★ 关键提示：** 世代 A 的 `evaluate.sh` 含一处**已确认的参数错误**（§7）；世代 A 的脚本多数**硬编码 AutoDL 路径，在本机不可直接运行**。

---

## 1. 实验一：Clean SFT（P0 基线）

### 1.1 研究目的
在官方预构建 SID 上做一次全参数 SFT，得到后续所有比较的**参照点**。

### 1.2 真实历史入口

| 入口 | 世代 | 性质 |
|---|---|---|
| **`sft_full.sh`** | **B** | ★ **正式入口（canonical）**。自述"portable paths… override any of them from the environment" |
| `sft.sh` | A | **上游启动器**。可用但硬编码 `sample=-1` 以外的默认较少，且**未强制显式传模型路径** |
| `deprecated/smoke_launchers/sft_smoke.sh` | A | 冒烟用（`sample=100`、1 epoch、batch 8）。**已在 P2b 归档** |

### 1.3 调用链
```
sft_full.sh
  └─ source scripts/common.sh            # 提供 PROJECT_ROOT/RUN_ROOT/DATA_ROOT/CATEGORY/PY/BASE/TRAIN/VALID/INDEX/ITEM_META
  └─ : "${BASE_MODEL:?…}"                # 强制显式传模型路径，未设则立即失败
  └─ torchrun --nproc_per_node 1 sft.py  # sft.py 在仓库根
       └─ from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, …
       └─ TokenExtender（读取 --sid_index_path）→ tokenizer.add_tokens → resize_token_embeddings
```
`` 【源码确认】``

### 1.4 核心配置（`sft_full.sh` 显式传参）
| 参数 | 值 |
|---|---|
| `--base_model` | `"$BASE_MODEL"`（**无默认，必须显式设置**） |
| `--train_file` / `--eval_file` | `$TRAIN` / `$VALID` |
| `--output_dir` | `$RUN_ROOT/industrial_sft` |
| `--category` | `$CATEGORY`（默认 `Industrial_and_Scientific`） |
| `--sid_index_path` / `--item_meta_path` | `$INDEX` / `$ITEM_META` |
| `--sample` | `-1`（全量） |
| `--num_epochs` | `2` |
| `--batch_size` / `--micro_batch_size` | `64` / `16`（⇒ 梯度累积 4） |
| `--learning_rate` | `3e-4` |
| `--cutoff_len` | `512` |
| `--seed` | `42` |
| `--train_from_scratch` / `--freeze_LLM` | `False` / `False`（全参数） |

**未显式传、由 `sft.py` 默认值决定**（`sft.py:145-146` 【源码确认】）：
* `--sft_mode` 默认 **`"full"`** ⇒ **`sft_full.sh` 实际运行的就是完整配方**（三个数据集全开）
* `--probe_optimizer_steps` 默认 **`-1`** ⇒ 探针关闭（正式行为）

### 1.5 输入 / 输出 / 证据
| 项 | 值 |
|---|---|
| 输入 | `data/Amazon/{train,valid}/Industrial_and_Scientific_5_2016-10-2018-11.csv`；`data/Amazon/index/*.index.json`、`*.item.json`；`data/Amazon/info/*.txt` |
| 输出 | `$RUN_ROOT/industrial_sft/{train.log, final_checkpoint/}` |
| 产物（**不在影子树内**） | `checkpoints/industrial_sft/final_checkpoint/model.safetensors`（SHA-256 `c362ed18…`）；预测 `checkpoints/grpo_evidence/eval_clean_sft/test_beam20.json` |
| 证据等级 | **E3**（本地预测 + 本地 metrics + 第三方 manifest 声明的预测哈希三方一致） |
| 权威记录 | `docs/notes/experiment_summary.md` §5 |

### 1.6 已知限制与缺失
* **训练超参不完整可核**：`runs/industrial_sft/` 的 `training_args.bin` / `trainer_state.json` **不在交付物中**；上表来自启动器**显式传参**，`sft.py` 内部的默认（如学习率调度、精度）只能由源码确认。
* **运行时 commit 未确认**：没有任何证据把这次运行绑定到某个 commit。
* **shuffled 臂曾被 resume**（≈step 2180 中断，从 `checkpoint-2125` 恢复）—— 见 `docs/notes/experiment_summary.md`。

### 1.7 经过验证的运行命令
**无。** 本轮**未运行**任何训练（G4/G5 未执行，环境不具备）。以下是**启动器的调用形态**（转述自脚本，非本轮验证过）：
```
BASE_MODEL=<path-to-Qwen2.5-0.5B> bash sft_full.sh
```

### 1.8 当前可运行性
| 检查 | 状态 |
|---|---|
| `bash -n` | ✅ 通过（见 `phaseH/…`） |
| 参数与 `sft.py` 是否匹配 | ✅ **16/16 全部被接受** |
| 端到端可运行 | ⛔ **未验证**（缺 `fire`/`torch`/`transformers`/`trl`/GPU/权重） |

---

## 2. 实验二：shuffled-SID 受控干预 ★ 核心

> **★ 深入阅读：[`experiments/E2_shuffled_sid.md`](experiments/E2_shuffled_sid.md)**
> 该文件是本组实验的**单一入口说明**：为什么做、为什么朴素替换不够、干预算法与分桶、
> 18 项不变量、两臂只换哪 3 个路径、结果说明什么、**不能证明什么**，以及逐阶段的**代码行号索引**。
> 本节只列启动入口；设计与证据请读上面那份。

### 2.1 研究目的
**保持 SID 码本、词表、trie 拓扑、碰撞结构、流行度分层与 SFT 配方全部不变，只打乱 item↔SID 的对应关系**，测量"语义 ID 的对应关系"本身承载多少推荐信号。

### 2.2 真实历史入口（三步，缺一不可）
| 步骤 | 入口 | 世代 |
|---|---|---|
| ① 构造干预数据 | `analysis/build_shuffled_sid_strict.py` | B（`__main__`，无 CLI 框架） |
| ② 训练 | **`scripts/sft_shuffled_sid.sh`** | **B** |
| ③ 评测 | **`scripts/eval_shuffled_sid.sh`** | **B** |
| 辅助（仅审计） | `analysis/build_shuffled_info.py`、`analysis/preflight_shuffled_sft.sh`、`analysis/preflight_eval_shuffled_sid.sh`、`analysis/audit_info_file_equivalence.py` | B |

### 2.3 调用链
```
analysis/build_shuffled_sid_strict.py        （确定性，seed 42）
   └─> analysis/shuffled_sid/{train,valid,test}.csv
   └─> analysis/shuffled_sid/<Category>.index.json      ← basename 是硬约束（见 §8）
   └─> analysis/shuffled_sid/mapping.json
   └─> analysis/results/shuffled_sid_preflight.json     （18 项不变量）

scripts/sft_shuffled_sid.sh
  └─ source scripts/common.sh
  └─ check_exists "$PROJECT_ROOT/sft.py"
  └─ 内嵌 Python 断言：TokenExtender 解析出的路径 == --sid_index_path   ← 防 §8 的陷阱
  └─ torchrun --nproc_per_node 1 sft.py --train_file/--eval_file/--sid_index_path/--output_dir 四项替换

scripts/eval_shuffled_sid.sh
  └─ source scripts/common.sh
  └─ chk evaluate.py / calc.py（存在性断言）
  └─ INFO_MODE=original ⇒ 正式评测仍用**原始 info 文件**做约束
  └─ CUDA_VISIBLE_DEVICES=0 "$PY" -u "$PROJECT_ROOT/evaluate.py" --base_model … --K … --seed …
  └─ "$PY" "$PROJECT_ROOT/calc.py" --path … --item_path …
```
`` 【源码确认】``

### 2.4 核心配置
| 项 | 值 |
|---|---|
| 干预设计 | 3 655 个 singleton 在**训练频次桶内**错排；**31 个碰撞项冻结**以保碰撞结构 |
| seed | 42（确定性） |
| 数据 | `analysis/shuffled_sid/`（**注意：该数据集在 T-root，不在影子树内**） |
| 训练参数 | **与 `sft_full.sh` 逐项相同**，仅 4 个路径不同（train/eval/index/output） |
| 评测 `--K` / `--seed` / `--num_beams` / `--length_penalty` | 由 `common.sh` 变量提供 |

### 2.5 输入 / 输出 / 证据
| 项 | 值 |
|---|---|
| 输出 | `$RUN_ROOT/industrial_sft_shuffled_sid/`、`$RUN_ROOT/eval_shuffled_sid/{test_beam20.json, metrics.txt, evaluate.log}` |
| 产物（不在影子树内） | `minionerec_final_artifacts.tar.gz` → `runs/eval_shuffled_sid/test_beam20.json` |
| 证据等级 | **E2**（归档预测 + 归档 metrics 一致；**该归档是四个实验归档中唯一没有 manifest 的**） |
| 关键数值 | HR@20 `0.10765497`（相对 clean 的 `0.19832341` **−45.72 %**） |
| 权威记录 | `docs/notes/experiment_summary.md`（干预章）、`tools/analysis/results/clean_vs_shuffled_eval_recipe.md`、`tools/analysis/results/clean_vs_shuffled_sft_recipe.md` |

### 2.6 已知限制与缺失
* ⚠ **干预数据集不在影子树中**（在 T-root 的 `analysis/shuffled_sid/`，28.6 MB）。在影子树内运行需 `PROJECT_ROOT` 指向 T-root。
* 归档**无 manifest**（E2 而非 E1）。
* 31 个碰撞项**未受干预**，其行为无法被本实验解释。

### 2.7 经过验证的运行命令
**无**（本轮未运行）。

### 2.8 当前可运行性
| 检查 | 状态 |
|---|---|
| `bash -n` | ✅ 通过 |
| 参数与 `evaluate.py` 是否匹配 | ✅ **11/11 全部被接受**（含 `--K`、`--seed`） |
| 端到端可运行 | ⛔ **未验证**，且**缺干预数据集**（在 T-root） |

---

## 3. 实验三：Seq-only 消融

> **★ 深入阅读：[`experiments/E3_seq_only_ablation.md`](experiments/E3_seq_only_ablation.md)**
> 该文件是本组实验的**单一入口说明**：SFT 多任务数据流（79 834 = 36 259 + 7 316 + 36 259）、
> 三类任务的 prompt/response/标签构造、Response-only 交叉熵、Full 与 Seq-only 的**完整对照表**、
> 结果与**结论边界**（为什么不是单变量实验）、以及 1/3 分钟面试讲解与连续追问。
> 本节只列启动入口；设计与证据请读上面那份。

### 3.1 研究目的
移除两条**显式文本辅助监督**（`SidItemFeatDataset` + `FusionSeqRecDataset`），只保留 `SidSFTDataset`，检验干预效应是否依赖这些监督路径。

### 3.2 真实历史入口
| 臂 | 入口 | 世代 |
|---|---|---|
| clean | `scripts/sft_seq_only_clean.sh` → `scripts/eval_seq_only_clean.sh` | B |
| shuffled | `scripts/sft_seq_only_shuffled.sh` → `scripts/eval_seq_only_shuffled.sh` | B |
| 门禁 | `analysis/preflight_seq_only_ablation.sh` | B |

### 3.3 ★ 与完整 SFT 的差别**只有一个开关**
```
scripts/sft_seq_only_clean.sh:91   --sft_mode         seq_only
```
`sft.py:145` 的 `sft_mode: str = "full"` ⇒ `full` 开三个数据集，`seq_only` 只开 `SidSFTDataset`。
`` 【源码确认】``
**⇒ 差别就是这一行参数**；其余参数与 clean SFT 配置一致。

### 3.4 输入 / 输出 / 证据
| 项 | 值 |
|---|---|
| 训练行数 | **36 259**（vs 完整配方 **79 834**，**−54.6 %**） |
| 输出 | `$RUN_ROOT/seq_only_*_sft/`、`$RUN_ROOT/eval_seq_only_*/` |
| 产物（不在影子树内） | `seq_only_ablation_seed42_artifacts.tar.gz`（46 成员 + manifest） |
| 证据等级 | **E1 ×2** |
| 关键数值 | clean `0.17802780` / shuffled `0.10125745`；臂内相对下降 **−43.12 %** |
| 权威记录 | `docs/notes/experiment_summary.md` §14 |

### 3.5 已知限制与缺失
* ⚠ **不是单变量隔离**：`seq_only` 同时把训练行数从 79 834 降到 36 259。**该比较改变的不止一个变量。**
* 该臂的**大部分超参未记录**在归档 manifest 中（lr / micro-batch / accum / cutoff / 精度 / 优化器 / 调度器均缺）。归档内 `runs/seq_only_clean_sft/checkpoint-1083/training_args.bin` **可能存在**这些值，但本次未反序列化。

### 3.6 / 3.7 运行命令与可运行性
| 检查 | 状态 |
|---|---|
| `bash -n` | ✅ 通过 |
| 参数匹配 | ✅ 含 `--sft_mode`，被 `sft.py` 接受 |
| 端到端 | ⛔ **未验证** |

---

## 4. 实验四：Qwen2.5-1.5B 骨干规模对照

### 4.1 研究目的
把同一协议换到 **3.124× 更大**的骨干上，检验干预效应是否只是小模型现象。

### 4.2 ★ 真实历史入口：**不存在**
**经全树审计：影子树内没有任何 1.5B 的启动脚本或配置文件。**
全树检索 `1.5B` / `1p5b` / `Qwen2.5-1.5B` 仅命中 **2 处文档**（`README.md` §6b 标题、`docs/notes/experiment_summary.md` §20），**0 处脚本**。
`` 【源码确认】``

**⇒ 1.5B 的启动器在本仓库中从未存在，本文件不伪造它。**

### 4.3 可验证的配置（全部来自归档 manifest 与归档内 `config.json`）
| 项 | 值 | 来源 |
|---|---|---|
| 骨干 | `Qwen/Qwen2.5-1.5B`，revision `8faed761…`，source SHA-256 `a961db72…` | manifest 【文档记载】 |
| 架构 | hidden 1536 / 28 层 / 12 头 / vocab 152 225 | 归档内 `config.json` 【运行产物确认】 |
| 配方 | epochs 2、micro 4、accum 16、有效 batch 64、lr 3e-4、cutoff 512、seed 42、bf16、adamw_torch、线性调度、warmup 20、全参数 | manifest 【文档记载】 |
| 峰值显存 | 20.2285 GiB（两臂） | manifest 【文档记载】 |
| 失败记录 | 曾在 step-125 因 **ENOSPC** 失败并被排除 | 【文档记载】 |

**【合理推测】（非事实）：** 其配方与 `sft_full.sh` 逐项相同，唯一差别是 `BASE_MODEL` 指向 1.5B 权重。**但这只是推测，不得当作历史入口。**

### 4.4 输入 / 输出 / 证据
| 项 | 值 |
|---|---|
| 产物（不在影子树内） | `qwen2p5_1p5b_scaling_seed42_artifacts.tar.gz`（47 成员 + manifest） |
| 证据等级 | **E1 ×2** |
| 关键数值 | clean `0.19964703` / shuffled `0.11625855`；**−41.77 %**；0.5B→1.5B 干净仅 **+0.13 pp** |
| 权威记录 | `docs/notes/experiment_summary.md` §20 |

### 4.5 已知限制与缺失
* **★ 启动脚本缺失**（本组最大缺口）。
* **memory-probe 提交与 1.5B 无专属提交**：血缘中 `2839891` 的提交主题是 `add full-SFT memory probe hook`，1.5B 的**结果文档**写在**下一个**提交 `9c49f4c`。
* manifest 声明的 HEAD `2839891` 是 seq-only manifest HEAD `2758985` 的**后代** —— 两份声明的"运行时刻"无法同时成立。

### 4.6 / 4.7 运行命令与可运行性
**无法列出**（无入口脚本）。可运行性：**不可运行**（缺入口 + 缺权重 + 缺 GPU）。

---

## 5. 实验五：GRPO / R1 / R2（三组负结果）

> **★ 深入阅读：[`experiments/E4_grpo_negative_results.md`](experiments/E4_grpo_negative_results.md)**
> 该文件是本组实验的**单一入口说明**：GRPO 实际训练流程、关键源码位置、**reward 与 advantage 的计算示例**
> （含 G=16 的名次分值表、"无 GT 组"的 std 推导与 `0.031992` 的独立佐证）、五个实验版本的**差异表**、
> 主要负向结果、R1/R2 的研究逻辑链、实验局限与**证据等级**、1/3 分钟面试讲解与 **22 条连续追问**。
> 本节只列启动入口；设计与证据请读上面那份。

### 5.1 研究目的
在冻结的 17 516 行子集上做排序型 GRPO；R1 换 ReRe 式 rank reward；R2 加可达性路由与提示前缀监督。**三者结论均为负**（详见各入口对应的记录）。

### 5.2 真实历史入口
| 臂 | 训练入口 | 评测入口 | 世代 |
|---|---|---|---|
| **GRPO 0.25 ep 原始** | `patches/grpo_short025.sh` | （无独立评测脚本；评测产物缺失） | B |
| **GRPO 0.25 ep 优化** | `patches/grpo_fast025.sh` | `patches/…`（评测经 `evaluate.py`） | B |
| **GRPO 2.0 ep（baseline）** | `patches/grpo_baseline.sh` | — | B |
| **R1（ReRe rank reward）** | `patches/grpo_rere_rank025.sh` | **`eval_grpo_rere_rank.sh`** | B / A |
| **R2（可达性引导）** | `patches/grpo_r21.sh` | **`patches/eval_grpo_r21_valid.sh`** | B |
| 上游 RL 入口 | `rl.sh` | `evaluate.sh` | **A** |

### 5.3 调用链（以 `grpo_baseline.sh` 为例）
```
patches/grpo_baseline.sh
  └─ set -e
  └─ SCRIPT_DIR=… ; source "$SCRIPT_DIR/../scripts/common.sh"
  └─ CKPT="$RUN_ROOT/industrial_sft/final_checkpoint"      # 继承 P0 的 checkpoint
  └─ nohup accelerate launch --num_processes 1 rl.py \
         --model_path … --subset_dir "$SPLITS" --subset_seq grpo_seq_10k.json \
         --subset_seqtitle grpo_seqtitle_1k.json --reward_type ranking --num_generations 16 \
         --beam_search True --train_batch_size 16 --num_train_epochs 2 --learning_rate 1e-5 \
         --beta 1e-3 --temperature 1.0 --save_steps_frac 0.25 … > "$OUT/train.log" 2>&1 &
  └─ 写 pid.txt / started_at.txt
```
`rl.py` 侧：`from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset`；`from minionerec_trainer import ReReTrainer`；`from sasrec import SASRec`；`import rl_reward`、`import rere_reward`。
`` 【源码确认】``

### 5.4 核心配置（各臂的差异点）
| 臂 | `--reward_type` | 其他差异 |
|---|---|---|
| short025 / fast025 / baseline | `ranking` | 三者仅**批结构 / eval_step / save_steps_frac** 不同 |
| **R1** | **`rere_rank`** | 其余为 fast025 的严格副本 |
| **R2** | （`r21_enable` + `--route_cache`） | 额外需 `splits/r21_route_cache.json`；`--eval_step` |

**所有臂共有：** `--num_generations 16`、`--beam_search True`、`--beta 1e-3`、`--learning_rate 1e-5`、`--temperature 1.0`、`--subset_dir "$SPLITS"`（冻结 17 516 行）。

### 5.5 输入 / 输出 / 证据
| 臂 | 产物位置（均**不在影子树内**） | 证据等级 |
|---|---|---|
| GRPO 2.0 ep | `checkpoints/grpo_evidence/eval_grpo_baseline/test_beam20.json` | **E3** |
| GRPO 1.5 ep | `checkpoints/grpo_evidence/eval_grpo_step26274/test_beam20.json` | **E3**（注：这是 2.0 ep 运行的中点，**不是独立运行**） |
| GRPO 0.25 优化 | `_backup_v1/freeze_v1_records.tar.gz` → `predictions/optimizedGRPO_test_eval_beam20.json` | **E1'**（含 32 项 SHA256 清单） |
| **GRPO 0.25 原始** | ★ **预测产物**全仓库与全部归档中均不存在（**但 checkpoint 与逐步日志在**：`checkpoints/grpo_evidence/grpo_short025/checkpoint-4379/` + `train.log` + `_EXTRACTED_metrics/grpo_short025_metrics.txt`） | **E4**（历史指标不可复算） |
| R1 | 同上 → `predictions/R1_test_beam20.json` | **E1'**（**权重不在本地**，仅哈希钉住） |
| R2 | 同上 → `predictions/R2_valid_predictions.json` | **E1'**（**validation split**） |

**权威记录：** `docs/provenance/rl_posttraining_results.md`（§1–§4）、`docs/notes/experiment_summary.md` §6。

### 5.6 已知限制与缺失
* **★ 0.25 ep 原始臂的预测产物不存在** —— 其 HR@20 `0.17626296` **只有文档记录**；★ **但其 checkpoint（`grpo_short025/checkpoint-4379/model.safetensors`，1 977 199 248 B）、`trainer_state.json`、`train.log` 与 4 379 行逐步诊断日志都在本地** ⇒ **缺的是评测产物，不是权重。**
* **R1/R2 的模型权重不在本地**（仅哈希 `7ebdf9c1…` / `a983f46b…`）；★ **R1/R2 也没有逐步诊断日志**（`_EXTRACTED_metrics/` 只有 short025/fast025/baseline/smoke 四个）⇒ 其逐步指标为台账转述。
* **R2 用的是 validation split**（4 532），**与其他 test-split 表不可混用**。
* **R2 同时改了奖励与 prompt 路由，无消融分离** ⇒ 其回归原因记为 **UNKNOWN**。
* **R1 的机制解释是假设**（无对照 run）；★ **其梯度方向为"反惩罚模型自身排序"**（见 `experiments/E4_…` §3.4）。
* 零优势组比例在 GRPO 三臂约 **69.9 – 70.2 %**；★ 因**每步只有一个 prompt 组**（`num_groups = 1.0`），该比例**就等于"一组 16 条候选全部未命中 GT"的步占比**。
* ★ **零优势比只能被否定其"充分性"**；它是否构成长期训练的瓶颈 **未做预算匹配对照 ⇒ UNKNOWN**。

### 5.7 经过验证的运行命令
**无**（本轮未运行）。

### 5.8 当前可运行性
| 检查 | 状态 |
|---|---|
| `bash -n` | ✅ 通过 |
| 参数与 `rl.py` 是否匹配 | ✅ 各臂所传 flag 均被 `rl.py` 接受（见 `phaseH/_launcher_params_accurate.csv`） |
| 端到端 | ⛔ **未验证**（缺 `accelerate`/`trl`；且需先有 P0 checkpoint） |

---

## 6. 其他（非五组核心实验，但属正式入口）

| 入口 | 用途 | 性质 |
|---|---|---|
| `baselines/sasrec_sweep.sh` → `baselines/sasrec_baseline.py` | **SUPPORT**：compact SASRec 对照基线（双协议） | 参数经 `"$@"` 透传；`--tag` 为唯一 argparse 参数，其余经配置字典 |
| `scripts/benchmark_inference.sh` → `analysis/benchmark_inference.py` | **ENGINEERING**：推理质量–延迟基准 | 加固世代；`--mode` / `--out-dir` / `--preflight` |
| `scripts/audit.sh` | 全仓**只读审计**（语法、CRLF、路径一致性等） | 加固世代；不产生实验数据 |
| `analysis/preflight_*.sh`（3 个） | 三个实验臂的**静态门禁** | 拒绝在不一致时启动 |
| `patches/preflight_grpo.py` | GRPO 子集与 `sample_id` 绑定的门禁 | 被 4 个 GRPO 启动器引用 |
| `convert_dataset.sh` / `data/amazon18_data_process.sh` / `data/amazon23_data_process.sh` | 上游数据准备 | **世代 A**；输入 `*.inter` **不在仓库中**，**不可运行** |
| `rq/*.sh`（5 个） | 上游 SID 构造 | **世代 A**；**无任何运行证据**（见 `phase0/OPEN_EVIDENCE.md`） |

---

## 7. ★ 已确认的参数不匹配（诊断，未修复）

| 项 | 内容 |
|---|---|
| **位置** | `evaluate.sh:51-52` |
| **传入** | `--temperature 1.0`、`--guidance_scale 1.0` |
| **`evaluate.py` 参数表** | `evaluate.py:41-56` 共 14 个参数，**不含这两个** |
| **经验验证** | 在 `fire==0.7.1`（本仓库 `requirements.txt` 钉住的版本）下，**未知 flag 不是被忽略，而是报错并 `exit 2`**：`ERROR: Could not consume arg: --temperature` |
| **⇒ 结论** | **`evaluate.sh` 的评测步骤必然失败**，不会产生结果。（`evaluate.sh` 是**上游文件**，三树逐字节相同） |
| **旁证** | 本项目的正式评测**没有**用 `evaluate.sh`，而是用世代 B 的 `eval_industrial.sh` / `scripts/eval_*.sh` |
| **本轮处置** | **只诊断，不修复**（按 Phase H 约束）。**也不通过新增兼容启动器来"绕过"** —— 那会掩盖这个缺陷 |

**另一处已确认的行为（非参数问题）：**
`evaluate.py:59` 无条件执行 `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`。
⇒ 任何**外部**的 GPU 指定（`evaluate.sh` 的 `CUDA_VISIBLE_DEVICES=$i`、`scripts/eval_*.sh` 的 `CUDA_VISIBLE_DEVICES=0`）**都在 `evaluate.py` 内部被覆盖为 `0`**。多卡并行评测因此**不会**分布在多张卡上。**本轮不修改该行为。**

**另一处已确认的代码缺陷（会使某条路径直接崩溃）：**

| 项 | 内容 |
|---|---|
| **位置** | `sft.py:200, 205, 211` |
| **问题** | `original_vocab_size` 被**读取三次**，但在整个 `sft.py` 中**从未被赋值**（该变量只在未使用的 `sft_gpr.py` 里定义） |
| **触发条件** | 仅当 `--freeze_LLM True` 时进入该分支；`--train_from_scratch False` 且 `--freeze_LLM False` 的正式配方**不会**触发 |
| **后果** | 该路径一执行即 **`NameError`** —— 不是静默错误，而是崩溃 |
| **影响面** | 本项目全部正式 SFT 运行都用 `--freeze_LLM False`（见 §1.4），**因此历史结果未受影响** |
| **本轮处置** | **只记录，不修复**（属核心训练代码；本轮未授权修改） |
| **建议修法（未实施）** | 在 `tokenizer.add_tokens(...)` **之前**加 `original_vocab_size = len(tokenizer)`（镜像 `sft_gpr.py:188` 的做法） |

**★ 本节标题说明：** 上述三项中只有第一项是"参数不匹配"；后两项是已确认的**行为**与**代码缺陷**，一并记录在此，以免读者以为只有一处问题。

---

## 8. 两条不得被改变的硬约束（迁移时）

| # | 约束 | 原因 |
|---|---|---|
| **1** | **SID 文件名必须保持 `<Category>.index.json`** | `sft.py` 的 `TokenExtender` **不按 `--sid_index_path` 原样打开**，而是重建为 `dirname/basename.split('.')[0] + ".index.json"`。改名成 `index.json` 会去找 `index.index.json` 而崩溃。`scripts/common.sh:45-51` 与 `analysis/_paths.py:50-55` 都以注释固化了这一点，`scripts/sft_shuffled_sid.sh` 还内嵌断言防此陷阱。 |
| **2** | **`ConstrainedLogitsProcessor` 的有状态行为不得改变** | 其 `self.count` 是**有状态**计数器，**每个 `generate()` 调用必须新建实例**。改成单例或缓存会**静默改变解码掩码**。 |

**两条路径解析库必须同深度、语义一致：** `scripts/common.sh`（shell）与 `analysis/_paths.py`（Python）。**不得分别独立重构。**

---

## 9. 方法与限制

**方法：** 对影子树 213 个文件做静态审计 —— AST 解析 116 个 `.py` 的函数签名与导入图、正则提取 38 个 `.sh` 的 `python X.py` 调用与 flag、逐 launcher 与目标参数表比对、全树引用检索。

**限制（必须与结论一同引用）：**
1. **本轮未运行任何训练、推理或评测**；`bash -n` **只证明 shell 语法有效，不证明命令能成功执行**。
2. **参数一致性检查是静态的**：它证明"所传 flag 存在于目标函数的签名中"，**不证明**运行期取值正确、默认值合适或路径存在。
3. **G4（导入自检）/ G5（指标重投）本轮"未执行"**（环境缺 `fire`/`torch`/`transformers`/`trl`/`accelerate`，且无 GPU/权重）。**不得据此声称运行行为一致。**
4. **未验证**：所有"端到端可运行"均标 ⛔。
5. **不在影子树内的对象**（SID 数据集、checkpoint、预测产物、归档）只作**引用**，本轮未读取或修改。
6. **1.5B 入口缺失是事实，不是省略** —— 本文件不伪造它。

**相关证据文件（仓库外）：** `MiniOneRec-review/phaseH/`。
