# docs/00_INDEX.md — 文档导航索引

> **本文件是 Phase C / P1a（文档迁移试点）在影子树中新增的文件。**
> 它**只做定位**：告诉读者某项内容在**哪个文件里**。
> **它不引入任何新结论、不引入任何新数值**，全部数值与证据等级仍以被链接的原文件、`notes` 台账与 `analysis/results/` 原始产物为准。
> 新增原因：P1a 把 6 份文档移入 `docs/` 结构，原 `README.md` 的 5 个相对链接随之失效（见 `04_LINK_AUDIT.md`）。
> **本索引尚未获得"内容一致性"背书**：详见文末"已知的文档状态注记"。

---

## 0. 如何读这份索引

> **★ 想找"从哪启动实验"？请先看 [`docs/ENTRYPOINTS.md`](ENTRYPOINTS.md)** —— 它按五组核心研究实验（Clean SFT / shuffled-SID / seq-only / 1.5B / GRPO·R1·R2）列出真实入口、调用链、配置、产物与已知缺口。
> 本文件负责**文档层**的导航；`ENTRYPOINTS.md` 负责**执行入口层**的导航。

| 标记 | 含义 |
|---|---|
| **证据等级 E1–E4** | 见 `docs/provenance/` 与被链接文件自身的声明；**本轮未重新评定** |
| **路径为阴影树路径** | 与 `migration_plan/04_MIGRATION_MATRIX.csv` 的目标路径一致 |
| **[未执行]** | 受环境限制本轮无法执行的门禁，**不表示通过** |

**三条根本限定（来自被链接文档自身的声明，此处仅作提示，不新增结论）：**
单 seed；单类目；beam-restricted 指标。

---

## 1. 主实验（MAIN）

| 实验 | 权威记录位置 | 相关原始产物（**未迁移，仍在原路径**） |
|---|---|---|
| **干净 SFT 基线** | `docs/notes/experiment_summary.md` §5 | `checkpoints/grpo_evidence/eval_clean_sft/test_beam20.json` |
| **★ shuffled-SID 受控干预** | **★ 先读 [`experiments/E2_shuffled_sid.md`](experiments/E2_shuffled_sid.md)**（单一入口：设计/代码/证据/边界）<br>完整记录：`docs/notes/experiment_summary.md` §10、§14<br>逐参数 diff：`tools/analysis/results/clean_vs_shuffled_{sft,eval}_recipe.md` | `minionerec_final_artifacts.tar.gz` → `runs/eval_shuffled_sid/` |
| **seq-only 消融** | **★ 先读 [`experiments/E3_seq_only_ablation.md`](experiments/E3_seq_only_ablation.md)**（单一入口：多任务数据流/源码定位/Full对Seq-only/边界）<br>完整记录：`docs/notes/experiment_summary.md` §14 | `seq_only_ablation_seed42_artifacts.tar.gz` |
| **1.5B 规模检查** | `docs/notes/experiment_summary.md` §20 | `qwen2p5_1p5b_scaling_seed42_artifacts.tar.gz` |
| **正式评测协议 / 约束解码** | `docs/notes/experiment_summary.md` §4；`docs/provenance/reproducibility.md` | `checkpoints/industrial_eval/`（旧协议）；`data/Amazon/index/*.index.json` |

---

## 2. 支撑实验（SUPPORT）

| 项目 | 权威记录位置 | 必须保留的定性 |
|---|---|---|
| **Compact SASRec 基线** | `docs/notes/experiment_summary.md` §8 | **对照基线**；含未受控变量（SFT 另有两条文本辅助监督目标）。**不是"受控的同机制击败"** |
| **SID 观测性分析**（流行度分层 / 前缀亲和 / 匹配控制） | `docs/notes/experiment_summary.md` §9 | **观测性、相关性，非因果** |
| **推理质量–延迟基准** | `docs/notes/experiment_summary.md` §16 | **工程内容**（部署工作点选择），非研究贡献 |
| **Legacy（pre-unification）协议** | `docs/notes/experiment_summary.md` §13 | **已被取代，作者主动排除于全部结果表** |

---

## 3. 负向实验（NEGATIVE）

> **★ 先读 [`experiments/E4_grpo_negative_results.md`](experiments/E4_grpo_negative_results.md)** —— GRPO / R1 / R2 的**单一入口说明**：
> 真实训练流程、reward 与 advantage 的精确算术（含名次分值表与"无 GT 组"的 std 推导）、五个实验版本的差异表、
> R1/R2 的研究逻辑链、证据等级，以及 22 条连续追问。本节只列权威记录位置与证据状态。

| 实验 | 权威记录位置 | 证据状态 |
|---|---|---|
| **GRPO 基准（test split，多个 epoch 配置）** | `docs/notes/experiment_summary.md` §6；`docs/provenance/rl_posttraining_results.md` §1–§2 | 有产物；**其中 0.25-epoch 的"原始（未优化）臂"预测产物在全仓库与全部归档中均不存在** |
| **R1 — ReRe 式 rank reward** | `docs/provenance/rl_posttraining_results.md` §3 | 预测 / 指标 / 训练日志保存在 `_backup_v1/freeze_v1_records.tar.gz`；**权重不在本地（仅哈希钉住）** |
| **R2 — reachability-guided GRPO** | `docs/provenance/rl_posttraining_results.md` §4 | 同上；**validation split**，与 test split 数字不可混用 |
| **LETTER 可学习 tokenizer 对照** | `docs/provenance/letter_sid_experiment.md` | ⚠ **全仓库与全部归档中不存在任何 LETTER 预测 / 指标 / 日志 / checkpoint。该对照的数字只存在于散文，不可独立复算。** 不得与其他有产物的负结果不加标注地并列 |

**已被作者保留的关键自我限定（原文位置见上表）：**
* R1 的机制解释标注为**假设**（无对照 run）。
* R2 **同时**改了奖励与 prompt 路由，**无消融把两者分开**，回归原因标注为 **UNKNOWN**。
* 全部 RL 运行**单 seed**，无显著性检验。

---

## 4. 工程与可复现（ENGINEERING）

| 项目 | 权威记录位置 | 定性 |
|---|---|---|
| GRPO 训练效率优化 | `docs/notes/experiment_summary.md` §7 | **工程优化**；归因已在后续文档中**更正**（省下的主要是训练中全量验证） |
| 可复现基础设施（preflight / 稳定 `sample_id` / 产物哈希） | `docs/provenance/reproducibility.md` | 基础设施，非研究结论 |
| 协议审计纠正（epoch 标签 / NDCG 对数底 / `length_penalty` 死代码 / pair-retention 定义） | `docs/notes/experiment_summary.md` §13 | 纠错记录 |

---

## 5. 历史实验与已弃用内容（LEGACY）

| 内容 | 位置 | 定性 |
|---|---|---|
| **2026-06 更早世代**（Office_Products / Steam / rule-ranking 等） | **不在阴影树内**；位于主仓库 `experiment_result/`、`experiment_result_all/` | **无任何产物**；内部无日期/提交/seed；部分数字与冻结结果冲突；作者已判定弃用。**不得与冻结结果同表并列。** 计划归档至 `deprecated/early_gen_2026_06/`（**P1c，未执行**） |
| 作者自审三报告（`MINIONEREC_*.md`） | **不在阴影树内**；位于主仓库根 | 计划归档至 `deprecated/self_audit/`（**P1c，未执行**） |
| 上游 GPR 死路径（`*_gpr.py`、`SidSFTDataset_GPR`） | 阴影树内仍在原路径 | 计划归档至 `deprecated/upstream_gpr_path/`（**P2，未执行**） |
| 上游 SID 构造（`rq/**`） | 阴影树内仍在原路径 | 计划归档至 `deprecated/upstream_sid_construction/`（**P2，未执行**）。**该子树是在用 SID 归属问题的证据，不得删除** |

---

## 6. 证据限制与环境限制

| 主题 | 位置 | 说明 |
|---|---|---|
| 证据等级定义（E1–E4）与本项目已知缺口 | `docs/provenance/reproducibility.md`；`notes` 台账 §13、§18 | E1–E4 **只评级"预测产物可复算程度"，不是端到端训练复现等级** |
| **SID 生成算法归属** | `docs/provenance/letter_sid_experiment.md`（其表述） + 项目自审文档 | **仍为未确认。** 本索引不提供归属结论 |
| **仅原始实例可解决的问题** | 审查目录 `storyline/AUTODL_MISSING_EVIDENCE.md`（**在仓库外**） | 列明缺什么、以及为何本地不可替代 |
| **本轮受环境限制未执行的门禁** | 审查目录 `phaseC/05_VALIDATION_RESULTS.md`（**在仓库外**） | 逐项标注 **[未执行]**，不得记为通过 |

---

## 7. 已知的文档状态注记（P1a 之后**需要但尚未**跟进的修正）

> 以下不是新结论，而是**"移动之后，某些文档的自我描述已不再与其文件位置相符"**这一事实的记录。
> **本轮未修改这些正文**（正文逐字节保持不变）。是否修正由 Review Lead 决定。

| # | 文件 | 需要跟进的描述 | 原因 |
|---|---|---|---|
| N-1 | `README.md` | 5 个相对链接失效（4 处指向 `notes/experiment_summary.md`、1 处指向 `docs/rl_posttraining_results.md`）；另有 2 处反引号路径提及同一批文件 | 目标文件已按迁移矩阵移入 `docs/notes/` 与 `docs/provenance/` |
| N-2 | `docs/notes/experiment_summary.md`（原 `notes/`） | 其自我描述为"**single authoritative record**"、以及若干"逐字节一致"类表述 | 该文件已改变路径；且新增的索引/登册属于**增量**记录，不改写其权威性，但需注明路径变更 |
| N-3 | `docs/provenance/reproducibility.md` | 文中以相对路径引用 `notes/experiment_summary.md` 的位置 | 该链接**当前仍可解析**（两文件同批移动、相对关系保持），无需修改；仅登记 |
| N-4 | `tools/README.md` | 以反引号提及 `docs/letter_sid_experiment.md` | 该文件计划在 **P2** 归档至 `deprecated/letter_tools/`，届时该提及需一并处理 |
| N-5 | `tools/analysis/results/*.md`（已移动的两份 recipe） | 正文以反引号提及仍在 `analysis/` 下的脚本与同名目录产物 | 反引号是**说明性文字**而非链接；**本次审计未发现这两份文件含任何 Markdown 链接**，故不构成断链 |

---

## 8. 索引自身的位置与适用边界

* 本索引位于**阴影树**（`MiniOneRec-review/phaseC/shadow/MiniOneRec/`），**不在原始仓库中**。
* 原始 `_verify_v1/` 与主仓库**未被修改**（见 `05_VALIDATION_RESULTS.md` G0）。
* 本索引**未**覆盖主仓库中尚存的 215 个未跟踪条目（早世代文档、归档、`_backup_v1/` 等）；它们的处置属 P1c/P2 及后续批次。
