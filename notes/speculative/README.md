# 投机解码：提案、验证与跨轮状态

[总索引](../README.md) · [可视化入口](visualizations/index.html) · [全家族课程](visualizations/sglang-course-v2/index.html)

投机解码跨越调度、模型执行、接受算法与 KV 生命周期。它作为独立专题组织；前置知识是 run_batch 的 worker 交接、ForwardBatch 与 runner、logits 与采样和 ordinary Prefill→Decode；这些前置概念可从 [SGLang 全景图](../index.html)定位。

## 主文与课程的分工

| 阅读目的 | 主文 | 衔接的交互课程 |
|---|---|---|
| 确定共同入口和算法边界 | [Draft→Target Verify 生命周期](sglang-draft-target-verify-lifecycle.md) | [共同合同](visualizations/sglang-course-v2/common-contract.html) · [EAGLE](visualizations/sglang-course-v2/eagle-tree.html) · [dFlash](visualizations/sglang-course-v2/dflash-block.html) |
| 完整追踪 dSpark 一轮及下一轮 | [dSpark 单轮 Decode](sglang-dspark-single-round-decode.md) | [执行地图](visualizations/dspark-complete-execution-flow.html) |
| 用固定坐标核对 tensor/slot | [Verify 图解练习](sglang-dspark-verify-visual-guide.md) | [Target Verify 显微镜](visualizations/dspark-verify-step-through.html) |
| 启动或迁移时查约束 | [启动参数与迁移参考](sglang-dflash-dspark-startup-args-and-migration.md) | [算法边界矩阵](visualizations/sglang-course-v2/boundary-matrix.html) |

普通路径的前置概念由全景图定位，后续按主题重新学习；本专题保留共享合同、算法自己的分支和状态写入。greedy、Target-only sampling 与使用真实 q 的拒绝采样必须分别核对，不能因页面标题相似就互相套用。

## dSpark 深入路线

主阶段负责本阶段输入、调用、状态和交接；就近专题拥有对应算法推导或跨轮机制。固定案例以各页明确给定的 `B/G/W` 和 token/KV 坐标为准。

| 阶段 | 主阶段页面 | 就近深入 |
|---|---|---|
| 00 | [完整执行地图](visualizations/dspark-complete-execution-flow.html) | 先确定对象所有权和一轮闭环 |
| 01 | [Scheduler 隔离](visualizations/dspark-forward-isolation.html) | [overlap / non-overlap](visualizations/dspark-overlap-nonoverlap-timeline.html) |
| 02 | [Prefill bootstrap](visualizations/dspark-prefill-bootstrap.html) | Target hidden 建立 Draft KV 和首轮 anchor |
| 03 | [Draft 提案](visualizations/dspark-draft-proposal.html) | [Markov Head](visualizations/dspark-markov-head-internals.html) |
| 04 | [Confidence / Budget / Layout](visualizations/dspark-confidence-budget-layout.html) | [历史预算与 SPS](visualizations/dspark-budget-history-sps.html) · [Survival / Top-k](visualizations/dspark-survival-topk-layout.html) |
| 05 | [Target Verify](visualizations/dspark-verify-step-through.html) | rectangle → compact → physical rows → strided |
| 06 | [Accept / Finalize](visualizations/dspark-accept-finalize.html) | [采样修正实验](visualizations/dspark-sampling-rejection-lab.html) |
| 07 | [Hidden / KV / Rolling anchor](visualizations/dspark-target-hidden-kv-explainer.html) | 输出、KV commit 和下一输入分别记账 |
| 08 | [Folded 快路](visualizations/dspark-folded-fast-path.html) | [Capture / Replay](visualizations/dspark-folded-capture-replay.html) |

第一次沿主阶段进入，需要推导时打开就近专题；复习时从问题进入。[全家族课程](visualizations/sglang-course-v2/index.html)串联算法视野与跨阶段实验，dSpark 深入站承接具体张量和时序交互。

## 来源与维护

- [dSpark 论文](references/DSpark-Confidence-Scheduled-Speculative-Decoding-arXiv-2607.05147.pdf)是论文参考；实现结论以对应主文的源码核对为准。
- [可视化维护说明](visualizations/README.md)区分共享 JavaScript 教学数据、课程生成源和历史快照。
- 主文保留各自核对日期和验证范围。当前文档组织检查不增加 GPU 实验或算法正确性证据。
