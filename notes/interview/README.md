# SGLang 推理岗位面试学习题库

这套题库覆盖推理引擎研发与性能优化、模型推理算法与算子优化、推理平台与大规模部署。学习目标是从机制和具体例子出发，追到当前源码中的状态变化，再用性能测量、正确性检查和故障处置回答生产追问。

50 道题沿用讨论中的 Q01–Q50，每题仅有一个主答案位置。岗位分类决定阅读入口，共同知识通过题号互相链接。每题包含问题与追问、详细参考答案、源码调用链与证据、生产实践追问及学习检查点。

[返回 notes 学习入口](../README.md) · [模块全景图](../index.html) · [实践任务与记录方式](practice.md) · [资料与源码基线](references.md)

## 三个岗位方向

| 方向 | 主文档 | 题目范围 |
| --- | --- | --- |
| 推理引擎研发与性能优化 | [推理引擎研发与性能优化](engine-development.md) | Q01、Q07–Q20、Q47–Q49，18 题 |
| 模型推理算法与算子优化 | [模型推理算法与算子优化](algorithms-kernels.md) | Q02、Q04、Q21–Q34，16 题 |
| 推理平台与大规模部署 | [推理平台与大规模部署](platform-deployment.md) | Q03、Q05、Q06、Q35–Q46、Q50，16 题 |

## 如何逐题学习

先学习共同基础 Q01–Q06，再选一个岗位方向推进；相关题可以跳转，不需要把所有术语读完再开始。一题的顺序是：先给出自己的 90 秒回答，再讨论参考答案和例子，打开源码确认关键状态，最后完成检查点。复杂题可以分几次讨论。

每道题的参考答案是准备材料，表中的学习状态只记录已经讨论和验证的进度。初始状态统一为“未开始”；只有实际讨论后才更新为“学习中”“待复习”或“已掌握”。读完答案本身不等于完成实践。

- 引擎方向先走 Q01 → Q07–Q17，再补 Q18–Q20、Q29–Q34，用 Q47–Q49 做综合推演。
- 算法与算子方向先走 Q02、Q04、Q21–Q28，再走 Q29–Q34；Q13–Q17 用来连接 kernel 与实际执行器。
- 平台方向先走 Q03、Q05、Q06、Q35–Q46，再用 Q47、Q48、Q50 做故障推演；Q07–Q12 帮助解释引擎内部背压。

## 50 题总目录与学习状态

★ 表示第一轮优先掌握。题号是稳定引用，不随岗位分类重新编号。

### 推理引擎研发与性能优化

| 题号 | 问题 | 优先级 | 学习状态 |
| --- | --- | --- | --- |
| Q01 | [一条流式请求如何完成生成和资源回收](engine-development.md#q01) | ★ | 未开始 |
| Q07 | [Scheduler 请求与批次的连续两轮状态](engine-development.md#q07) | ★ | 未开始 |
| Q08 | [显存空闲时 Prefill 为何仍被拒绝](engine-development.md#q08) | ★ | 未开始 |
| Q09 | [Chunked Prefill Mixed 与 Decode 公平性](engine-development.md#q09) | ★ | 未开始 |
| Q10 | [Eviction 与 retraction 的回收和恢复](engine-development.md#q10) | ★ | 未开始 |
| Q11 | [KV 分配映射和有效内容的区别](engine-development.md#q11) | 深入 | 未开始 |
| Q12 | [共享前缀的引用锁与资源所有权](engine-development.md#q12) | ★ | 未开始 |
| Q13 | [Worker 和 ModelRunner 的实际执行分派](engine-development.md#q13) | ★ | 未开始 |
| Q14 | [Attention metadata 的预规划与重规划](engine-development.md#q14) | ★ | 未开始 |
| Q15 | [Decode CUDA Graph 的静态地址与动态输入](engine-development.md#q15) | ★ | 未开始 |
| Q16 | [Prefill Graph 和两种 prefill 切分](engine-development.md#q16) | 深入 | 未开始 |
| Q17 | [Overlap 的跨轮依赖和读写保护](engine-development.md#q17) | ★ | 未开始 |
| Q18 | [Grammar 约束和采样状态如何推进](engine-development.md#q18) | 深入 | 未开始 |
| Q19 | [新模型的接入与权重加载契约](engine-development.md#q19) | 深入 | 未开始 |
| Q20 | [投机与多种并行策略的兼容边界](engine-development.md#q20) | ★ | 未开始 |
| Q47 | [TTFT 翻倍而 ITL 稳定的定位](engine-development.md#q47) | ★ | 未开始 |
| Q48 | [长输入造成 Decode 卡顿的定位](engine-development.md#q48) | ★ | 未开始 |
| Q49 | [批处理 Graph Overlap 投机下的偶发错 token](engine-development.md#q49) | ★ | 未开始 |

### 模型推理算法与算子优化

| 题号 | 问题 | 优先级 | 学习状态 |
| --- | --- | --- | --- |
| Q02 | [Prefill 与 Decode 的瓶颈为何不同](algorithms-kernels.md#q02) | ★ | 未开始 |
| Q04 | [KV 容量和并行分片如何估算](algorithms-kernels.md#q04) | ★ | 未开始 |
| Q21 | [注意力架构如何改变计算与缓存](algorithms-kernels.md#q21) | ★ | 未开始 |
| Q22 | [FlashAttention 与 online softmax 的两块推导](algorithms-kernels.md#q22) | ★ | 未开始 |
| Q23 | [Decode attention 的 split KV 取舍](algorithms-kernels.md#q23) | ★ | 未开始 |
| Q24 | [算子 Roofline 和延迟下界](algorithms-kernels.md#q24) | ★ | 未开始 |
| Q25 | [Kernel 融合的依赖与资源成本](algorithms-kernels.md#q25) | 深入 | 未开始 |
| Q26 | [量化的收益成本和质量验证](algorithms-kernels.md#q26) | ★ | 未开始 |
| Q27 | [Tensor Parallel 的张量切分与通信](algorithms-kernels.md#q27) | ★ | 未开始 |
| Q28 | [MoE 从路由到 combine 的整层成本](algorithms-kernels.md#q28) | ★ | 未开始 |
| Q29 | [投机无损的 greedy 与 sampling 含义](algorithms-kernels.md#q29) | ★ | 未开始 |
| Q30 | [EAGLE 树接受后的 KV 和状态压紧](algorithms-kernels.md#q30) | ★ | 未开始 |
| Q31 | [dFlash 和 dSpark 的提案与验证计数](algorithms-kernels.md#q31) | ★ | 未开始 |
| Q32 | [Target hidden 怎样建立 Draft KV](algorithms-kernels.md#q32) | 深入 | 未开始 |
| Q33 | [接受长度与投机加速收益的区别](algorithms-kernels.md#q33) | ★ | 未开始 |
| Q34 | [投机解码的正确性验证矩阵](algorithms-kernels.md#q34) | ★ | 未开始 |

### 推理平台与大规模部署

| 题号 | 问题 | 优先级 | 学习状态 |
| --- | --- | --- | --- |
| Q03 | [TTFT TPOT ITL 吞吐和 goodput 的统计口径](platform-deployment.md#q03) | ★ | 未开始 |
| Q05 | [如何公平比较配置与推理引擎](platform-deployment.md#q05) | ★ | 未开始 |
| Q06 | [SGLang Dynamo Mooncake NIXL 的职责与组合](platform-deployment.md#q06) | 深入 | 未开始 |
| Q35 | [PD 分离的成本和适用负载](platform-deployment.md#q35) | ★ | 未开始 |
| Q36 | [PD 交接的状态与完成门槛](platform-deployment.md#q36) | ★ | 未开始 |
| Q37 | [Mooncake Transfer Engine 和 Store 的区别](platform-deployment.md#q37) | ★ | 未开始 |
| Q38 | [HiCache 搬运事件引用锁和 ACK](platform-deployment.md#q38) | ★ | 未开始 |
| Q39 | [KV aware routing 的缓存与负载权衡](platform-deployment.md#q39) | ★ | 未开始 |
| Q40 | [Dynamo Planner 的扩缩容信号与反馈](platform-deployment.md#q40) | 深入 | 未开始 |
| Q41 | [并行策略与 P D 配比如何选择](platform-deployment.md#q41) | ★ | 未开始 |
| Q42 | [TCP RDMA 与真实 PD 的验证层级](platform-deployment.md#q42) | ★ | 未开始 |
| Q43 | [多机 hang 的 rank 和通信排查](platform-deployment.md#q43) | ★ | 未开始 |
| Q44 | [传输中取消请求的安全回收](platform-deployment.md#q44) | ★ | 未开始 |
| Q45 | [模型与引擎的滚动升级和回滚](platform-deployment.md#q45) | ★ | 未开始 |
| Q46 | [多租户长请求的预算和背压](platform-deployment.md#q46) | ★ | 未开始 |
| Q50 | [Prefill 已完成但 bootstrap 超时的定位](platform-deployment.md#q50) | ★ | 未开始 |

## 练到什么程度

| 层次 | 能给出的回答 | 可以保留的学习证据 |
| --- | --- | --- |
| 机制理解 | 解释问题、状态变化、收益和成本；用具体数值走一遍 | 自己画的状态图、公式与口述 |
| 源码理解 | 找到运行分派、关键写入、返回值和下一消费者 | 当前文件、符号、行号与精简调用链 |
| 实验能力 | 提出可证伪假设，固定对照条件，检查正确性和目标指标 | 可复现配置、日志、曲线、trace 与原始结果 |
| 生产判断 | 处理容量、尾延迟、取消、故障、灰度和回滚 | 故障时间线、恢复边界、资源回收与发布验收记录 |

[六项实践任务](practice.md)把这些层次连接起来。复述经历时标明版本、硬件、负载、原始问题、对照证据、结果与限制，并区分源码分析、单机实验、多机验证和线上运行。

## 源码与外部资料的使用范围

源码锚点核对于 2026-10-15 的本地工作树，HEAD 为 `f01a0c7f97`，包含现有未提交修改。行号配合函数和类名使用；换版本后先检查分派、状态契约及配置 gate。本文集给出的实验均是备考任务，没有据此增加 GPU 或真实 PD 的运行结论。

普通运行链以 CUDA 通用 `TpModelWorker` 为主要情景。MLX 会改派 worker；ROCm、NPU、XPU、DP/PP/CP、Mixed、EAGLE/dFlash/dSpark 的差异在相关题中定位，不能把一个分支的性质扩展成所有组合的保证。

社区面经提供问题线索，技术答案依当前源码、原始论文和官方资料。Dynamo 与 Mooncake 不在本次 SGLang 源码树中的部分，以外部设计流程和集成交界说明，不能当作已经在本地追过的调用链。详见[资料与源码基线](references.md)。
