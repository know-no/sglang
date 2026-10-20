# SGLang 推理系统学习题库

这套题库覆盖推理引擎研发与性能优化、模型推理算法与算子优化、推理平台与大规模部署。学习目标是从机制和具体例子出发，追到当前源码中的状态变化，再用性能测量、正确性检查和故障处置回答生产追问。

共 68 道题：保留 Q01–Q50，新增 Q51–Q68。新增题从补充社区资料中提取硬件、执行、算子和部署决策的知识点，以故障现象、计算任务和反例重新出题；每题仅有一个主答案位置。专题分类决定阅读入口，共同知识通过题号互相链接。每题先说明考察重点、知识点、能力要求和常见误区，再给问题与追问、完整参考回答、源码依据、生产场景问答以及带解答的学习检查点。

[返回 notes 学习入口](../README.md) · [模块全景图](../index.html) · [实践任务与记录方式](practice.md) · [资料与源码基线](references.md)

## 三个技术方向

| 方向 | 主文档 | 题目范围 |
| --- | --- | --- |
| 推理引擎研发与性能优化 | [推理引擎研发与性能优化](engine-development.md) | Q01、Q07–Q20、Q47–Q49、Q52–Q55、Q63，23 题 |
| 模型推理算法与算子优化 | [模型推理算法与算子优化](algorithms-kernels.md) | Q02、Q04、Q21–Q34、Q56–Q62、Q68，24 题 |
| 推理平台与大规模部署 | [推理平台与大规模部署](platform-deployment.md) | Q03、Q05、Q06、Q35–Q46、Q50、Q51、Q64–Q67，21 题 |

## 如何逐题学习

先学习共同基础 Q01–Q06，再选技术方向推进。每题先看“学习目标”：它会说明本题要区分什么，以及需要用哪种推导、状态追踪或设计结果证明自己理解了。

学习时先读完整解释，用题内同一个例子走通公式和状态，再合上答案自己复算。随后改变一个条件：请求变长、batch 增大、带宽下降、开启 Graph 或某个请求取消，检查结论怎样变化。最后打开源码确认决定性写入、返回值和下一消费者。已经理解后再练 3–5 分钟口述，并接受追问；90 秒摘要用于开场，不能代替完整机制解释。

参考答案以回答者口吻直接完成主问和两项追问，给出结论、适用条件、机制、具体算例及后果；生产场景和学习检查点也附完整解答。跨题链接用于扩展，不承担补齐本题关键步骤的责任。没有现场观测的故障题会分别给出可判定的证据分支，不把未获得的数据编成个人经历。

学习状态只记录实际讨论、推导和验证后的进度；“未开始”“学习中”“待复习”“已掌握”继续沿用原记录。掌握一题意味着能独立复现其解答，并能在改变条件后重新推演。

- 引擎方向先走 Q01 → Q07–Q17，再补 Q18–Q20、Q29–Q34，用 Q47–Q49 做综合推演；Q52–Q55、Q63 专练执行与并发边界。
- 算法与算子方向先走 Q02、Q04、Q21–Q28，再走 Q29–Q34；Q13–Q17 用来连接 kernel 与实际执行器。再用 Q56–Q62、Q68 补模型计算、训练对照、手写 Softmax 与并行反例。
- 平台方向先走 Q03、Q05、Q06、Q35–Q46，再用 Q47、Q48、Q50 做故障推演；Q07–Q12 帮助解释引擎内部背压；Q51、Q64–Q67 专练硬件选型、缓存收益、性能模型、仿真和路由决策。

## 68 题总目录与学习状态

★ 表示第一轮优先掌握。题号是稳定引用，不随专题分类重新编号。

### 推理引擎研发与性能优化

| 题号 | 问题 | 核心考点 | 要证明的能力 | 优先级 | 学习状态 |
| --- | --- | --- | --- | --- | --- |
| Q01 | [一条流式请求如何完成生成和资源回收](engine-development.md#q01) | 请求生命周期、进程交接与资源所有权 | 追踪首 token、结束和取消后的下一消费者 | ★ | 未开始 |
| Q07 | [Scheduler 请求与批次的连续两轮状态](engine-development.md#q07) | 请求、获准列表和批次的跨轮状态 | 逐轮写出谁执行、谁保留、谁被回收 | ★ | 未开始 |
| Q08 | [显存空闲时 Prefill 为何仍被拒绝](engine-development.md#q08) | 接纳预算、页容量与工作集限制 | 由拒绝分支反推瓶颈并验证容量假设 | ★ | 未开始 |
| Q09 | [Chunked Prefill Mixed 与 Decode 公平性](engine-development.md#q09) | 分块、执行优先级与 Mixed 批次归属 | 用连续轮次反驳公平性误判并推演重复合并 | ★ | 未开始 |
| Q10 | [Eviction 与 retraction 的回收和恢复](engine-development.md#q10) | 缓存 eviction、活跃请求 retraction 与恢复 | 区别回收对象并追踪重算或恢复的代价 | ★ | 未开始 |
| Q11 | [KV 分配映射和有效内容的区别](engine-development.md#q11) | 逻辑 token、物理槽位与有效 KV | 追踪分配、映射、写入和提交的顺序 | 深入 | 未开始 |
| Q12 | [共享前缀的引用锁与资源所有权](engine-development.md#q12) | Radix 最长前缀、引用锁与共享回收 | 用共享前缀实例证明何时可安全释放 | ★ | 未开始 |
| Q13 | [Worker 和 ModelRunner 的实际执行分派](engine-development.md#q13) | Scheduler／Worker／Runner 和运行分派 | 从配置定位真实执行者及返回对象 | ★ | 未开始 |
| Q14 | [Attention metadata 的预规划与重规划](engine-development.md#q14) | Attention metadata 的内容、时机与复用 | 解释形状变化后谁重规划、谁消费 | ★ | 未开始 |
| Q15 | [Decode CUDA Graph 的静态地址与动态输入](engine-development.md#q15) | Graph 静态地址、动态内容与桶形状 | 追踪输入更新并识别捕获与回放边界 | ★ | 未开始 |
| Q16 | [Prefill Graph 和两种 prefill 切分](engine-development.md#q16) | Prefill Graph、token 分块与层切分 | 按具体执行路径区分复用条件和收益 | 深入 | 未开始 |
| Q17 | [Overlap 的跨轮依赖和读写保护](engine-development.md#q17) | Overlap 的跨轮依赖和读写生命周期 | 画出 CPU／设备时间线并定位竞态 | ★ | 未开始 |
| Q18 | [Grammar 约束和采样状态如何推进](engine-development.md#q18) | Grammar 合法集合与采样状态推进 | 追踪约束、采样和接受后的状态一致性 | 深入 | 未开始 |
| Q19 | [新模型的接入与权重加载契约](engine-development.md#q19) | 模型结构、权重映射与执行契约 | 制定逐层到端到端的接入验证路径 | 深入 | 未开始 |
| Q20 | [投机与多种并行策略的兼容边界](engine-development.md#q20) | 投机算法与 DP／PP／CP 的组合边界 | 由 guard 和状态契约建立兼容矩阵 | ★ | 未开始 |
| Q47 | [TTFT 翻倍而 ITL 稳定的定位](engine-development.md#q47) | TTFT 的队列、Prefill、采样与传输拆解 | 依据指标变化提出可证伪假设并缩小范围 | ★ | 未开始 |
| Q48 | [长输入造成 Decode 卡顿的定位](engine-development.md#q48) | 长输入对 Decode 的排队及执行干扰 | 用批次时间线选择 chunk／Mixed／PD | ★ | 未开始 |
| Q49 | [批处理 Graph Overlap 投机下的偶发错 token](engine-development.md#q49) | Graph／overlap／投机组合下的状态一致性 | 从首个错 token 追到 stale 数据或提交边界 | ★ | 未开始 |
| Q52 | [CUDA Graph 复用地址契约与进程重启](engine-development.md#q52) | Graph 实例、地址契约与进程生命周期 | 区别可复用编译产物、内存和执行实例 | ★ | 未开始 |
| Q53 | [Graph 命中率很高却变慢的成本分析](engine-development.md#q53) | 命中率、padding 与 capture 摊销 | 按同一工作负载计算 Graph 的净收益 | ★ | 未开始 |
| Q54 | [CPU 饱和时怎样划分进程线程和设备流](engine-development.md#q54) | 进程、线程、设备流与对象归属 | 还原部署拓扑并诊断 CPU 提交瓶颈 | ★ | 未开始 |
| Q55 | [原生 MTP 与 Eager Graph 的分段执行](engine-development.md#q55) | 原生 MTP 目标与 draft／verify／commit | 解释各阶段 eager／Graph 分段的合理性 | ★ | 未开始 |
| Q63 | [多 Stream 为何没有重叠又偶发读旧 Buffer](engine-development.md#q63) | Stream 依赖、event 记录和 buffer 生命周期 | 区分执行顺序、内存复用与真实重叠 | ★ | 未开始 |

### 模型推理算法与算子优化

| 题号 | 问题 | 核心考点 | 要证明的能力 | 优先级 | 学习状态 |
| --- | --- | --- | --- | --- | --- |
| Q02 | [Prefill 与 Decode 的瓶颈为何不同](algorithms-kernels.md#q02) | Prefill／Decode 的算术强度与临界点 | 用 FLOPs／bytes 和 batch 解释瓶颈迁移 | ★ | 未开始 |
| Q04 | [KV 容量和并行分片如何估算](algorithms-kernels.md#q04) | KV 有效字节、并行分片与物理预留 | 手算每 rank 容量并解释估算和实测差额 | ★ | 未开始 |
| Q21 | [注意力架构如何改变计算与缓存](algorithms-kernels.md#q21) | MHA／GQA／MLA／SWA 的表示和计算 | 用相同维度区分存储收益与计算收益 | ★ | 未开始 |
| Q22 | [FlashAttention 与 online softmax 的两块推导](algorithms-kernels.md#q22) | 分块 attention 与 online softmax 递推 | 手算两块的 max／sum／输出并证明稳定性 | ★ | 未开始 |
| Q23 | [Decode attention 的 split KV 取舍](algorithms-kernels.md#q23) | Split KV 的并行收益和合并成本 | 推导局部统计合并并选择合理 split | ★ | 未开始 |
| Q24 | [算子 Roofline 和延迟下界](algorithms-kernels.md#q24) | Roofline、算术强度和性能下界 | 由形状和字节估算瓶颈并验证测量 | ★ | 未开始 |
| Q25 | [Kernel 融合的依赖与资源成本](algorithms-kernels.md#q25) | 融合的访存收益和寄存器／同步代价 | 用依赖及资源占用解释融合后的退化 | 深入 | 未开始 |
| Q26 | [量化的收益成本和质量验证](algorithms-kernels.md#q26) | 量化表示、scale 和精度／性能权衡 | 区分存储量化与实际低精度执行并验收 | ★ | 未开始 |
| Q27 | [Tensor Parallel 的张量切分与通信](algorithms-kernels.md#q27) | 行列并行、张量布局和 collective | 手画切分矩阵并指出通信发生的位置 | ★ | 未开始 |
| Q28 | [MoE 从路由到 combine 的整层成本](algorithms-kernels.md#q28) | MoE 路由、派发、专家计算与合并 | 按整层关键路径诊断热点和通信尾部 | ★ | 未开始 |
| Q29 | [投机无损的 greedy 与 sampling 含义](algorithms-kernels.md#q29) | 投机接受、拒绝残差与分布一致性 | 用小分布推导无损条件及错误接受的后果 | ★ | 未开始 |
| Q30 | [EAGLE 树接受后的 KV 和状态压紧](algorithms-kernels.md#q30) | EAGLE 树路径到线性 KV 的提交 | 追踪接受索引、KV 搬运和下一轮位置 | ★ | 未开始 |
| Q31 | [dFlash 和 dSpark 的提案与验证计数](algorithms-kernels.md#q31) | dFlash／dSpark 的提案宽度和提交计数 | 分别列出验证布局、接受量和真实输出量 | ★ | 未开始 |
| Q32 | [Target hidden 怎样建立 Draft KV](algorithms-kernels.md#q32) | Target hidden 与 Draft KV 的数据契约 | 追踪层／位置／维度对齐及下一轮消费者 | 深入 | 未开始 |
| Q33 | [接受长度与投机加速收益的区别](algorithms-kernels.md#q33) | 接受长度、投机额外成本与收益分母 | 建立整轮时间模型并解释高接受低收益 | ★ | 未开始 |
| Q34 | [投机解码的正确性验证矩阵](algorithms-kernels.md#q34) | Greedy／sampling、停止条件与组合正确性 | 设计能定位首个分歧的验证矩阵 | ★ | 未开始 |
| Q56 | [MoE 总参数激活参数与常驻显存](algorithms-kernels.md#q56) | MoE 总参数、激活计算和物理常驻 | 复算主干参数／FLOPs 并纠正显存计划 | ★ | 未开始 |
| Q57 | [MLA 压缩缓存为何不等于等比例加速](algorithms-kernels.md#q57) | MLA latent、位置分量与投影吸收 | 由矩阵恒等式解释压缩缓存和剩余成本 | ★ | 未开始 |
| Q58 | [训练显存优化为何不能原样搬到推理](algorithms-kernels.md#q58) | 训练状态、activation 重算与推理 KV | 建立两类显存账本并判断技术能否迁移 | ★ | 未开始 |
| Q59 | [训练 Prefill Decode 的并行策略如何切换](algorithms-kernels.md#q59) | 训练／Prefill／Decode 的目标和通信 | 基于模型、拓扑与 SLO 枚举并比较八卡布局 | ★ | 未开始 |
| Q60 | [Softmax 数值稳定性与三遍两遍扫描](algorithms-kernels.md#q60) | 稳定 softmax、扫描次数与片上驻留 | 推导 online normalizer 并明确访存口径 | ★ | 未开始 |
| Q61 | [Warp 与 Block Softmax 如何覆盖不规则形状](algorithms-kernels.md#q61) | Warp／block 归约、尾部和同步契约 | 手写不规则形状基线并分析数值与边界 | ★ | 未开始 |
| Q62 | [CUDA 优化建议何时收益何时适得其反](algorithms-kernels.md#q62) | Bank、对齐、分歧与资源占用 | 用反例和测量检验 shared／向量化优化 | ★ | 未开始 |
| Q68 | [DP 乘 TP 等于 EP 的适用前提与通信成本](algorithms-kernels.md#q68) | Attention 与 MoE 的独立 rank 映射 | 构造 DP×TP≠EP 的反例并分析通信成本 | ★ | 未开始 |

### 推理平台与大规模部署

| 题号 | 问题 | 核心考点 | 要证明的能力 | 优先级 | 学习状态 |
| --- | --- | --- | --- | --- | --- |
| Q03 | [TTFT TPOT ITL 吞吐和 goodput 的统计口径](platform-deployment.md#q03) | 延迟分位数、流式统计和达标吞吐 | 从原始时间戳复算指标并识别口径偏差 | ★ | 未开始 |
| Q05 | [如何公平比较配置与推理引擎](platform-deployment.md#q05) | 公平对照、到达过程与混杂因素 | 设计可复现且受同一 SLO 约束的比较 | ★ | 未开始 |
| Q06 | [SGLang Dynamo Mooncake NIXL 的职责与组合](platform-deployment.md#q06) | 执行器、控制面、缓存与传输的职责 | 画出组件交界并定位责任归属 | 深入 | 未开始 |
| Q35 | [PD 分离的成本和适用负载](platform-deployment.md#q35) | PD 的阶段隔离、传输和容量经济性 | 用时延与排队账本选择分离或合部 | ★ | 未开始 |
| Q36 | [PD 交接的状态与完成门槛](platform-deployment.md#q36) | PD 预分配、传输、恢复和接纳门槛 | 用双端时间线定位尚未满足的完成条件 | ★ | 未开始 |
| Q37 | [Mooncake Transfer Engine 和 Store 的区别](platform-deployment.md#q37) | Transfer Engine 与分布式 Store 的契约 | 区分搬运完成、对象可见和生命周期 | ★ | 未开始 |
| Q38 | [HiCache 搬运事件引用锁和 ACK](platform-deployment.md#q38) | HiCache 多级表示、锁、ACK 和 DMA 事件 | 解释何时可消费或回收并诊断缓存竞态 | ★ | 未开始 |
| Q39 | [KV aware routing 的缓存与负载权衡](platform-deployment.md#q39) | 缓存亲和、负载和状态新鲜度 | 估算缓存收益与排队损失并避免热点 | ★ | 未开始 |
| Q40 | [Dynamo Planner 的扩缩容信号与反馈](platform-deployment.md#q40) | Planner 的容量反馈、延迟与稳定性 | 设计扩缩容信号、排空和防振荡机制 | 深入 | 未开始 |
| Q41 | [并行策略与 P D 配比如何选择](platform-deployment.md#q41) | 并行拓扑与 P／D 容量配比 | 在固定卡数和 SLO 下定位瓶颈并配置资源 | ★ | 未开始 |
| Q42 | [TCP RDMA 与真实 PD 的验证层级](platform-deployment.md#q42) | 网络可达、组件带宽与真实 KV 交接 | 分层证明传输正确性和端到端可用性 | ★ | 未开始 |
| Q43 | [多机 hang 的 rank 和通信排查](platform-deployment.md#q43) | Rank 一致性、collective 顺序与分布式等待 | 找到首个分歧而非只定位最后卡住的位置 | ★ | 未开始 |
| Q44 | [传输中取消请求的安全回收](platform-deployment.md#q44) | 取消、迟到写入、attempt 与安全回收 | 建立资源所有权和幂等／隔离状态机 | ★ | 未开始 |
| Q45 | [模型与引擎的滚动升级和回滚](platform-deployment.md#q45) | 版本兼容、预热、灰度与在途请求 | 给出可操作的发布验收、排空和回滚边界 | ★ | 未开始 |
| Q46 | [多租户长请求的预算和背压](platform-deployment.md#q46) | 多租户预算、背压和尾延迟隔离 | 把 SLO 落为接纳、限额和恢复策略 | ★ | 未开始 |
| Q50 | [Prefill 已完成但 bootstrap 超时的定位](platform-deployment.md#q50) | PD bootstrap、容量、metadata 和恢复等待 | 用同一 request／room 的双端证据定位门槛 | ★ | 未开始 |
| Q51 | [硬件迁移如何从规格走到容量与延迟判断](platform-deployment.md#q51) | 硬件 SKU、软件后端与服务容量 | 把规格转成每 rank 账本和迁移验收 | ★ | 未开始 |
| Q64 | [多级缓存命中率提升为何端到端变慢](platform-deployment.md#q64) | 多级缓存的关键路径和机会成本 | 把命中转换为净收益并定位带宽竞争 | ★ | 未开始 |
| Q65 | [低 MAPE 的性能模型为何仍选错部署](platform-deployment.md#q65) | 实测／SOL／插值与模型决策误差 | 设计留出验证并计算 SLO 误判和选型 regret | ★ | 未开始 |
| Q66 | [PD 离散事件仿真怎样避免过度乐观](platform-deployment.md#q66) | 虚拟时钟、PD 状态机和资源守恒 | 手推事件并验证仿真没有制造容量或并发 | ★ | 未开始 |
| Q67 | [路由命中率更高却 TTFT 更差如何重设计评分](platform-deployment.md#q67) | 缓存收益、负载评分和 Dynamo 状态来源 | 用同单位成本重设计路由并校准陈旧反馈 | ★ | 未开始 |

## 补充社区资料如何转成可检验的问题

同一个知识点可以先回答机制题，再回答它在生产中失效的场景。新增题不按术语另列一套重复八股，而是补下面的能力缺口；原题保留作先修。

| 从问题中提取的知识点 | 需要掌握的生产判断 | 原题基础 → 新场景 |
| --- | --- | --- |
| 硬件／软件栈与容量 | SKU、可用内存、带宽和后端能力能否支撑同一 SLO；峰值如何误导选型 | Q04、Q24、Q41 → [Q51](platform-deployment.md#q51) |
| Graph 的复用条件与成本 | 地址／生命周期、capture 摊销、padding、重启预热，解释命中但变慢 | Q15、Q16 → [Q52](engine-development.md#q52)、[Q53](engine-development.md#q53) |
| 进程、线程与设备流 | 分清对象与 OS 进程，追到 IPC、设备依赖和 buffer 的下一消费者 | Q01、Q13、Q17 → [Q54](engine-development.md#q54)、[Q63](engine-development.md#q63) |
| 原生 MTP 与投机执行 | 区分训练目标、draft／verify／commit，判断各阶段 eager／Graph 是否合理 | Q29–Q34 → [Q55](engine-development.md#q55) |
| DeepSeek V3 MoE／MLA | 手算总参数、激活计算和 KV；识别常驻权重与带宽没有同比下降的原因 | Q04、Q21、Q28 → [Q56](algorithms-kernels.md#q56)、[Q57](algorithms-kernels.md#q57) |
| 训练与服务的显存／并行策略 | 区分参数、优化器、激活和 KV，结合阶段目标设计八卡布局 | Q02、Q27、Q41 → [Q58](algorithms-kernels.md#q58)、[Q59](algorithms-kernels.md#q59)、[Q68](algorithms-kernels.md#q68) |
| Softmax 与 GPU 优化 | 推导扫描次数、稳定归约与尾部；用资源、对齐和测量检验优化建议 | Q22、Q24、Q25 → [Q60](algorithms-kernels.md#q60)–[Q62](algorithms-kernels.md#q62) |
| Radix／HiCache 多级缓存 | 先追 token ID 最长前缀与锁，再算关键路径、带宽竞争和机会成本 | Q11、Q12、Q38 → [Q64](platform-deployment.md#q64) |
| AIConfigurator 性能建模 | 区分实测／插值／SOL，检查留出集、尾部误差与配置选择损失 | Q05、Q24、Q41 → [Q65](platform-deployment.md#q65) |
| PD、Mixed Chunk 与虚拟时钟 | 容量、排队、传输和资源占用守恒；何时 PD 比合部更差 | Q09、Q35、Q36、Q41 → [Q66](platform-deployment.md#q66) |
| Cache-aware routing 与 Dynamo | 把缓存收益和排队转成同单位成本，处理陈旧状态、热点与取消 | Q06、Q39、Q40 → [Q67](platform-deployment.md#q67) |

## 怎样判断回答足够具体

| 层次 | 学习中应展示的能力 | 以 Q09 为例 |
| --- | --- | --- |
| 机制理解 | 从基线问题讲起，说明额外机制为何必要 | 长 Prefill 会推迟 B 的下一 token；分块提供重新选批边界，但调度顺序决定是否执行 B |
| 数值与状态推演 | 给定具体输入，逐步算出状态和输出，能改变条件重算 | 非 Mixed、A 剩 12288 token、每轮实际可处理 4096 且无其他预算阻挡，画出三轮 Prefill 与 B 的等待；Mixed 后追踪 `[A,B]` 与空运行批 |
| 源码理解 | 定位实际分支，指出关键写入、返回值和下一消费者 | 先续块再 waiting；有 Prefill 批则优先；Mixed 换空后下一轮由 last_batch 接回 B |
| 取舍判断 | 说明收益、代价，以及结论在哪些条件下改变 | 较小 chunk 缩短单次暴露，也增加轮数；Mixed 推进 B，但 B 的单步仍承受混批时长 |
| 实验与排障 | 提出竞争假设，列出能支持或推翻它们的观测 | 区分 B 未入执行批、混批本身慢、输出交付慢，分别检查 batch composition、设备关键路径、客户端时间戳 |

“开启分块提升公平性”没有交代重新选批后谁被选中；“Mixed 后清空防止重复”没有指出下一轮的合并路径。能把这些省略的因果补出来，才经得起继续追问。每题的“常见误区”和带解答的“学习检查点”用于核对同类因果是否真正讲清。

题干的两项追问在参考答案中逐项直接回答；检查点在原问题后附“解答”。公式需要定义变量、轴和单位，运行机制需要明确对象、写入和下一消费者，故障诊断需要写清不同观测对应的结论与处理动作。参考回答不会把这些步骤留给读者补全。

[九项实践任务](practice.md)连接理解、源码与可复现证据。复述个人经历时，写明实际做过的层级、版本、硬件、负载、对照和结果；书面推演、CPU reference、GPU 实验和线上经历分别记录。

## 源码与外部资料的使用范围

题库最初的 Q01–Q50 编写于 2026-10-15，参考 `f01a0c7f97` 与当时工作树；Q51–Q68 最初编写于 2026-10-20。本轮问答核查基于 `HEAD=441c855fef`；推理源码与 `bb669346d5` 相同。本轮逐项核对主问、追问和检查点，补齐直接回答，并核对结论所依赖的关键分支、状态写入与下一消费者。源码链接配合符号名使用；切换版本后，优先重新确认这些决定结论的边界。示例中的数字分别标为手算假设或外部资料数据，实践任务的 GPU、多机与真实 PD 结果在实际完成后记录。

普通运行链以 CUDA 通用 `TpModelWorker` 为主要情景。MLX 会改派 worker；ROCm、NPU、XPU、DP/PP/CP、Mixed、EAGLE/dFlash/dSpark 的差异在相关题中定位，不能把一个分支的性质扩展成所有组合的保证。

社区经验记录提供问题线索，技术答案依当前源码、原始论文和官方资料。Dynamo 与 Mooncake 不在本次 SGLang 源码树中的部分，以外部设计流程和集成交界说明，不能当作已经在本地追过的调用链。详见[资料与源码基线](references.md)。
