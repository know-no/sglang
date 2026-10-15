# 面试题资料与源码基线

[题库总索引](README.md) · [引擎研发](engine-development.md) · [算法与算子](algorithms-kernels.md) · [平台与部署](platform-deployment.md)

## 源码基线

本题库的 SGLang 实现说明对应 2026-10-15 的本地工作树：`HEAD=f01a0c7f97`，包含已有未提交修改。源码链接采用文件、当前行号和正文中的类／函数名共同定位。仅用 HEAD checkout 不能还原这些工作树修改；升级或切换 checkout 后，应重新核对引用处及其下一消费者。

主要研究范围是请求与批次、KV 分配和引用、worker 与执行器分派、attention metadata、CUDA Graph、overlap、EAGLE/dFlash/dSpark、PD、HiCache、量化与并行，以及 benchmark 的数据消费路径。外部 Dynamo、Mooncake 的实现不由这份 SGLang 工作树证明。

源码中的可达路径、配置允许的组合、GPU 实验正确性与线上性能是不同证据。本题库的源码解释属于前两类；[实践任务](practice.md)定义后续实验的条件与验收，不包含本轮新增的 GPU、多机或真实 PD 运行结果。

## 公开面经与技术文章

社区资料提供出题线索。候选人自述不能证明公司统一面试标准或某题的考察频次，技术文章也不能代替当前源码。

<a id="n1"></a>
### 部署优化与高性能计算社招面经

[模型部署／推理优化／高性能计算方向社招面经总结](https://www.nowcoder.com/discuss/599177965083054080)，牛客，作者 Litt1e0range，2024-03-18。原始候选人自述；正文报告了 CUDA Graph、stream、kernel launch，以及 softmax、reduction、transpose 的实现与对齐。关联 Q15、Q17、Q22、Q24、Q25。

<a id="n2"></a>
### AIGC 推理框架研发面经

[26 届春招 寒武纪 AIGC 推理框架研发工程师](https://www.nowcoder.com/discuss/874315356406304768)，牛客。原始候选人记录，包含请求流转、KV 容量、prefix cache、TP、MTP、融合和 Graph。帖内 2025／2026 日期标记不一致，不能据此静默确定面试年份。关联 Q01、Q04、Q12、Q15、Q27、Q29。

<a id="n3"></a>
### AI Infra 通信与在线服务面经

[0914 字节跳动一面 AI Infra](https://www.nowcoder.com/feed/main/detail/a1ce24c561874141abe6ba18a466f982)，牛客，镜子Miracoln。页面显示 09-14，未展示年份。原始候选人自述，涉及多机硬件、MPI reduce 瓶颈、灰度回滚、容灾、profiling 经历和带过期的 LRU。关联 Q24、Q41、Q43、Q45。

<a id="n4"></a>
### AI Infra 实习面经转载

[蚂蚁 AI Infra 实习三轮面试实录](https://www.10100.com/article/149156215)，二手转载，页面标 2026-07-10，文内原作者草原小可汗标 2026-04-18。可读正文涉及算子优化价值判断、计时、量化、融合与随机崩溃；没有将转载者的总结当作原公司的标准答案。关联 Q24–Q26、Q34、Q49。

<a id="t1"></a>
### Cache Req 与 Scheduler 源码笔记

[知乎原文](https://zhuanlan.zhihu.com/p/17186885141)在检索时直读受限；可读的是[极术社区转载](https://community.aijishu.com/a/1060000000509950)，转载日期 2025-05-15。属于技术文章，讨论请求槽位、KV 槽位、prefix 匹配和批次。用于理解问题背景，具体实现已按当前工作树重新核对。关联 Q07–Q12。

<a id="t2"></a>
### 公开源码面试题系列

[vLLM／SGLang 源码面试题系列](https://blog.cptz.space/posts/vllm-sglang-interview-series/)，Cptz，2026-07-20。原作者自编技术题库，声明固定版本与实验；它属于备考材料，不能计入原始候选人面经样本。本题库保留自己的 50 题编号与本地源码基线。

<a id="t3"></a>
### Mooncake 与 SGLang 结合点研究

[Mooncake 与 SGLang 的结合点全景](https://miraclefarms.github.io/notes/2026/07/17/sglang-mooncake-integration-panorama/)，Ethan，2026-07-17。原作者研究博客，注明当时的两个仓库版本，适合作为 HiCache 与 PD 职责问题的线索；当前行为以本地源码和下面的官方资料为准。关联 Q06、Q36–Q38、Q50。

## 原始论文与官方资料

下面链接在 2026-10-15 的研究中使用。`latest` 与 `dev` 文档会变化，讨论具体版本能力或准备运行时应再次核对，尤其是集成后端、Planner 行为和配置兼容性。

<a id="flashattention"></a>
### FlashAttention

[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)解释 IO-aware tiling 与精确 attention；[FlashAttention 2](https://arxiv.org/abs/2307.08691)进一步讨论并行划分。主要关联 Q22–Q24。论文的加速数字只属于论文中的实验，不是本题库的测量结果。

<a id="pagedattention"></a>
### PagedAttention

[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)讨论动态 KV 的分页、共享和内存效率。主要关联 Q04、Q11、Q12；SGLang 的具体 allocator、映射和 Radix 引用以本地实现为准。

<a id="speculative-decoding"></a>
### 标准投机采样

[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192v2)的 Algorithm 1 给出标准接受与残差采样，后续分析有明确的独立性和成本假设。主要关联 Q29、Q33、Q34。EAGLE、dFlash、dSpark 的提案结构和工程提交逻辑不能直接从标准伪代码推定。

<a id="cuda-graphs"></a>
### CUDA Graph 与 stream 依赖

[CUDA Graph 官方编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)说明图定义、实例化、执行和跨 stream 捕获依赖。主要关联 Q15–Q17、Q49；SGLang 的选桶、padding、metadata 与读写保护需继续读 runner 和 Scheduler。

<a id="dynamo-overview"></a>
### Dynamo 组件与 SGLang 集成

[Dynamo 官方概览](https://docs.nvidia.com/dynamo/latest/)列出 frontend、router、planner、cache manager 与 worker；[SGLang PD 集成](https://docs.dynamo.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/disaggregation)说明集成路径。主要关联 Q06、Q35、Q36、Q41。

<a id="dynamo-routing"></a>
### Dynamo 路由

[KV-aware routing](https://docs.nvidia.com/dynamo/cli/kv-aware-routing/overview)和 [Router 概览 dev](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/overview)讨论缓存状态、负载与 worker 选择。关联 Q39。SGLang Gateway 本地的文本前缀启发式与 Dynamo 的 KV event 索引是两个需要分别说明的实现。

<a id="dynamo-planner"></a>
### Dynamo Planner

[Planner 模式选择 dev](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/choose-a-planner-mode)、[Planner 设计](https://docs.nvidia.com/dynamo/latest/knowledge-base/modular-components/planner/planner-design)与 [Planner 概览 dev](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview)用于 Q40、Q41、Q45。不同版本的信号、性能模型和缩容流程可能不同；题中的排空／迁移方案应与实际实现分开评价。

<a id="mooncake-integration"></a>
### Mooncake 与 SGLang

[Mooncake 官方 SGLang 集成](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/)区分 PD 的 Transfer Engine 路径和 HiCache L3 Store 路径；[Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html)给出入口。关联 Q06、Q35–Q38。

<a id="mooncake-te"></a>
### Mooncake Transfer Engine

[Transfer Engine 设计](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)解释 Segment、注册的 buffer、BatchTransfer、完成查询和拓扑。关联 Q37、Q42、Q44。具体传输能力还受编译选项、协议、设备和运行环境限制。

## 与已有学习材料的关系

模块定位仍从 [SGLang 全景图](../index.html)进入。投机机制的已有主文和交互课程保留在[投机专题](../speculative/README.md)；本题库按面试问题组织答案，并链接相关专题，不重建已退役的全课程。

后续逐题学习时，先讨论和纠正当前题，再更新该题的主答案与学习状态。实验记录写入具体条件和证据，不用理论上界替代实测，也不把组件通过扩展成所有配置的端到端保证。
