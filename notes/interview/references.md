# 学习题资料与源码基线

[题库总索引](README.md) · [引擎研发](engine-development.md) · [算法与算子](algorithms-kernels.md) · [平台与部署](platform-deployment.md)

## 源码基线

题库的历史编写基线分为两批：Q01–Q50 最初对应 2026-10-15 的 `f01a0c7f97` 与当时工作树修改；Q51–Q68 最初对应 2026-10-20 的 `bb669346d5`。当前版本按 `bb669346d5` 重写全部 68 题的解释、学习目标和生产追问，核对决定答案的运行分派、关键状态写入、返回值和下一消费者。

源码链接采用文件、行号及正文的类／函数名共同定位。更新版本后先核对这些决定结论的边界。历史源码基线不能只凭 commit 还原当时额外的工作树修改；当前解释使用的关键路径以各题列出的实现为准。

主要研究范围是请求与批次、KV 分配和引用、worker 与执行器分派、attention metadata、CUDA Graph、overlap、EAGLE/dFlash/dSpark、PD、HiCache、量化与并行，以及 benchmark 的数据消费路径。外部 Dynamo、Mooncake 的实现不由这份 SGLang 工作树证明。

源码中的可达路径、配置允许的组合、GPU 实验正确性与线上性能是不同证据。本题库的源码解释属于前两类；[实践任务](practice.md)定义后续实验的条件与验收，不包含本轮新增的 GPU、多机或真实 PD 运行结果。

## 社区经验与技术文章

社区资料提供问题线索；经验记录与技术文章中的结论仍需由当前源码、原始论文和官方资料核对。

<a id="n1"></a>
### 部署优化与高性能计算经验记录

[模型部署／推理优化／高性能计算经验整理](https://www.nowcoder.com/discuss/599177965083054080)，牛客，作者 Litt1e0range，2024-03-18。作者原始记录；正文报告了 CUDA Graph、stream、kernel launch，以及 softmax、reduction、transpose 的实现与对齐。关联 Q15、Q17、Q22、Q24、Q25。

<a id="n2"></a>
### AIGC 推理框架研发经验记录

[AIGC 推理框架技术记录](https://www.nowcoder.com/discuss/874315356406304768)，牛客。作者原始记录，包含请求流转、KV 容量、prefix cache、TP、MTP、融合和 Graph。帖内 2025／2026 日期标记不一致，不能据此静默确定记录年份。关联 Q01、Q04、Q12、Q15、Q27、Q29。

<a id="n3"></a>
### AI Infra 通信与在线服务经验记录

[AI Infra 通信与在线服务讨论记录](https://www.nowcoder.com/feed/main/detail/a1ce24c561874141abe6ba18a466f982)，牛客，镜子Miracoln。页面显示 09-14，未展示年份。作者原始记录，涉及多机硬件、MPI reduce 瓶颈、灰度回滚、容灾、profiling 经历和带过期的 LRU。关联 Q24、Q41、Q43、Q45。

<a id="n4"></a>
### AI Infra 技术经验转载

[AI Infra 技术交流记录（转载）](https://www.10100.com/article/149156215)，二手转载，页面标 2026-07-10，文内原作者草原小可汗标 2026-04-18。可读正文涉及算子优化价值判断、计时、量化、融合与随机崩溃；没有将转载者的总结当作原公司的标准答案。关联 Q24–Q26、Q34、Q49。

<a id="t1"></a>
### Cache Req 与 Scheduler 源码笔记

[知乎原文](https://zhuanlan.zhihu.com/p/17186885141)在检索时直读受限；可读的是[极术社区转载](https://community.aijishu.com/a/1060000000509950)，转载日期 2025-05-15。属于技术文章，讨论请求槽位、KV 槽位、prefix 匹配和批次。用于理解问题背景，具体实现已按当前工作树重新核对。关联 Q07–Q12。

<a id="t2"></a>
### 公开源码学习题系列

[vLLM／SGLang 源码学习题系列](https://blog.cptz.space/posts/vllm-sglang-interview-series/)，Cptz，2026-07-20。原作者自编技术题库，声明固定版本与实验；它作为学习题库线索使用，技术结论仍需按当前源码和一手资料核对。本题库保留自己的 Q01–Q50 编号与当时的本地源码基线；2026-10-20 再扩充 Q51–Q68。

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

## 2026-10-20 新增题的资料边界

用户补充的网上问题用于提取考点，没有把未提供出处的摘录作为已核验经验记录样本。新增题是据此重构的场景题，不代表某公司的固定题单。技术答案继续优先采用当前源码、原始论文和下列官方资料；产品路线图、设计文档、实际实现和实测数据分别标记。

<a id="hardware-migration"></a>
### 硬件 SKU、路线图与后端能力

[NVIDIA Hopper 设备表](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper.html)列有 H20 SXM5 96 GB／141 GB。容量需按实际 SKU 和服务可用设备内存核对；不能把一款 H20 的规格套到整个系列。[华为 2025 年官方演讲](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech)提供 950DT 的容量／带宽及 2026 Q4 路线图，不能据此确认目标整机交付或实测性能。[vLLM Ascend 集成文档](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/user_guide/feature_guide/kv_cache_pool_ascend_store.html)中的 A5／950PR／950DT 标签用于命名核对，不是本仓库已经验证的支持声明。[HCCL 950 支持表](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/comm_ops_support_list/comm_ops_support_950.md)按通信引擎、网络与调用模式列能力。关联 Q51。

<a id="deepseek-v3"></a>
### DeepSeek V3 模型与推理表示

[DeepSeek V3 技术报告](https://arxiv.org/abs/2412.19437)与[官方模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)用于核对 MoE、MLA 和 MTP 的维度、逻辑参数和目标。模型权重常驻、专家副本、量化 scale、运行时缓存表示与实际 kernel 路径继续以具体部署为准。关联 Q55–Q57；671B／37B 不能直接代替显存或逐步推理耗时。

<a id="training-memory"></a>
### 训练状态分片与重计算

[ZeRO 原论文](https://arxiv.org/abs/1910.02054)、[PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)和[activation checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)用于训练显存账本与通信／重计算取舍。它们是外部训练机制的证据，不是 SGLang 含有训练路径的证明。关联 Q58、Q59。

<a id="cuda-runtime"></a>
### CUDA 执行、资源与归约

[CUDA Graph 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)、[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)和 [Tensor.record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)分别支撑图与地址契约、访存／归约优化、stream 依赖与内存生命周期。kernel 少读一遍、API 异步返回或使用多个 stream，都需要再用实际设备工作和关键路径验证收益。关联 Q52–Q54、Q60–Q63。

<a id="aiconfigurator"></a>
### AIConfigurator 的数据来源与版本

新增题核对官方仓库快照 `f254959eb89e2f206b8f9a77051644d7c1cbdb89`：[混合建模设计](https://github.com/ai-dynamo/aiconfigurator/blob/f254959eb89e2f206b8f9a77051644d7c1cbdb89/docs/design/hybrid_modeling_design.md)、[FPM 建模计划](https://github.com/ai-dynamo/aiconfigurator/blob/f254959eb89e2f206b8f9a77051644d7c1cbdb89/docs/fpm/aic-fpm-modeling-plan.md)。设计描述、数据 provenance、实际 Python／Rust 执行分工及计划中的指标不能相互替代；“MAPE 小于 10%”在本题中是待检验假设，未作为官方通用精度保证。当前模型能力与线上 TTFT／ITL、排队仿真也要区分。关联 Q65、Q66，具体实现链见题内固定版本链接。

## 与已有学习材料的关系

模块定位仍从 [SGLang 全景图](../index.html)进入。投机机制的已有主文和交互课程保留在[投机专题](../speculative/README.md)；本题库按学习问题组织答案，并链接相关专题，不重建已退役的全课程。

后续逐题学习时，先讨论和纠正当前题，再更新该题的主答案与学习状态。实验记录写入具体条件和证据，不用理论上界替代实测，也不把组件通过扩展成所有配置的端到端保证。
