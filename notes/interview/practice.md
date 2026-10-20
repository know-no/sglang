# 推理技术实践任务与学习记录

[题库总索引](README.md) · [资料与源码基线](references.md)

九项任务把参考答案转成自己能够复述和复现的经历。P1–P6 保留原计划，P7–P9 对应新增的手写算子、容量／性能建模和仿真设计。按可用设备逐步完成；源码追踪、CPU reference、单 GPU、多 GPU、CUDA Graph／overlap、真实 PD 传输分别记录证据。题库中的例子是推演输入，实验结果在完成后填写。

## P1 请求与调度状态追踪

关联 [Q01](engine-development.md#q01)、[Q07](engine-development.md#q07)至 [Q12](engine-development.md#q12)、[Q17](engine-development.md#q17)。适合三个方向的共同起点。

固定两个请求：A 输入较长并分块，B 已在 Decode；随后让 A/B 共享一段 prefix，再分别结束和取消。先手工走连续两轮，再在具备运行条件时用最小日志或调试器验证。日志只保存必要的 RID、长度、slot 与状态，避免记录敏感 prompt。

交付一张对象图和一张时间线。对象图区分 `Req`、获准列表、`ScheduleBatch`、执行批、持久运行批、上轮批；时间线标出队列变化、KV 映射、分配与提交边界，以及 CPU 提交、GPU 工作和 CPU 消费的先后。

验收时应能回答：A 剩余 chunk 在哪里；B 哪轮实际执行；共享页是谁拥有；取消后哪个消费者完成清理。对照关闭／开启 overlap，解释 `last_batch`、结果队列和 FutureMap 的差异。检查空流量下 request slots、未完成任务、protected KV 与缓存驻留之间是否符合各自语义。

## P2 性能曲线与容量解释

关联 [Q02](algorithms-kernels.md#q02)、[Q03](platform-deployment.md#q03)、[Q04](algorithms-kernels.md#q04)、[Q05](platform-deployment.md#q05)、[Q47](engine-development.md#q47)、[Q48](engine-development.md#q48)。

选定一个实际能加载的模型和设备，记录权重／tokenizer 版本、精度、完整配置。工作负载至少包含短输入短输出、长输入短输出、短输入长输出，并分别测冷缓存和可复用前缀。batch／并发与请求率分开控制；明确是闭环客户端还是按给定到达过程发请求。

交付吞吐、TTFT、TPOT／ITL 与并发／到达率的曲线，保留成功／失败／取消数和实际输出长度。goodput 自行明确定义，例如每秒同时满足约定 TTFT 与 TPOT 阈值的成功请求数，并将阈值随报告保存。记录真实 stream chunk；多 token 摊分只是统计口径，不能重建每个 token 的实际到达时间。

容量估算先算有效 KV，再列权重、graph、workspace、页舍入和预留。至少解释一次“GPU 看起来不满但请求进不去”或“吞吐变高而尾延迟恶化”。用 queue／admission、runner 命中和关键路径证据区分假设，给出本次适用的负载范围。

当前在线测量入口是 [benchmark.serving](../../python/sglang/benchmark/serving.py)，旧 `sglang.bench_serving` 是兼容入口。运行参数以目标 checkout 的帮助和实际模型为准。

## P3 算子正确性与性能归因

关联 [Q22](algorithms-kernels.md#q22)至 [Q28](algorithms-kernels.md#q28)。适合算法与算子方向深入，引擎方向也应能解释其结果。

从 softmax、RMSNorm 或 decode attention 选一个小范围问题。先写清输入／输出、shape、dtype、mask 与数值规则，再建立可信 reference。优化只改变一项，例如访存、切分或融合，避免同时更换模型、精度和 kernel。

正确性输入覆盖非整齐长度、尾块、极值、不同 dtype；attention 另测 page 索引、GQA、空 split 和长上下文。解释绝对／相对误差及累积误差；出现 NaN 时保存最小输入和首个错误位置。GPU 计时包含 warmup、足够迭代与必要同步，区分实际执行与 Python enqueue；同时报告环境、波动和计时范围。

交付 reference、优化前后结果和一段关键路径分析。用 FLOPs／bytes 下界检查数字是否合理，再测一次完整模型或请求路径。若 kernel 更快但服务没有收益，定位它占比不足、CPU／通信暴露、调度改变或额外变换成本。microbenchmark 通过与完整生成正确是两个验收层次。

## P4 投机解码的提交与收益

关联 [Q29](algorithms-kernels.md#q29)至 [Q34](algorithms-kernels.md#q34)、[Q49](engine-development.md#q49)。

选定实际可用且兼容的 Target／Draft。先确认算法和参数的计数定义，再比较 Target-only 与一个投机分支。逐步增加 batch、prefix cache、Graph／overlap 和多卡条件，不把一个维度通过推定为组合通过。

每轮同时记录 proposal width、实际 verify 布局、正确 draft 数、实际提交长度、输出 token 数、KV 已分配／已提交长度，以及 Draft、Verify、accept／finalize 的时间。对于 dSpark，要分开实际 `commit_lens` 与潜在整块接受指标；对于 EAGLE，要说明树接受路径怎样变成线性 KV。

greedy 定位首个 token 或 logits 分歧，保留数值与 tie-break 条件。sampling 在可解析的小分布 reference 上检查算法，再用多次样本检验指定统计量，说明有限样本检验的能力；相同 seed 的逐字相同不是分布证明。加入 stop／EOS、grammar 和长度边界。

交付一份正确性矩阵和一张收益曲线。至少解释一个接受长度提高但吞吐下降的设置，并用总耗时、显存／batch 或关键路径变化证实原因。所有加速都绑定本次负载和配置。

## P5 PD 与 HiCache 的真实交接

关联 [Q35](platform-deployment.md#q35)至 [Q38](platform-deployment.md#q38)、[Q42](platform-deployment.md#q42)、[Q50](platform-deployment.md#q50)。

先确认本次硬件、网络、容器、传输协议与 buffer 类型。分层检查组件：本地 buffer 生命周期，跨端有效载荷和校验，SGLang 的源／目的 KV 映射，metadata／room 提交，再到 D 的首轮生成。记录 P/D 两侧相同 request／room 的时间线。

交付状态表：bootstrap、D admission／prealloc、metadata 已发送、P 执行、传输、HiCache 恢复、提交、PREBUILT／Decode。HiCache 本地恢复可以与远端传输并行；原始 receiver 的 Success 与经过恢复／metadata gate 的状态分别标记。微基准通过不能替代真实 KV 布局与状态交接验证。

至少比较 cache off、冷缓存和暖缓存，并注入一次可控失败或取消。失败后检查未完成数据不会进入可复用前缀，request／metadata slots 与预留页最终回收。报告 TCP 与 RDMA、组件与端到端、eager 与 Graph 的分别结果；选路、吞吐和协议能力都绑定实际环境。

## P6 发布升级与故障恢复

关联 [Q39](platform-deployment.md#q39)至 [Q46](platform-deployment.md#q46)。适合平台方向深入。

写出一个具体服务的 SLO、资源预算和发布条件。指定版本兼容矩阵：权重、tokenizer／template、adapter、KV dtype／布局、page size、并行拓扑和传输协议。新实例经历加载、预热、计算／传输验证，再逐步接流量；旧实例停止接新请求后如何处理在途请求必须明确。

选择一个故障演练：慢 rank、D 容量下降、L3 变慢、worker 重启或网络不可达。预先定义停止条件、失败范围和恢复目标，控制注入范围。对请求重试使用 attempt 标识和预算；取消、重复通知和迟到传输必须考虑物理 buffer 是否仍可写。

交付灰度与回滚步骤、故障时间线、恢复指标和资源回收结果。区分失败率恢复、已有请求处理和缓存恢复三个时间点；扩容的加载／预热成本和路由反馈延迟也纳入报告。drain、迁移、重算和快速失败各有条件，不能假设所选框架版本自动完成全部步骤。

## P7 手写 Softmax 与反证式优化

关联 [Q60](algorithms-kernels.md#q60)–[Q62](algorithms-kernels.md#q62)，沿用 P3 的计时和正确性规范。

先在纸上推导 safe 三次扫描与 online 两次扫描，写明每一遍读写什么、归约什么，以及中间结果留在寄存器还是落到全局内存。以 `[1000, 1001, -1000]` 解释稳定性，再实现 CPU reference；CPU 结果只能证明公式与测试输入，不能验证 CUDA 同步。

有 GPU 后分别实现一 warp／行和一 block／行版本，覆盖 N=1、31、32、33、1025、8193、非对齐行 stride、有限极值和全 mask；先规定全 mask 的输出语义。加入多行和 FP16／BF16／FP32，核对 FP32 累积、shuffle 的参与 mask、所有线程到达 barrier、尾部填充值与输出边界。若使用半精度指数近似，单独记录允许误差。

交付正确性结果与按 shape 分组的延迟、有效带宽和资源数据。挑一个“减少扫描／增加向量宽度／加 shared memory 后变慢”的反例，检查 registers、spill、occupancy、内存事务、barrier 或 bank conflict。最终说明优化适用的形状，而不是宣布某个版本普遍最快。

## P8 容量账本与部署选择误差

关联 [Q51](platform-deployment.md#q51)、[Q56](algorithms-kernels.md#q56)–[Q59](algorithms-kernels.md#q59)、[Q65](platform-deployment.md#q65)、[Q68](algorithms-kernels.md#q68)。

交付两张可复算的表。第一张从固定模型配置逐项算逻辑总参数、每 token 激活的权重计算、每 rank 实际常驻权重、精度与 scale 开销、KV、workspace 和 Graph 预留；显式列 dense／MoE、MHA／GQA／MLA 及并行切分差异。训练另列梯度、优化器和反向激活。第二张固定实际硬件 SKU、软件后端、拓扑与工作负载，记录候选部署的 latency 来源：实测、同域插值、迁移估计或 SOL。

没有设备时可使用带标签的教学数据演练公式，但不能给出测得的加速结论。有实测后，将建模数据与按硬件／模型／shape 族留出的验证集分开；同时报告 MAPE、分桶误差、接近零时的处理、SLO 误判和最终选型的 regret。例如预测 A=90 ms、B=100 ms，却实测 A=105 ms、B=95 ms，两个单点百分比误差已经说明排序可能翻转。还需评价吞吐、质量和资源预算。

至少展示一项被账本或留出数据推翻的结论：37B 激活不等于 37B 常驻、KV 减半不等于 TPOT 减半，或平均误差低不等于在目标负载下选对配置。保留各项公式、单位、出处和未测范围。

## P9 PD 离散事件仿真设计与校准

关联 [Q09](engine-development.md#q09)、[Q35](platform-deployment.md#q35)、[Q36](platform-deployment.md#q36)、[Q64](platform-deployment.md#q64)、[Q66](platform-deployment.md#q66)、[Q67](platform-deployment.md#q67)。此任务先做设计与手工事件回放，不要求把当前仓库的轻量调度示例当作完整 PD 仿真器。

固定三个带到达时间的请求、一个 P replica、一个 D replica、一条受限传输链路及明确 KV 容量。画出 waiting／prefill／transit／decode／finished 及 cancel／failure 转移，定义 admission、源／目的页的分配和释放、可重叠的资源、同时刻事件的稳定顺序。以虚拟时钟走至少两轮，写出每个事件前后的队列、busy-until、KV 账本与后继事件。

先用退化情景验证守恒：链路耗时设零、D 容量仅容一条请求、一次传输中取消、同一 GPU 承担 P 和 D、一个 rank 变慢。任何时刻都不能重复分配物理页或让同一执行资源凭空并发；P 完成也不自动证明 D 可接纳，真实恢复／提交门槛见 Q36。Mixed Chunk 应编码为可比较的策略，不能预设每次都最优。

交付事件表、资源不变量、服务时间数据来源、仿真边界和校准计划。具备真实服务后，分别校准 compute、queue、transfer、KV admission，再比较端到端 TTFT／TPOT 分布和排队曲线。根据误差决定能否用来选 P/D 配比，不能用几条相符的均值宣称尾延迟模型成立。

## 单题解释与追问演练

每题先看“学习目标”，确定这次要证明的能力，再完成以下演练。参考答案负责把机制讲透，练习负责检查自己能否独立重建解释。

1. **讲清原始问题。**描述没有该机制时一次请求、一个算子或一个部署如何运行，成本在哪里。比如 Q09 先交代长 Prefill 为什么推迟 B 的下一 token，不能从“chunk 提升公平性”直接开讲。
2. **走通具体例子。**用题内数字逐步算，并说明每个中间值下一步交给谁。Q22 应自己算出两块 softmax 的 max、归一化和及输出；Q35 应把 P 等待、计算、传输、D 接纳逐项计入，而非只背带宽公式。
3. **改变一个条件。**由提问者改变 batch、前缀命中、带宽、取消时机或运行分支，重新推演。Q09 可把 Mixed 关掉，或删除换空运行批这一步，检查何时会得到 `[B,B]`。
4. **给出可区分的证据。**对同一个症状至少提出两个竞争原因，说明分别观察什么、什么结果支持或推翻它们。例如“ITL 高”可能是没有执行 B，也可能是混批本身慢，两者需要不同时间线。
5. **组织技术讲解。**先给结论和适用场景，再解释例子的关键步骤，随后说明取舍和验证方法。未做过的实验讲成验证计划，已做过的实验提供条件和证据；不把文档中的教学数字变成个人业绩。

演练合格的标志是能解释“为什么这一步必须存在”“省略后具体坏在哪里”和“换一个条件结论怎样变”。说出函数名、公式名称或某个指标改善方向，只能作为继续展开的入口。

## 单题学习记录模板

完成讨论后在总目录更新状态，并在相应题末补简短记录；不要把未运行的实践勾成完成。

```text
题号与日期：
本题考察重点、知识点与要展示的能力：
我最初的解释与需要修正处：
固定场景（模型、硬件、模式、长度／batch）：
基线机制与成本来源：
具体算例、关键状态和下一消费者：
改变一个条件后的重新推演：
核对的文件、符号、行号：
检查点答案与仍解释不清的因果：
竞争假设、支持或推翻它们的观测：
实践证据（源码／CPU／单GPU／多GPU／真实PD）：
尚未解决的问题与下次起点：
学习状态（学习中／待复习／已掌握）：
```

## 实践结果记录模板

```text
任务与假设：
版本（HEAD、工作树差异、模型与依赖）：
硬件、拓扑、协议、完整启动配置：
工作负载、到达模型、采样、缓存状态：
基线与唯一改变项：
计时范围、指标口径、SLO、成功／失败数：
正确性检查和失败路径：
原始日志、trace、数据与复现入口：
测得结果、波动与瓶颈证据：
结论适用范围、未覆盖组合与恢复检查：
```

技术总结可以按“当时的约束 → 可证伪假设 → 对照与首个异常 → 方案取舍 → 测量结果与边界”组织。先说明自己实际完成的层级，再回答能够继续验证的下一步。
