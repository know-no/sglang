# 推理岗位实践任务与学习记录

[题库总索引](README.md) · [资料与源码基线](references.md)

六项任务把参考答案转成自己能够复述和复现的经历。按可用设备逐步完成；源码追踪、CPU reference、单 GPU、多 GPU、CUDA Graph／overlap、真实 PD 传输分别记录证据。题库中的例子是推演输入，实验结果在完成后填写。

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

## 单题学习记录模板

完成讨论后在总目录更新状态，并在相应题末补简短记录；不要把未运行的实践勾成完成。

```text
题号与日期：
本次目标：
我最初的解释与需要修正处：
固定场景（模型、硬件、模式、长度／batch）：
关键状态与下一消费者：
核对的文件、符号、行号：
检查点答案：
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

面试叙述可以按“当时的约束 → 可证伪假设 → 对照与首个异常 → 方案取舍 → 测量结果与边界”组织。先说明自己实际完成的层级，再回答能够继续验证的下一步。
