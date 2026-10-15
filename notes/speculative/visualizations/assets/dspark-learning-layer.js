(() => {
  "use strict";

  const HEAD = "f01a0c7f97ec";
  const O = (instance, type, role, owner, input, output, next, kind = "data") => ({ instance, type, role, owner, input, output, next, kind });
  const F = (label, title, text, tokens) => ({ label, title, text, tokens });
  const E = (anchor, action, state, next) => ({ anchor, action, state, next });

  const lessons = {
    "00": {
      conclusion: "dSpark V2 的稳定主语是 DSparkWorkerV2：Scheduler 只把 ScheduleBatch 交给它；Draft、预算、Target Verify、接受、KV 提交都由它持有的组件协作完成。",
      scope: "CUDA dSpark V2 · PP=1",
      intuition: [
        ["先抓住唯一入口", "run_batch() 的动态接收者是 scheduler.model_worker : DSparkWorkerV2，不是内部的 target TpModelWorker。", "main"],
        ["把一轮看成对象闭环", "DFlashDraftInputV2 进入，GenerationBatchResult 返回；中间对象都有明确生产者与下一消费者。", "data"],
        ["三条路线不改语义", "STATIC、COMPACT、FOLDED 改变验证长度或执行位置；Target 校验与最终提交合同仍必须成立。", "guard"]
      ],
      objects: [
        O("scheduler : Scheduler", "Scheduler", "调度请求并调用动态 worker", "Server runtime", "ScheduleBatch", "GenerationBatchResult", "Scheduler.process_batch_result()", "main"),
        O("scheduler.model_worker : DSparkWorkerV2", "DSparkWorkerV2", "dSpark 一轮总编排者", "scheduler", "ScheduleBatch + DFlashDraftInputV2", "GenerationBatchResult", "Scheduler result processor", "main"),
        O("self._proposer : DraftBlockProposer", "DraftBlockProposer", "装配 Draft forward 并产生 proposal", "DSparkWorkerV2", "anchor + VerifyWindow", "DraftProposal", "self._verify_planner", "data"),
        O("self._verify_planner : DSparkVerifyPlanner", "DSparkVerifyPlanner", "预算与 layout 决策", "DSparkWorkerV2", "confidence + SPS + batch", "budget + RaggedVerifyLayout?", "self._verify_executor", "guard"),
        O("self._verify_executor : TargetVerifyExecutor", "TargetVerifyExecutor", "Target forward、接受与 hidden commit", "DSparkWorkerV2", "candidates + layout", "TargetVerifyResult + AcceptOuts", "self._kv_injector", "cuda"),
        O("self._kv_injector : TargetHiddenKvInjector", "TargetHiddenKvInjector", "Target auxiliary hidden 投影并写入 Draft KV", "DSparkWorkerV2", "accepted input hidden", "prefix-valid Draft K/V", "next DFlashDraftInputV2", "data")
      ],
      frames: [
        F("Before", "轮次 t 开始", "跨轮状态只带已提交长度与 bonus anchor；候选草稿尚不存在。", [["X@20", "data"], ["seq=20", "off"], ["draft_input", "data"]]),
        F("执行中", "Draft → layout → Target → accept", "DraftProposal、layout、TargetVerifyResult、AcceptOuts 依次接力；每个对象只服务本轮。", [["X", "data"], ["a", "main"], ["b", "guard"], ["c", "off"], ["d", "off"]]),
        F("After", "轮次 t 提交", "用户输出与下一轮 anchor 分开：提交 [a,Y]，Y 同时成为 t+1 的 bonus token。", [["a", "main"], ["Y@22", "guard"], ["seq=22", "main"]])
      ],
      delta: [
        ["batch.spec_info", "DFlashDraftInputV2(t)", "DFlashDraftInputV2(t+1)", "跨轮只保留下一轮需要的 durable 状态"],
        ["候选窗口", "尚未装配", "本轮对象释放/不再作为状态", "DraftProposal 与 DFlashVerifyInput 都是轮内对象"],
        ["seq_lens", "[20]", "[22]", "commit_lens=2 才能推进可见前缀"],
        ["next anchor", "X@20", "Y@22", "bonus 是下一轮 Draft Transformer 的真实起点"]
      ],
      wfb: [
        ["为什么", "组件拆分让 Draft、budget、Target 与 KV 注入分别可替换/复用，但 worker 保留一轮事务顺序。", "main"],
        ["失败", "若把 target_worker 当 run_batch 的接收者，会跳过 proposal、layout、accept 与跨轮状态构造。", "fail"],
        ["边界", "COMPACT、CUDA Graph、folded 是三个不同维度；某一维命中不代表另外两维也命中。", "guard"]
      ],
      evidence: [
        E("scheduler.py:3370–3577 · Scheduler.run_batch", "解析前向输入并调用 scheduler.model_worker.forward_batch_generation(batch)", "ScheduleBatch 进入 worker", "Scheduler.process_batch_result"),
        E("dspark_worker_v2.py:66–250 · DSparkWorkerV2.__init__", "构造 planner/proposer/executor/kv_injector", "组件所有权固定", "forward_batch_generation"),
        E("dspark_worker_v2.py:482–700 · DSparkWorkerV2._forward_decode", "按 proposal→budget→layout→verify→accept→commit 编排", "GenerationBatchResult", "Scheduler result processor")
      ],
      faq: [
        ["run_batch 最终调用哪个 forward_batch_generation？", "当前 dSpark 配置下调用 scheduler.model_worker : DSparkWorkerV2.forward_batch_generation；内部 Target 才调用 self._target_worker : TpModelWorker。"],
        ["layout=None 是否表示不运行 Target Verify？", "不是。它只表示没有 ragged compact layout；STATIC 仍按完整 B×W 候选窗口执行 Target forward。"],
        ["一轮结束后保留 TargetVerifyResult 吗？", "不把它原样跨轮保存。接受、hidden commit 与 next draft input 会提炼出下一轮真正需要的状态。"],
        ["哪一页先看 CUDA Graph？", "先读 05 的 Target runner 分派，再读 08/08A 的 folded capture、begin_step、replay 与 buffer 采用边界。"]
      ]
    },

    "01": {
      conclusion: "_forward_isolation 是 ScheduleBatch 的轮内事务：临时让同一个对象呈现 TARGET_VERIFY 视图，ForwardBatch 捕获执行引用后立即恢复，再只提交 durable 结果。",
      scope: "Scheduler CPU 线程 · overlap/non-overlap 共用",
      intuition: [
        ["临时视图不是新批次", "没有复制整个 ScheduleBatch；context manager 快照并重绑关键字段，finally 恢复。", "guard"],
        ["ForwardBatch 已经脱钩", "init_new() 构造出的 ForwardBatch 持有 input_ids/spec_info 等执行引用，恢复 ScheduleBatch 不会回收它们。", "data"],
        ["恢复与提交分两步", "restore 防止临时 verify 视图泄漏；commit 再把接受长度与下一轮 spec_info 写回。", "main"]
      ],
      objects: [
        O("batch : ScheduleBatch", "ScheduleBatch", "跨 Scheduler 与 worker 复用的可变批对象", "Scheduler", "DECODE fields", "临时 TARGET_VERIFY view", "Scheduler result processing", "guard"),
        O("snapshot : tuple", "_forward_isolation local state", "保存 input_ids/spec_info/forward_mode 等旧引用", "Scheduler._forward_isolation", "batch fields", "restore material", "finally block", "off"),
        O("verify_input : DFlashVerifyInput", "DFlashVerifyInput", "只描述本轮 Target verify 窗口", "TargetVerifyExecutor", "candidates + layout", "attention/spec metadata", "ForwardBatch.init_new", "data"),
        O("verify_forward_batch : ForwardBatch", "ForwardBatch", "已装配的执行快照", "ModelRunner boundary", "temporary batch view", "Target model inputs", "TpModelWorker", "cuda"),
        O("batch_result : GenerationBatchResult", "GenerationBatchResult", "携带 commit 与下一轮 spec_info", "DSparkWorkerV2", "AcceptOuts + next draft input", "durable writes", "Scheduler.process_batch_result", "main")
      ],
      frames: [
        F("Before", "普通 DECODE 视图", "batch 指向跨轮 DFlashDraftInputV2，forward_mode=DECODE，input_ids 是当前 anchor/调度输入。", [["DECODE", "main"], ["X", "data"], ["draft_info", "data"]]),
        F("临时", "TARGET_VERIFY 视图", "同一个 batch 临时绑定 candidates=[X,a,b,c,d] 与 DFlashVerifyInput，供 ForwardBatch.init_new() 装配。", [["TARGET_VERIFY", "guard"], ["X", "data"], ["a", "main"], ["b", "main"], ["c", "off"], ["d", "off"]]),
        F("After", "恢复并 durable commit", "旧 input_ids/forward_mode 先恢复；随后 result processor 写入 seq_lens=[22] 与下一轮 anchor Y。", [["DECODE", "main"], ["a", "main"], ["Y@22", "guard"]])
      ],
      delta: [
        ["batch.forward_mode", "DECODE", "临时 TARGET_VERIFY → 恢复 DECODE", "否则 result processor 最先按 Prefill/extend 路径错分"],
        ["batch.input_ids", "跨轮输入", "临时候选 → 恢复原引用", "ForwardBatch 已保存执行引用，不会丢输入"],
        ["batch.spec_info", "DFlashDraftInputV2", "临时 DFlashVerifyInput → next DFlashDraftInputV2", "轮内视图不能直接跨轮"],
        ["batch.seq_lens", "[20]", "[22]", "只由接受后的 durable commit 推进"]
      ],
      wfb: [
        ["为什么", "ModelRunner/attention backend 从 ScheduleBatch 视图装配正确的 candidates、positions 与 spec metadata。", "main"],
        ["如果不临时绑定", "ForwardBatch 会读到 decode anchor 与旧 spec_info，Target 行数、attention 窗口和 hidden 对齐都会错。", "fail"],
        ["边界", "恢复不是撤销 GPU 工作；恢复的是 Scheduler 对象引用，已构造 ForwardBatch 仍保活其 tensor。", "guard"]
      ],
      evidence: [
        E("scheduler.py:3326–3367 · Scheduler._forward_isolation", "snapshot、yield、finally restore", "同一 ScheduleBatch 回到稳定视图", "run_batch/result processor"),
        E("scheduler.py:3496–3505 · Scheduler.run_batch non-overlap", "resolve_forward_inputs 后进入 isolation，再调用 worker", "临时绑定范围", "GenerationBatchResult"),
        E("dflash_info.py:53–94 · DFlashVerifyInput.prepare_for_verify", "把 verify 视图装配为 ForwardBatch", "执行引用已捕获", "TpModelWorker.forward_batch_generation"),
        E("scheduler.py:3653–3673 · Scheduler.process_batch_result", "按 forward_mode 分派结果", "durable request/batch state", "下一调度轮")
      ],
      faq: [
        ["恢复 input_ids 为什么不让正在运行的 Target 丢输入？", "因为 Target 接收的是已经构造好的 ForwardBatch/tensor 引用，不会在 kernel 执行中重新读取 ScheduleBatch.input_ids。"],
        ["DFlashVerifyInput 为什么不能直接跨轮？", "它绑定本轮 candidates、layout、positions 与窗口 KV；下一轮 anchor、长度与候选都已变化。"],
        ["只忘记恢复 forward_mode，哪里最先出错？", "Scheduler.process_batch_result() 根据 forward_mode 选择 prefill/decode 处理；会最先发生错误分派。"],
        ["是否会复制整个 ScheduleBatch？", "不会；核心机制是保存并恢复字段引用，因此 finally 覆盖所有退出路径很关键。"]
      ]
    },

    "01A": {
      conclusion: "non-overlap 与 overlap 都执行 resolve_forward_inputs 和 _forward_isolation；区别不是‘跳过方法’，而是 FutureMap 值何时可用、由谁等待，以及异步引用需要保活多久。",
      scope: "Scheduler 时间线 · CPU/设备生命周期",
      intuition: [
        ["调用与做事分开看", "pure decode 的某些字段已经是 tensor，resolve_forward_inputs 仍被调用，但 guard 让它成为 no-op。", "guard"],
        ["FutureMap 管值", "overlap 把未完成结果以 future key 发布，后续轮在真正消费前 resolve。", "data"],
        ["record buffer 管寿命", "CUDA 异步工作可能晚于 Python 返回；batch_record_buf 保留仍被设备引用的对象。", "cuda"]
      ],
      objects: [
        O("scheduler.future_map : FutureMap", "FutureMap", "发布/解析跨轮值", "Scheduler overlap runtime", "Future keys", "resolved tensors/scalars", "resolve_forward_inputs", "data"),
        O("batch : ScheduleBatch", "ScheduleBatch", "携带值或 Future key", "Scheduler", "spec_info/hidden/confidence refs", "resolved forward view", "_forward_isolation", "guard"),
        O("batch_record_buf", "deque/list of retained batches", "延长异步 tensor/object 生命周期", "Scheduler", "previous batch refs", "safe release", "next stream/event boundary", "cuda"),
        O("scheduler.model_worker", "DSparkWorkerV2", "同步返回或 overlap 发布结果", "Scheduler", "resolved ScheduleBatch", "GenerationBatchResult/FutureMap writes", "result processor", "main")
      ],
      frames: [
        F("N", "non-overlap 同轮完成", "resolve 被调用；字段已解析则 guard-only no-op。isolation 包住同步 worker call，返回后直接处理 result。", [["resolve", "guard"], ["forward", "main"], ["result", "data"]]),
        F("N", "overlap 发布未完成值", "worker 结果中的可跨轮值先 stash/publish；CPU 可继续调度，设备工作由 stream/event 约束。", [["FutureKey", "data"], ["CUDA event", "cuda"], ["record ref", "cuda"]]),
        F("N+1", "消费前解析", "下一消费者调用 resolve_forward_inputs，把 future key 替换成真实值，再进入相同 isolation。", [["resolve", "main"], ["tensor", "data"], ["isolation", "guard"]])
      ],
      delta: [
        ["forward input field", "tensor 或 future key", "消费前统一为 tensor", "resolve 的 guard 决定是否真正查 FutureMap"],
        ["result ownership", "non-overlap 当前栈帧", "overlap FutureMap + retained refs", "把值可用性与对象寿命分开"],
        ["isolation", "两种模式都需要", "两种模式都 restore", "overlap 不改变临时视图的正确性要求"]
      ],
      wfb: [
        ["为什么", "异步 overlap 允许 CPU 调度与设备执行重叠，但必须显式表达 future 值与 stream 生命周期。", "main"],
        ["失败", "只 publish 值却不保活引用，设备可能继续读取已被复用/释放的 backing tensor。", "fail"],
        ["边界", "resolve_forward_inputs 被调用不等于每次阻塞；输入已是实值时 guard 直接返回。", "guard"]
      ],
      evidence: [
        E("overlap_utils.py:84–116 · resolve_forward_inputs", "按字段类型/guard 解析 FutureMap", "ScheduleBatch 获得实值", "_forward_isolation"),
        E("scheduler.py:3398–3505 · Scheduler.run_batch", "overlap/non-overlap 分支都 resolve + isolation", "不同等待时点", "model_worker forward"),
        E("overlap_utils.py:153–229 · ConfidenceRelay", "维护 generation-aware relay", "历史 confidence lag", "budget planner"),
        E("scheduler.py:3608–3621", "保留上一轮 batch/tensor 引用", "异步 lifetime", "安全复用")
      ],
      faq: [
        ["non-overlap 第 3503 行是不是直接跳过 resolve_forward_inputs？", "不是，函数确实被调用；当字段已是实际值而非 future key 时，内部 guard 很快返回。"],
        ["FutureMap 能替代 batch_record_buf 吗？", "不能。FutureMap 解决值的发布/解析；record buffer 解决 Python/tensor backing 的生命周期。"],
        ["overlap 会跳过 _forward_isolation 吗？", "不会。它仍需临时 verify 视图，只是 worker call 与后续值解析的时间组织不同。"],
        ["什么会导致跨请求读到旧 confidence？", "generation/row 复用若不校验会造成陈旧值；ConfidenceRelay 的 generation guard 用于阻断这种串线。"]
      ]
    },

    "02": {
      conclusion: "Prefill 先用 Target 对真实 prompt 建立完整语义，再把 Target auxiliary hidden 投影为 Draft 每层 K/V；Target 预测的 X 只是下一轮 anchor，本轮还没有 K/V(X)。",
      scope: "Prefill bootstrap · Target→Draft KV",
      intuition: [
        ["Draft 不能凭空起步", "dSpark Draft 层需要与 Target 语义对齐的 prefix K/V，bootstrap 由 Target hidden 提供。", "main"],
        ["预测与输入错一位", "hidden@19 预测 X@20；得到 token X 不代表已经 forward 过 X。", "guard"],
        ["写入只覆盖真实输入", "KV injector 只向 prompt 的 prefix-valid slots 写入，不会为未来 anchor 伪造 hidden/KV。", "data"]
      ],
      objects: [
        O("self : DSparkWorkerV2", "DSparkWorkerV2", "Prefill 编排与 hidden 交接", "Scheduler.model_worker", "ScheduleBatch(PREFILL)", "GenerationBatchResult + DFlashDraftInputV2", "Scheduler", "main"),
        O("self._target_worker : TpModelWorker", "TpModelWorker", "运行 Target prefill", "DSparkWorkerV2", "ForwardBatch", "logits + FULL hidden", "self._kv_injector", "cuda"),
        O("target_worker.model_runner", "ModelRunner", "选择 eager/prefill graph/split prefill runner", "TpModelWorker", "ForwardBatch", "target hidden", "attention backend", "cuda"),
        O("self._kv_injector", "TargetHiddenKvInjector", "投影 auxiliary hidden 为 Draft K/V", "DSparkWorkerV2", "hidden + cache loc", "Draft layer KV writes", "Draft ModelRunner", "data"),
        O("next_spec_info", "DFlashDraftInputV2", "保存 X 与长度作为 decode 起点", "DSparkWorkerV2", "sampled X + seq_lens", "bonus_tokens=[X]", "next decode round", "guard")
      ],
      frames: [
        F("Before", "prompt 尚未建立 Draft KV", "请求长度 20；Target/Draft cache slot 已分配，但 Draft 尚无与 Target 对齐的 prefix state。", [["0…19 prompt", "data"], ["Draft KV empty", "off"]]),
        F("Forward", "Target prefill + hidden 注入", "Target 处理 positions 0…19，返回 FULL auxiliary hidden；injector 投影并写入 Draft K/V(0…19)。", [["h0…h19", "cuda"], ["K/V 0…19", "main"]]),
        F("After", "X 成为下一轮 anchor", "logits@19 采样 X@20；Draft KV 仍只到 19，下一轮 Draft forward 才把 X 作为真实输入写到 20。", [["X@20", "guard"], ["KV 0…19", "main"], ["KV20 absent", "off"]])
      ],
      delta: [
        ["Target KV", "无/已有 cache prefix", "包含本轮真实 prompt 输入", "由 Target prefill runner 写入"],
        ["Draft KV", "未 bootstrap", "包含 prompt positions 0…19", "TargetHiddenKvInjector 写 prefix-valid slots"],
        ["next anchor", "无", "X@20", "来自 Target next-token logits"],
        ["K/V(X@20)", "不存在", "仍不存在", "X 尚未作为输入进入 Transformer"]
      ],
      wfb: [
        ["为什么", "Target auxiliary hidden 是 Draft KV 投影的语义源，使后续轻量 Draft 能从真实 prompt 状态起步。", "main"],
        ["失败", "若把 X 提前写入 Draft KV，需要一个不存在的 hidden(X)；位置与因果语义都会错一位。", "fail"],
        ["边界", "ModelRunner 内部必须区分 eager、prefill CUDA Graph 与 split prefill；本页只陈述它们共同的输出合同。", "cuda"]
      ],
      evidence: [
        E("dspark_worker_v2.py:361–432 · _forward_prefill", "请求 FULL hidden、调用 target worker、注入 Draft KV", "next DFlashDraftInputV2", "Scheduler"),
        E("tp_worker.py:532–621 · TpModelWorker.forward_batch_generation", "把 ForwardBatch 送入 ModelRunner", "logits/hidden", "DSparkWorkerV2"),
        E("model_runner.py:1335–1580", "选择 eager/prefill graph/split runner", "Target execution path", "attention backend"),
        E("dspark_kv_inject.py:13–78 · inject_target_hidden", "投影并写 Draft K/V", "prefix-valid cache", "next Draft forward")
      ],
      faq: [
        ["Target 已经预测 X，为什么 Draft KV 没有 X？", "next-token logits 来自前一个输入位置的 hidden；X 自己还没被 Transformer 消费。"],
        ["Prefill 必须 capture_hidden_mode=FULL 吗？", "dSpark bootstrap 需要每个真实输入位置的 auxiliary hidden 来构造 Draft 层 KV，而不只是最后一个 hidden。"],
        ["存在 prefix-cache 命中时 positions 仍是 0…19 吗？", "不一定；实际 extend 从 prefix_lens 起。固定例子假设无 prefix-cache 命中。"],
        ["overlap 会改变 K/V 语义吗？", "不会，只改变结果发布/引用保活时点；prefix-valid 写入范围不变。"]
      ]
    },

    "03": {
      conclusion: "Draft Transformer 只对 [X,MASK,MASK,MASK] 做一次四行 forward；a→b→c→d 是之后由 Markov head 用这四个 hidden 串行修正 logits 并采样得到的。",
      scope: "B=1 · G=4 · W=5",
      intuition: [
        ["并行准备槽位语义", "四个 MASK 行让 Transformer 一次产生四个位置的 base hidden/logits。", "cuda"],
        ["串行补 token 依赖", "b 的 proposal 分布必须看 a，c 必须看 a/b；Markov 小头承担这条依赖链。", "main"],
        ["proposal 保存两套东西", "draft_block_ids 保留形状输入；draft_tokens 保存真实 a/b/c/d，sampling 还保留实际 P_D 材料。", "data"]
      ],
      objects: [
        O("draft_input : DFlashDraftInputV2", "DFlashDraftInputV2", "提供跨轮 anchor X", "previous round", "bonus_tokens=[X]", "draft start state", "DraftBlockProposer", "data"),
        O("verify_window : VerifyWindow", "VerifyWindow", "预留 W=G+1 个 Target slots", "DSparkWorkerV2", "seq_lens/cache pool", "positions/cache loc", "Draft forward + Target verify", "guard"),
        O("draft_forward_batch", "ForwardBatch", "承载 [X,MASK×3] 四行", "DraftBlockProposer", "anchor + mask shape", "Draft runner metadata", "draft_model_runner", "data"),
        O("self._proposer.draft_model_runner", "ModelRunner", "一次计算四个 hidden/base logits", "DraftBlockProposer", "ForwardBatch[4 rows]", "raw_hidden[4,H]", "Markov head", "cuda"),
        O("markov_head", "Vanilla/Gated/RNN Markov head", "串行修正与采样", "Draft model", "base logits + prior sampled tokens", "a,b,c,d + corrected_logits", "DraftProposal", "main"),
        O("proposal : DraftProposal", "DraftProposal", "本轮 proposal 合同", "DraftBlockProposer", "tokens/hidden/P_D material", "planner + verifier inputs", "DSparkVerifyPlanner", "data")
      ],
      frames: [
        F("Before", "跨轮 anchor 与五格窗口", "prefix=20，X@20；VerifyWindow 已保留 positions 20…24。", [["X@20", "data"], ["W=5", "guard"]]),
        F("Forward", "Draft 只运行四行", "输入 [X,MASK,MASK,MASK]，positions 20…23；一次 Transformer forward 得到四个 hidden。", [["X", "data"], ["MASK", "off"], ["MASK", "off"], ["MASK", "off"]]),
        F("After", "Markov 返回真实 proposal", "四步串行采样得到 [a,b,c,d]；Target 候选随后才装成 [X,a,b,c,d]。", [["X", "data"], ["a", "main"], ["b", "main"], ["c", "main"], ["d", "main"]])
      ],
      delta: [
        ["draft input ids", "[X,MASK,MASK,MASK]", "保持为 shape/block 输入", "不会被真实 tokens 覆盖"],
        ["draft_tokens", "不存在", "[a,b,c,d]", "Markov 串行采样的结果"],
        ["seq_lens", "[20]", "device prefix 仍以 20 装配", "attention prefix 边界"],
        ["seq_lens_cpu", "[20]", "[24]", "CPU 计入四个 Draft 行，不代表已接受"]
      ],
      wfb: [
        ["为什么", "一次 Transformer forward 摊薄重计算；Markov 小头用较低成本恢复后置 token 对前置 token 的依赖。", "main"],
        ["失败", "若 sampling 丢失 corrected_logits/temperature，06 无法重建实际 proposal 分布 P_D，拒绝采样会失真。", "fail"],
        ["边界", "folded proposal 只在 Draft CUDA Graph + sampler + 全贪心 guard 下读图内 buffer；不代表 Target accept 也 folded。", "guard"]
      ],
      evidence: [
        E("dspark_worker_v2.py:482–561 · _forward_decode", "分配窗口、调用 proposer、把 confidence 交 planner", "DraftProposal", "schedule_layout"),
        E("dspark_draft.py:241–426 · DraftBlockProposer", "构造 ForwardBatch、Draft forward、采样、封装", "proposal fields", "DSparkWorkerV2"),
        E("dspark_draft.py:166–238 · sample_draft_block", "选择 folded/greedy/sampling 并保留分布材料", "draft tokens + P_D material", "accept sampling"),
        E("models/dspark.py:34–263 · Markov heads", "逐位置修正 logits 并采样", "a→b→c→d", "DraftProposal")
      ],
      faq: [
        ["Draft forward 的输入是不是 [X,a,b,c,d]？", "不是。Draft Transformer 输入是四行 [X,MASK,MASK,MASK]；[X,a,b,c,d] 是之后装给 Target 的五格候选。"],
        ["Markov 串行是否重复四次 Transformer？", "不重复。四个 hidden 已一次并行计算；串行的是小头 logits 修正与采样。"],
        ["为什么 W=5 而 Draft 只跑 G=4？", "Target 需要 anchor 行来预测第一个 draft token，因此 verify window 是 1 anchor + 4 drafts。"],
        ["seq_lens_cpu=[24] 是否表示已提交到 24？", "不是；它是本次 Draft forward 的 CPU 侧计数，最终提交长度由 AcceptOuts.commit_lens 决定。"]
      ]
    },

    "03A": {
      conclusion: "Vanilla、Gated、RNN 三种 Markov head 的共同边界是：Transformer hidden 同时就绪；每一步 logits 修正与采样必须串行，且不会为后置 token 回写真实 hidden/KV。",
      scope: "Markov head 内部 · 非 Transformer 重算",
      intuition: [
        ["hidden 是位置槽位表征", "MASK 槽位 hidden 预先并行得到，但不含最终 a/b/c/d 的完整因果表征。", "data"],
        ["token 链驱动小头", "第 i 步刚采样出的 token 会成为第 i+1 步 Markov 修正的输入。", "main"],
        ["不要误当 autoregressive forward", "后置 token 没有重新经过 backbone，所以 Draft KV 不会被它们的真实 hidden 覆盖。", "guard"]
      ],
      objects: [
        O("draft_hidden_3d", "Tensor[B,G,H]", "四个并行 hidden", "Draft ModelRunner", "[X,MASK×3]", "per-step hidden slices", "Markov head", "data"),
        O("base_logits", "Tensor[B,G,V]", "每槽基础分布", "Draft model", "draft hidden", "step logits", "sample_block", "data"),
        O("markov_head", "VanillaMarkov/GatedMarkovHead/RNNHead", "根据已采 token 计算 bias/state", "Draft model", "token prefix + base logits", "corrected logits", "sampler", "main"),
        O("draft_tokens", "Tensor[B,G]", "串行生成的真实 token 链", "sample_block", "step samples", "[a,b,c,d]", "Target candidates", "main")
      ],
      frames: [
        F("并行", "一次得到 h0…h3", "Transformer 对四行同时完成，不依赖 a/b/c/d 已经存在。", [["h0(X)", "cuda"], ["h1(M)", "cuda"], ["h2(M)", "cuda"], ["h3(M)", "cuda"]]),
        F("串行", "a→b→c→d", "step0 采 a；step1 用 a 修正 b；step2 用 a/b 修正 c；step3 用 a/b/c 修正 d。", [["a", "main"], ["b|a", "main"], ["c|a,b", "main"], ["d|a,b,c", "main"]]),
        F("返回", "保留 token 与分布材料", "sampling 路径返回 corrected_logits/temperatures；真实后置 hidden/KV 仍不存在。", [["tokens", "data"], ["P_D", "guard"], ["no KV(b/c/d)", "off"]])
      ],
      delta: [
        ["Transformer forward count", "0", "1 次 G 行 forward", "不是每 token 一次"],
        ["Markov sample steps", "0", "G=4 次串行步骤", "恢复 proposal token 依赖"],
        ["Draft KV for a/b/c/d", "未写", "仍未按真实 token hidden 回写", "这些 token 未重新进入 backbone"],
        ["proposal distribution", "base logits", "Markov/temperature 后的实际 P_D", "sampling verify 必须使用它"]
      ],
      wfb: [
        ["为什么", "把昂贵 backbone 并行化，把依赖性压到小头串行路径，是 dSpark proposal 的关键折中。", "main"],
        ["失败", "把四个 base logits 独立采样会改变 b/c/d 的条件分布，Target sampling 修正也不再对应真实 proposal。", "fail"],
        ["边界", "三种 head 的 bias/state 公式不同，但本页不把某个变体的内部状态误推广到全部实现。", "guard"]
      ],
      evidence: [
        E("models/dspark.py:34–64", "公共逐位置 sample_block 合同", "step token dependency", "draft tokens"),
        E("models/dspark.py:67–263", "Vanilla/Gated/RNN 具体实现", "corrected logits/state", "DraftBlockProposer"),
        E("dspark_draft.py:177–238", "调用 Markov sampling 并封装分布材料", "DraftProposal", "Target verify/accept")
      ],
      faq: [
        ["h1 是否已经看见 a？", "作为 MASK 槽位的 backbone hidden，它没有通过一次新的 causal Transformer forward 看见真实 a；a 的影响由 Markov head 注入 logits。"],
        ["RNN head 会把后置 hidden 写回 KV 吗？", "不会。RNN 状态属于 Markov head 的采样内部状态，不等于 Draft Transformer KV。"],
        ["为什么 sampling 要保留 corrected_logits？", "拒绝采样需要实际 q=P_D；base logits 不是经过 Markov 与 temperature 后的 proposal 分布。"],
        ["greedy 是否也需要概率修正？", "greedy 接受按 token 相等比较，不执行 sampling residual 修正；但 Markov 仍决定 draft token 本身。"]
      ]
    },

    "04": {
      conclusion: "adaptive verify 同轮使用两套 survival：历史 C_(t-2) 与 SPS 选 batch-global extra budget；当前 C_t 再把这些名额按全局 top-k 分给请求。layout=None 只是不 compact，Target 仍运行。",
      scope: "默认 adaptive planner · B=2,G=4,W=5",
      intuition: [
        ["先决定总成本", "历史统计回答‘这一批多验证几行划算’，输出一个全 batch budget scalar。", "guard"],
        ["再决定分给谁", "当前 proposal confidence 回答‘哪几个 extra prefix slots 最值得验证’，输出 verify_lens。", "data"],
        ["layout 只描述物理形状", "STATIC 的 None、CAP_ACCEPT 的 lens-only、COMPACT 的 ragged layout 都不会取消 mandatory Target verify。", "main"]
      ],
      objects: [
        O("confidence_history", "historical confidence tensor", "两轮前的接受概率证据", "relay/planner carry", "C_(t-2)", "historical survival", "HostConfidenceBudgetPlanner", "guard"),
        O("self._verify_planner._host_budget_planner", "HostConfidenceBudgetPlanner", "结合 SPS 选 extra budget", "DSparkVerifyPlanner", "historical survival + SpsCostTable", "budget scalar", "ScheduleVerifyLensTopk", "main"),
        O("confidence : Tensor[B,G]", "current confidence", "本轮 proposal 的位置置信度", "DraftProposal", "C_t", "current survival", "ScheduleVerifyLensTopk", "data"),
        O("ScheduleVerifyLensTopk", "schedule kernel wrapper", "全局 top-k 分配 budget", "DSparkVerifyPlanner", "current survival + budget", "verify_lens", "RaggedVerifyLayout", "main"),
        O("layout", "RaggedVerifyLayout | None", "描述 gather/scatter 与 graph tier", "DSparkVerifyPlanner", "verify_lens + mode", "compact metadata or None", "TargetVerifyExecutor", "guard")
      ],
      frames: [
        F("历史", "C_(t-2) 选择多少", "历史 confidence→cumprod survival→tau_star→SPS objective，示例得到 budget=2 extra。", [["C_(t-2)", "guard"], ["budget=2", "main"]]),
        F("当前", "C_t 决定分给谁", "对当前 survival 的 extra slots 全局排名；两个名额分别落到 req0/req1，得到 verify_lens=[2,2]。", [["req0=2", "data"], ["req1=2", "data"]]),
        F("布局", "逻辑 4 行，物理 tier 5", "COMPACT gather 四个真实行并 pad 到五行 graph tier；STATIC layout=None 仍走完整 10 行。", [["logical=4", "main"], ["physical=5", "cuda"], ["strided=10", "off"]])
      ],
      delta: [
        ["budget", "None/未决", "2 extra slots", "不包含每请求 mandatory anchor"],
        ["verify_lens", "未决", "[2,2]", "每项包含 anchor floor 1"],
        ["layout", "None", "COMPACT: RaggedVerifyLayout", "STATIC 仍保持 None"],
        ["Target rows", "STATIC=10", "COMPACT logical=4 / physical=5", "layout 改成本，不改验证语义"]
      ],
      wfb: [
        ["为什么", "把总成本决策与本轮分配分开，可用稳定历史估计收益，同时利用当前 proposal 的请求差异。", "main"],
        ["失败", "把历史 survival 直接用于分配会把旧请求排序套到当前 batch；把当前 survival 用于 SPS 又会引入同轮时序依赖。", "fail"],
        ["边界", "budget=0 是明确零 extra；budget=None 是本轮无 budget 控制。二者不能互换。", "guard"]
      ],
      evidence: [
        E("dspark_worker_v2.py:538–612", "计算 current confidence、resolve budget、schedule layout、选择 compact", "Target verify dispatch", "TargetVerifyExecutor"),
        E("dspark_planner.py:373–522 · resolve_verify_token_budget/schedule_layout", "汇合历史与当前两条轨", "budget + layout", "worker"),
        E("dspark_planner.py:948–1118", "tau_star 与 SPS objective", "batch-global budget", "ScheduleVerifyLensTopk"),
        E("dspark_schedule.py:19–115", "current survival 全局 top-k", "verify_lens", "RaggedVerifyLayout")
      ],
      faq: [
        ["历史 survival 与当前 survival 各决定什么？", "历史决定‘多少’；当前决定‘分给谁’。它们各自独立 cumprod。"],
        ["layout=None 是否不运行 Target？", "不是。STATIC 正是 layout=None，但仍按完整 B×W 候选矩阵执行 Target。"],
        ["CAP_ACCEPT 得到 verify_lens 后 Target 只跑这些行吗？", "不一定；CAP_ACCEPT 可保留完整 Target forward，只在接受/截断语义上使用 lens。真正物理压紧由 COMPACT layout 驱动。"],
        ["verify_lens 是否包含 anchor？", "包含；budget 只数 anchor 之外的 extra slots。"]
      ]
    },

    "04A": {
      conclusion: "默认稳态下 overlap 与 non-overlap 都让预算使用两轮前 confidence；overlap 的 lag 由 ConfidenceRelay 持有，non-overlap 的 lag 由 HostConfidenceBudgetPlanner carry。",
      scope: "历史轨 · SPS budget 实验",
      intuition: [
        ["年龄相同，持有者不同", "不要只看某一函数的局部变量；沿 generation 从 t0 跟到 t2 才能看到真实 lag。", "guard"],
        ["survival 转成收益", "confidence 逐位置 cumprod 后形成 prefix 存活率，tau_star 的 prefixSum 随 budget 递增。", "data"],
        ["SPS 把收益变成速度", "更多 token 可能降低 steps/s；optimizer 比较每个 budget 档位的 theta，而不是只看接受率。", "main"]
      ],
      objects: [
        O("confidence_relay", "ConfidenceRelay", "overlap 下 generation-aware 两轮 relay", "overlap_utils", "C_t", "C_(t-2) or None", "DSparkVerifyPlanner", "data"),
        O("self._host_budget_planner", "HostConfidenceBudgetPlanner", "non-overlap carry + budget optimization", "DSparkVerifyPlanner", "historical survival", "budget", "worker", "main"),
        O("sps_table", "SpsCostTable/SpsAdditiveCostTable", "给 batch token 档位成本", "server profile/config", "batch_tokens", "steps_per_sec/step_time", "theta objective", "cuda"),
        O("tau_star", "Tensor[num_budgets]", "累计期望提交 token", "planner function", "historical survival prefix", "benefit curve", "budget argmax", "guard")
      ],
      frames: [
        F("t0", "产生 C0", "overlap 发布到 relay；non-overlap 放入 planner carry。此时都不能把 C0 当 t0 自己的预算历史。", [["C0", "data"], ["lag slot", "guard"]]),
        F("t1", "推进一轮", "C1 进入新的槽位；C0 仍只是一轮旧，generation guard 防止跨请求复用。", [["C1", "data"], ["C0", "off"]]),
        F("t2", "C0 参与预算", "C0→survival→tau_star；结合 SPS table 比较档位，固定教学例得到 budget=2。", [["C0", "guard"], ["tau*", "data"], ["budget=2", "main"]])
      ],
      delta: [
        ["history age", "warm-up/不可用", "稳态 C_(t-2)", "overlap 与 non-overlap 默认结论相同"],
        ["tau_star", "B anchors", "anchors + survival prefixSum", "随预算不减"],
        ["cost", "未查表", "SPS/step_time 对应档位", "收益必须结合 Target cost"],
        ["decision", "budget=None 或 fallback", "budget=2", "generation/profile guard 决定是否有效"]
      ],
      wfb: [
        ["为什么", "使用历史 confidence 解开同轮 proposal 与预算决策的依赖，并允许 overlap 提前准备。", "main"],
        ["失败", "请求/row 更新后若不校验 generation，planner 可能把上一请求的 confidence 当作当前历史。", "fail"],
        ["边界", "无 SPS profile 时不是得到可靠的性能最优；flat fallback 常退化为 verify-all。", "guard"]
      ],
      evidence: [
        E("overlap_utils.py:118–229 · ConfidenceRelay", "publish/stash/resolve 并校验 generation", "历史 confidence", "planner"),
        E("dspark_planner.py:295–417", "根据 overlap 配置选择 relay lag/carry lag", "history source", "resolve budget"),
        E("dspark_planner.py:948–1020", "构造 tau_star 与 theta", "每个 budget objective", "argmax"),
        E("dspark_planner.py:1023–1130 · HostConfidenceBudgetPlanner", "warm-up/carry/generation guard", "budget or fallback", "DSparkWorkerV2")
      ],
      faq: [
        ["non-overlap 用的是当前 confidence 吗？", "默认稳态不是；planner carry 让它也使用两轮前 confidence。当前 confidence 另用于本轮 top-k 分配。"],
        ["为什么 tau_star prefixSum 递增？", "survival 非负；每增加一个 extra slot，只会增加或保持期望可提交 token。最终目标仍可能因 SPS 成本下降而提前最优。"],
        ["budget=0 与 None 有何不同？", "0 是明确选择零 extra；None 表示本轮没有有效预算控制，后续按 mode/fallback 处理。"],
        ["warm-up 一定返回 None 吗？", "不统一。overlap relay 未 ready 可为 None；non-overlap carry 的 generation 不匹配可回退全 1 survival 再结合 SPS。"]
      ]
    },

    "04B": {
      conclusion: "ScheduleVerifyLensTopk 对当前 C_t 再做 cumprod，把 batch-global extra budget 作为全局名额排名；每请求入选数量加 anchor floor，形成连续 prefix verify_lens。",
      scope: "当前轨 · top-k · graph tier alignment",
      intuition: [
        ["排名的是 prefix slot", "每个 extra 位置的分数是到该位置仍存活的概率，不是单点 confidence。", "data"],
        ["全局共享名额", "budget=2 不预先规定每请求各 1 个；top-k 可以按分数把名额偏向某个请求。", "main"],
        ["单调性恢复连续 prefix", "同一请求 survival 随位置不增，因此统计入选数即可得到从 anchor 开始的连续长度。", "guard"]
      ],
      objects: [
        O("confidence", "Tensor[B,G]", "当前 proposal 逐位置 confidence", "DraftProposal", "C_t", "survival", "schedule kernel", "data"),
        O("survival", "Tensor[B,G]", "cumprod 后的 prefix 存活率", "ScheduleVerifyLensTopk", "C_t", "global ranking scores", "top-k", "data"),
        O("allocator budget", "int extra slots", "限制全 batch 入选名额", "DSparkVerifyPlanner", "raw budget/alignment allowance", "k", "top-k", "guard"),
        O("verify_lens", "Tensor[B]", "anchor floor + 每请求入选 extra 数", "schedule kernel", "top-k ids", "[2,2] or [2,3]", "RaggedVerifyLayout", "main"),
        O("layout", "RaggedVerifyLayout", "logical rows→physical tier→strided scatter", "planner", "verify_lens", "extend_start_loc/indices", "TargetVerifyExecutor", "cuda")
      ],
      frames: [
        F("分数", "当前 survival 矩阵", "两请求各四个 extra 分数；每行沿 draft depth 不增。", [["r0:.72", "data"], ["r1:.81", "data"], ["r0:.39", "off"], ["r1:.44", "off"]]),
        F("Top-k", "raw budget=2", "全局 #1/#2 分别落到 req1 slot1 与 req0 slot1，两个请求各获得一个 extra。", [["#1 r1", "main"], ["#2 r0", "main"]]),
        F("布局", "anchor floor 后 [2,2]", "logical rows=4；graph tier=5。开启 alignment 可把 padding allowance 变为真实 #3，得到 [2,3]。", [["[2,2]", "main"], ["pad=1", "off"], ["[2,3]", "guard"]])
      ],
      delta: [
        ["raw budget", "2", "2", "alignment 不改历史 SPS decision"],
        ["allocator allowance", "2", "显式对齐时 3", "最多利用 graph tier 空槽"],
        ["verify_lens", "[1,1] anchor floor", "默认 [2,2] / 对齐 [2,3]", "包含 anchor"],
        ["physical rows", "未布局", "tier=5", "超出 logical 的槽为 padding 或额外真实验证"]
      ],
      wfb: [
        ["为什么", "用当前 survival 排名能把固定成本名额分给本轮最可能继续接受的位置。", "main"],
        ["失败", "若按单点 confidence 排名，可能选中一个前缀已低概率失败的后置位置，破坏连续验证的收益解释。", "fail"],
        ["边界", "--speculative-dspark-align-verify-tokens-to-graph-tier 默认关闭；开启后只扩大 allocator allowance。", "guard"]
      ],
      evidence: [
        E("dspark_schedule.py:19–115 · ScheduleVerifyLensTopk.execute", "cumprod、global top-k、按请求计数", "verify_lens", "planner"),
        E("dspark_planner.py:419–618 · schedule_layout", "处理 mode、budget、alignment 与 layout", "RaggedVerifyLayout/None", "Target verify"),
        E("ragged_verify.py:46–172 · RaggedVerifyLayout", "生成 logical/physical/scatter 元数据", "compact flat buffer", "DFlashVerifyInput")
      ],
      faq: [
        ["budget=2 为什么不是 verify_lens=[1,1]？", "budget 只数 extra；每请求 mandatory anchor floor 另加 1，因此各获得一个 extra 后是 [2,2]。"],
        ["top-k 会选同一请求的不连续后置位置吗？", "survival 沿位置不增；若后置位置入选，其前置分数不低，会先入选。按计数可恢复连续 prefix。"],
        ["graph-tier alignment 是不是修改 SPS 预算？", "不是。raw decision 仍为 2；allocator allowance 可扩到 3，用原本 padding 的物理槽做真实验证。"],
        ["extend_start_loc 在哪里用？", "它描述 compact flat buffer 中每个请求逻辑段的起点，供 gather/attention/scatter 对齐 ragged 行。"]
      ]
    },

    "05": {
      conclusion: "TargetVerifyExecutor 的 run_non_compact/run_compact 都是在做 Target forward + 结果对齐；差别是候选行是否先 gather 压紧、是否 pad 到 graph tier，以及 hidden/logits 如何 scatter 回固定 B×W 槽。",
      scope: "Target Verify · CUDA eager/decode graph",
      intuition: [
        ["Target 一次看完整候选链", "输入 [anchor,d1…] 让每个位置的 causal attention 同时覆盖 prefix KV 与前序候选。", "main"],
        ["逻辑行与物理行分开", "COMPACT 的 verify_lens 决定真实 logical rows；CUDA Graph tier 决定实际 physical rows。", "guard"],
        ["返回尚未接受", "TargetVerifyResult 与 hidden_strided 只是验证材料；06 才决定 token，07 才提交 Draft KV。", "data"]
      ],
      objects: [
        O("self._verify_executor", "TargetVerifyExecutor", "准备候选、选择 compact、调用 Target", "DSparkWorkerV2", "DraftProposal + layout", "TargetVerifyResult", "accept_and_finalize", "main"),
        O("verify_input", "DFlashVerifyInput", "提供 candidates/layout/attention 参数", "TargetVerifyExecutor", "B×W or compact rows", "spec metadata", "ForwardBatch.init_new", "data"),
        O("verify_forward_batch", "ForwardBatch", "冻结 Target 执行输入与 metadata", "DFlashVerifyInput", "temporary ScheduleBatch view", "input_ids/positions/spec_info", "target_worker", "data"),
        O("target_worker : TpModelWorker", "TpModelWorker", "Target 模型 worker 边界", "TargetVerifyExecutor", "ForwardBatch", "LogitsProcessorOutput", "ModelRunner", "cuda"),
        O("target_worker.model_runner", "ModelRunner", "选择 decode graph 或 eager extend runner", "TpModelWorker", "TARGET_VERIFY ForwardBatch", "logits + FULL hidden", "executor", "cuda"),
        O("attn_backend", "AttentionBackend", "规划 prefix KV + verify window attention", "ModelRunner", "DFlashVerifyInput attn args", "attention output", "Target model", "guard")
      ],
      frames: [
        F("候选", "B=2,G=4,W=5", "两请求各有五格 candidates；STATIC 直接 10 行，COMPACT verify_lens=[2,2] 只取四个逻辑行。", [["2×5", "data"], ["logical=4", "main"]]),
        F("执行", "physical graph tier=5", "COMPACT gather 四行并 pad 一行；ModelRunner 选择 decode_cuda_graph_runner.execute 或 eager_runner._execute_extend。", [["gather4", "main"], ["pad1", "off"], ["CUDA5", "cuda"]]),
        F("返回", "scatter 到 10 个 strided lanes", "logits/hidden 按 layout 映回 B×W backing；无效 suffix 保持不可见，TargetVerifyResult 交给 06。", [["valid4", "main"], ["strided10", "data"], ["suffix hidden", "off"]])
      ],
      delta: [
        ["Target input rows", "STATIC 10 / compact logical 4", "graph physical 5 或 eager 4", "runner 只接收物理装配后的行"],
        ["attention metadata", "未初始化", "prefix KV + verify window KV", "DFlashVerifyInput.generate_attn_arg_prefill 提供"],
        ["logits", "无", "Target next-token logits", "06 greedy/sampling acceptance"],
        ["hidden", "无", "FULL hidden → strided backing", "07 只提交接受前缀的输入侧 hidden"]
      ],
      wfb: [
        ["为什么", "Target 候选链可在 causal mask 下并行计算，避免逐 draft token 串行跑完整 Target。", "main"],
        ["失败", "compact gather 后若不 scatter 回固定 W stride，接受 kernel 会把请求/位置对应错，hidden commit 也会写错 KV。", "fail"],
        ["边界", "TARGET_VERIFY 在 ForwardMode 语义上属于 is_extend()；CUDA Graph 命中时复用 decode graph，eager 时走 _execute_extend。", "cuda"]
      ],
      evidence: [
        E("dspark_worker_v2.py:570–612", "根据 planner 选择 run_compact/run_non_compact", "TargetVerifyResult", "accept"),
        E("dspark_verify.py:215–425 · TargetVerifyExecutor", "装配、forward、gather/scatter", "target_verify + hidden_strided", "accept/commit"),
        E("dflash_info.py:53–163 · DFlashVerifyInput.prepare_for_verify", "构造 ForwardBatch 与 attention metadata", "TARGET_VERIFY execution input", "TpModelWorker"),
        E("model_runner.py:1479–1580 + eager_runner.py:194–338 + decode_cuda_graph_runner.py:1058–1288", "runner dispatch 与 eager/graph 执行", "logits/hidden", "TargetVerifyExecutor")
      ],
      faq: [
        ["run_non_compact/run_compact 是否就是 Target forward + verify？", "是，它们进入 Target worker 做 forward，并把输出整理为后续 acceptance 所需的验证结果；token 是否接受由 06 决定。"],
        ["STATIC layout=None 时 Target 跑几行？", "固定 B=2,W=5 例子跑完整 10 行，不会因为 layout=None 跳过。"],
        ["COMPACT verify_lens=[2,2] 为什么可能跑 5 行？", "四行是 logical rows；decode CUDA Graph 可能要求五行 tier，额外一行是物理 padding。"],
        ["ForwardBatch 构造后 attention metadata 在哪里准备？", "DFlashVerifyInput.prepare_for_verify 先判断图可行性；图分支 load_batch，eager 分支 init_forward_metadata，backend 读取 verify attention 参数。"]
      ]
    },

    "06": {
      conclusion: "接受阶段把 Target logits 与 Draft proposal 变成 AcceptOuts：greedy 比较 token；sampling 用实际 P_D 做拒绝采样和 residual 随机补偿；任何 cutoff 后都必须重选边界 bonus。",
      scope: "accept → finalize → output token backing",
      intuition: [
        ["接受长度不含 bonus", "correct_len 只数连续匹配/接受的 draft token；边界 Target token 作为 bonus 另算。", "main"],
        ["cutoff 改变预测边界", "若 verify_lens 截短，原完整窗口的 bonus 位置可能已不可用，必须从新的边界 logits 重新选择。", "guard"],
        ["backing 与有效前缀分开", "BuildOutTokens 可保留 [a,Y,c,d,0] backing；commit_lens=2 决定只有 [a,Y] 生效。", "data"]
      ],
      objects: [
        O("target_verify", "TargetVerifyResult", "提供 aligned logits/hidden 与图命中信息", "TargetVerifyExecutor", "Target forward output", "acceptance material", "accept_and_finalize", "data"),
        O("draft_block", "DraftBlock", "提供 draft tokens 与 sampling P_D 材料", "DraftProposal", "a,b,c,d", "proposal distribution", "AcceptGreedy/Sampling", "data"),
        O("AcceptGreedy/AcceptSampling", "accept kernel wrappers", "逐请求得到 correct_len/bonus/cap_trim", "accept_draft_tokens", "P_T + P_D + lens", "boundary decision", "FinalizeAcceptLens", "main"),
        O("FinalizeAcceptLens", "finalize kernel", "计算 commit_lens/new_seq_lens", "TargetVerifyExecutor", "correct_len + old seq", "finalized lengths", "BuildOutTokens", "guard"),
        O("BuildOutTokens", "output builder", "构造固定 backing 并以 lens 标记有效前缀", "TargetVerifyExecutor", "draft tokens + bonus", "output_token_ids", "AcceptOuts", "main"),
        O("accept : AcceptOuts", "AcceptOuts", "接受与提交的统一合同", "TargetVerifyExecutor", "all finalized fields", "commit/hidden/next input", "DSparkWorkerV2", "data")
      ],
      frames: [
        F("输入", "Target 看 [X,a,b,c,d]", "第一个 draft a 被接受；下一个边界由 Target 选择 Y。", [["X", "data"], ["a", "main"], ["b", "fail"], ["c", "off"], ["d", "off"]]),
        F("长度", "correct_len=1", "bonus=Y，commit_lens=correct_len+1=2，new_seq_lens=20+2=22。", [["a accepted", "main"], ["Y bonus", "guard"], ["len=2", "main"]]),
        F("输出", "backing 与有效前缀", "BuildOutTokens backing=[a,Y,c,d,0]；只有前 commit_lens=2 个 token 对用户可见。", [["a", "main"], ["Y", "main"], ["c", "off"], ["d", "off"], ["0", "off"]])
      ],
      delta: [
        ["correct_len", "未决", "1", "不包含 bonus"],
        ["bonus", "旧窗口边界候选", "Y", "cutoff 后按新边界重选"],
        ["commit_lens", "0", "2", "a + Y"],
        ["output_token_ids", "未构造", "[a,Y,c,d,0] backing / [a,Y] valid", "lens 控制有效输出"]
      ],
      wfb: [
        ["为什么", "统一 AcceptOuts 让后续 hidden commit、Scheduler 长度更新与输出处理不依赖具体接受算法。", "main"],
        ["失败", "sampling 拒绝后直接取 Target argmax 会改变 P_T；必须从 max(P_T-P_D,0) 归一化后的 residual 随机采样。", "fail"],
        ["边界", "cap_trim_lens 描述 ragged/cutoff 丢弃的潜在接受量，不只属于 CAP_ACCEPT 命名路径。", "guard"]
      ],
      evidence: [
        E("dspark_verify.py:91–149 · accept_and_finalize", "选择 folded/普通 accept，finalize 并 BuildOutTokens", "AcceptOuts", "DSparkWorkerV2"),
        E("dspark_verify.py:658–722 · accept_draft_tokens", "greedy/sampling/mixed 分派", "correct_len/bonus/cap_trim", "FinalizeAcceptLens"),
        E("dspark_accept.py:80–174,558–730", "接受与 sampling residual kernel", "per-request decision", "wrapper"),
        E("dspark_verify_window.py:766–827", "finalize lengths/output backing", "commit_lens/new_seq_lens/tokens", "commit")
      ],
      faq: [
        ["d1/d2 匹配而 d3 不匹配，greedy bonus 一定是 Target 的 b 吗？", "在完整未截短的 greedy 边界，bonus 是最后接受位置的 Target next-token prediction；若 layout/cutoff 改变边界，要按有效边界重新取。"],
        ["sampling 拒绝后多个 residual token 概率都为正怎么办？", "只按 residual 分布随机抽一个 token，不是同时接受多个，也不是取 argmax。"],
        ["为什么 backing 还保留 c/d？", "为固定形状/图执行复用 backing；commit_lens 明确有效前缀，suffix 不会被当作输出。"],
        ["folded accept 是否改变 AcceptOuts 语义？", "不改变；只把部分 kernel 放进 CUDA Graph，Worker 仍按 guard 决定是否采用图内 buffer。"]
      ]
    },

    "06A": {
      conclusion: "sampling 拒绝后从 residual ∝ max(p−q,0) 随机抽一个 token；在固定例中 u=.90 抽到 C，即使 Target argmax 是 B，也绝不能改成 argmax。",
      scope: "q=[.50,.20,.20,.10] · p=[.20,.45,.25,.10]",
      intuition: [
        ["接受先抵消共同质量", "draft=A 时接受率 min(1,p(A)/q(A))=.4；被拒绝的 draft 多余质量必须减掉。", "fail"],
        ["residual 补 Target 多出的质量", "max(p−q,0) 只在 B/C 为正，归一化为 [0,.8333,.1667,0]。", "data"],
        ["随机数决定单个结果", "CDF 上 u=.90 落在 C 区间；多个正概率只表示多个可能结果。", "main"]
      ],
      objects: [
        O("draft q=P_D", "Categorical distribution", "真实 proposal 分布", "corrected_logits + temperature", "draft token A", "accept ratio denominator", "AcceptSampling", "data"),
        O("target p=P_T", "Categorical distribution", "Target 条件分布", "Target logits", "same prefix", "accept ratio numerator", "AcceptSampling", "main"),
        O("accept uniform", "u_accept", "判断当前 draft 是否保留", "sampling RNG", "p(A)/q(A)=.4", "reject in example", "residual", "guard"),
        O("residual", "normalized max(p-q,0)", "修复拒绝后的 Target 分布", "reject sampling kernel", "p and q", "B/C probabilities", "CDF sampler", "data"),
        O("bonus", "single sampled token", "边界输出", "CDF sample", "u=.90", "C", "FinalizeAcceptLens", "main")
      ],
      frames: [
        F("提案", "draft 抽到 A", "q(A)=.50，而 Target p(A)=.20；A 只以 .4 概率被保留。", [["A q=.50", "data"], ["A p=.20", "main"]]),
        F("拒绝", "减去 Draft 多给的质量", "A 被拒绝；计算 positive residual：B=.25，C=.05，归一化后 B=.8333、C=.1667。", [["B .8333", "main"], ["C .1667", "guard"]]),
        F("补偿", "u=.90 → C", "CDF(B)=.8333，CDF(C)=1；.90 落在 C。只返回 C 一个 token。", [["B", "off"], ["C", "main"]])
      ],
      delta: [
        ["draft A", "已提案", "被拒绝", "draft 在 A 上多出的概率被扣除"],
        ["residual mass", "未计算", "B=.8333,C=.1667", "只保留 Target 超出 Draft 的部分"],
        ["Target argmax", "B", "仍是 B", "但不是拒绝后确定输出"],
        ["bonus", "未决", "C (u=.90)", "从 residual CDF 随机抽一个"]
      ],
      wfb: [
        ["为什么", "接受+residual 组合保证最终边界 token 的边际分布仍等于 Target P_T。", "main"],
        ["失败", "拒绝后取 argmax B 会把 B 的概率提升到 1，破坏 Target 原始分布。", "fail"],
        ["边界", "当请求处于 greedy_mask 时走 greedy 比较；mixed batch 按请求选择不同接受器。", "guard"]
      ],
      evidence: [
        E("dspark_accept.py:80–174", "sampling accept ratio 与分支准备", "accepted/rejected positions", "reject sampling"),
        E("reject_sampling.py:48–160", "positive residual、归一化与随机采样", "single replacement token", "AcceptSampling output"),
        E("dspark_verify.py:679–722", "按 greedy mask 选择/合并结果", "per-request correct_len/bonus", "FinalizeAcceptLens")
      ],
      faq: [
        ["多个 residual token 都能被接受吗？", "不是‘多个同时接受’；它们只是 residual 分布中的多个候选，最终随机返回一个。"],
        ["为什么 residual 不是 p 本身？", "draft 已经通过接受步骤贡献了与 Target 重叠的概率质量；拒绝后只需补 p 超出 q 的部分。"],
        ["temperature/top-k/top-p 与 sampling verify 什么关系？", "它们先塑造实际 proposal q 和 Target p；verify 必须使用采样时真实分布，而不是未截断/未温度缩放的 logits。"],
        ["greedy 也执行 residual 吗？", "不执行。greedy 验证按离散 token 匹配推进，拒绝边界直接使用 Target greedy prediction。"]
      ]
    },

    "07": {
      conclusion: "Target hidden commit 写的是已接受输入位置 [X,a] 的 hidden→Draft K/V；用户输出是 [a,Y]；Y 没有本轮 hidden/KV，却作为 Y@22 写进下一轮 DFlashDraftInputV2.bonus_tokens。",
      scope: "两轮滚动锚点 · hidden/KV commit",
      intuition: [
        ["输入 rail 与预测 rail 错一位", "hidden(X) 产生对 a 的预测，hidden(a) 产生对 Y 的预测；用户输出对应 next-token rail。", "guard"],
        ["只提交有 hidden 的输入", "本轮 correct_len=1，所以可提交输入侧 [X,a]；Y 只是 bonus，尚未作为输入。", "main"],
        ["长度隐藏 suffix", "未接受 suffix 不写、不清零；new_seq_lens 让它们保持逻辑不可见。", "data"]
      ],
      objects: [
        O("hidden_strided", "Tensor[B,W,H] backing", "Target 每候选输入位置的 auxiliary hidden", "TargetVerifyExecutor", "Target FULL hidden", "accepted prefix slices", "commit_hidden", "data"),
        O("accept : AcceptOuts", "AcceptOuts", "给出 correct_len/commit_lens/new_seq_lens", "accept_and_finalize", "Target/Draft decision", "commit bounds", "TargetHiddenKvInjector", "main"),
        O("self._kv_injector", "TargetHiddenKvInjector", "hidden→projection→Draft K/V prefix-valid writes", "DSparkWorkerV2", "[X,a] hidden + cache loc", "Draft K/V@20,21", "next Draft forward", "cuda"),
        O("next_draft_input", "DFlashDraftInputV2", "保存 bonus Y@22 与新长度", "make_next_draft_input", "AcceptOuts", "bonus_tokens=[Y]", "next Scheduler round", "guard"),
        O("batch_result", "GenerationBatchResult", "把输出/长度/spec_info 交 Scheduler", "DSparkWorkerV2", "accept + next input", "durable state", "Scheduler.process_batch_result", "main")
      ],
      frames: [
        F("Target 输入", "positions 20/21", "有效输入侧是 X@20 与 a@21；它们各自有 Target auxiliary hidden。", [["X@20", "data"], ["a@21", "main"], ["b@22", "off"]]),
        F("本轮提交", "用户输出 [a,Y]", "hidden(X) 支持接受 a；hidden(a) 产生边界 bonus Y。Draft KV 写 X/a，对外输出 a/Y。", [["KV(X)", "main"], ["KV(a)", "main"], ["Y output", "guard"]]),
        F("下一轮", "Y@22 成为 anchor", "new_seq_lens=[22] 把 suffix 隐藏；Y 作为真实 Draft 输入进入下一轮，届时才产生自己的 hidden/KV。", [["Y@22", "main"], ["KV(Y) next", "cuda"]])
      ],
      delta: [
        ["Draft KV visible prefix", "到 position 19", "新增 X@20,a@21", "只写 accepted input hidden"],
        ["用户输出", "[]", "[a,Y]", "draft accepted token + Target bonus"],
        ["seq_lens", "[20]", "[22]", "commit_lens=2"],
        ["next bonus_tokens", "[X]", "[Y]", "Y@22 滚动为下一轮 anchor"]
      ],
      wfb: [
        ["为什么", "Draft 下一轮需要已提交前缀的 K/V，但 bonus 自己尚无 hidden；滚动 anchor 恰好在下一轮补算它。", "main"],
        ["失败", "若把 Y 当本轮 hidden 写 KV，只能错用 hidden(a) 或未初始化 suffix，导致位置/语义错位。", "fail"],
        ["边界", "suffix 不是写零；物理 backing 可保留旧值，new_seq_lens/cache 可见边界阻止读取。", "guard"]
      ],
      evidence: [
        E("dspark_worker_v2.py:627–700", "accept 后 commit_hidden、构造 next input 与 batch result", "next DFlashDraftInputV2", "Scheduler"),
        E("dspark_verify.py:287–318 · commit_hidden", "按 accepted input prefix 选择 hidden/cache loc", "injector inputs", "TargetHiddenKvInjector"),
        E("dspark_kv_inject.py:31–157", "projection 并写 Draft 各层 K/V", "prefix-valid cache", "next Draft forward"),
        E("dspark_draft.py:158–163 · make_next_draft_input", "把 bonus/new_seq_lens 封装为跨轮状态", "Y@22", "next decode")
      ],
      faq: [
        ["为什么提交 [X,a]，用户却看到 [a,Y]？", "hidden/KV rail按本轮输入位置；用户输出 rail按 next-token prediction。两条 rail 天生错一位。"],
        ["Y 的 position 为什么是 22？", "旧长度20，提交 a 与 Y 共2个 token，新长度22；Y 是下一轮在 position22 处理的输入。"],
        ["不清零 suffix 会污染下一轮吗？", "只要长度/cache metadata 正确，suffix 保持不可见；错误是越界读取，而不是 backing 中存在旧值。"],
        ["overlap 与 non-overlap 最终提交有何不同？", "语义相同；non-overlap 直接提交 batch fields，overlap 先 publish/resolve，再由 Scheduler result processor 落 durable 状态。"]
      ]
    },

    "08": {
      conclusion: "folded 是把 proposal/accept/可选 commit 的设备操作覆盖进既有 CUDA Graph，不是新的解码算法；每个 guard 失败都回到普通主链，并在共同尾部重新汇合。",
      scope: "NVIDIA CUDA only · execution placement",
      intuition: [
        ["绿色主链是语义基准", "普通 Draft→Target→accept→commit 定义正确结果；紫色 folded 只替换某些 kernel 的执行位置。", "main"],
        ["三个 folded guard 独立", "proposal folded、accept folded、commit folded 不是一个总开关，后者依赖更严格。", "guard"],
        ["执行过不等于采用", "graph 内 epilogue 可能写了 buffer；Worker 只有 folded_accept=true 才读取并信任它。", "cuda"]
      ],
      objects: [
        O("self._draft_sampler", "Draft graph sampler", "可选在 Draft graph 内产生 greedy tokens", "DSparkWorkerV2", "Draft logits", "folded proposal buffer", "DraftBlockProposer", "cuda"),
        O("epilogue", "DsparkVerifyEpilogue", "在 Target decode graph 内 scatter/accept/可选 commit", "TargetVerifyExecutor", "static graph buffers", "AcceptOuts buffers", "DSparkWorkerV2", "cuda"),
        O("fold_eligible", "bool guard", "检查 epilogue、compact、greedy 等前置", "DSparkWorkerV2", "mode/runtime", "permission to arm", "run_compact", "guard"),
        O("folded_accept", "bool guard", "要求 fold_eligible + compact + graph hit", "DSparkWorkerV2", "TargetVerifyResult.can_run_cuda_graph", "read_accept decision", "accept_and_finalize", "guard"),
        O("folded_commit", "bool guard", "要求 folded_accept 且 epilogue.folds_commit", "DSparkWorkerV2", "accept guard", "skip/execute Python commit", "common tail", "main")
      ],
      frames: [
        F("普通", "语义主链", "Draft proposal、Target forward、accept、hidden commit 分别由 Python 发起。", [["proposal", "main"], ["verify", "main"], ["accept", "main"], ["commit", "main"]]),
        F("覆盖", "Target graph epilogue", "compact+greedy+CUDA graph 命中时，epilogue 可在 replay 内完成 scatter/accept；可选继续 commit。", [["graph replay", "cuda"], ["accept buf", "cuda"], ["KV write", "cuda"]]),
        F("回退", "false 分支重新汇流", "任一 guard false 即执行对应普通 Python/Triton 路径，最终仍返回相同 AcceptOuts/GenerationBatchResult 合同。", [["fallback", "guard"], ["common tail", "main"]])
      ],
      delta: [
        ["算法语义", "普通主链", "不变", "folded 只改变执行位置"],
        ["Python launch", "逐 kernel 发起", "部分被 graph replay 覆盖", "减少 CPU launch/同步开销"],
        ["accept source", "普通 accept kernels", "guard 真时 epilogue buffer", "Worker 决定是否采用"],
        ["commit source", "commit_hidden", "可选 graph 内 commit", "folded_commit false 必须回退"]
      ],
      wfb: [
        ["为什么", "把 shape 稳定的尾处理并入 replay，可减少 Python launch 与中间同步，不改变接受概率或长度合同。", "main"],
        ["失败", "把‘epilogue 被捕获’误当‘本轮采用’，会在 folded_accept=false 时读取未武装/不匹配的 buffer。", "fail"],
        ["边界", "COMPACT、CUDA Graph、folded 是不同条件；当前 DsparkVerifyEpilogue 仅支持 NVIDIA CUDA。", "guard"]
      ],
      evidence: [
        E("dspark_worker_v2.py:570–659", "计算 run_compact/fold_eligible/folded_accept/folded_commit", "fallback/rejoin", "GenerationBatchResult"),
        E("dspark_draft.py:121–155,268–336", "Draft folded proposal guard 与 buffer 读取", "DraftProposal", "planner"),
        E("dspark_verify.py:453–655 · DsparkVerifyEpilogue", "graph hook、buffer、accept、可选 commit", "epilogue outputs", "Worker")
      ],
      faq: [
        ["folded 是第四种 speculative decoding 算法吗？", "不是；它是现有 dSpark 语义主链的 CUDA Graph 执行位置优化。"],
        ["COMPACT 一定 folded 吗？", "不一定。还需 CUDA graph 命中、全贪心、epilogue 可用等 guard。"],
        ["graph 内执行了 accept，Worker 一定读它吗？", "不一定；只有 folded_accept=true 才采用对应 buffer，否则普通 accept 路径给出权威结果。"],
        ["folded_commit=false 会丢 KV 写入吗？", "不会；Worker 会回退调用 commit_hidden，然后与 folded 路径汇入共同尾部。"]
      ]
    },

    "08A": {
      conclusion: "DsparkVerifyEpilogue 的输出 buffer 必须在非 capture 的构造/warm-up 阶段分配；每轮 begin_step 写入 verify_lens/inject_gate，replay 只操作静态地址，read_accept 再按当轮 guard 读取。",
      scope: "warm-up → capture → N/N+1 replay",
      intuition: [
        ["capture 记录地址与操作", "CUDA Graph 不是每轮重新解释 Python 对象；输入、输出和 metadata 必须映射到稳定 buffer。", "cuda"],
        ["warm-up 完成惰性分配", "_ensure_out() 在 capture 前触发，避免捕获期间首次创建 strided output。", "main"],
        ["begin_step 是轮次写屏障", "静态 buffer 地址不变，但 verify_lens/inject_gate 的内容每轮必须覆盖。", "guard"]
      ],
      objects: [
        O("epilogue", "DsparkVerifyEpilogue", "持有静态 lens/gate/output buffers", "TargetVerifyExecutor", "max graph shapes", "stable addresses", "decode graph runner", "cuda"),
        O("runner", "DecodeCudaGraphRunner", "warm-up、capture 与 replay", "ModelRunner", "ForwardBatch static buffers", "Target outputs", "epilogue.capture_hook", "cuda"),
        O("verify_lens_buf", "static Tensor", "每轮 compact lens 输入", "epilogue", "begin_step copy", "graph-visible lens", "scatter/accept", "data"),
        O("inject_gate_buf", "static Tensor/bool", "控制图内 KV commit/accept 采用", "epilogue", "armed flag", "graph branch input", "epilogue ops", "guard"),
        O("accept buffers", "correct_len/bonus/cap_trim buffers", "图内结果 backing", "epilogue", "replay outputs", "AcceptOuts view", "Worker.read_accept", "main")
      ],
      frames: [
        F("构造/预热", "capture 前分配", "构造核心 static buffers；warm-up 调 _ensure_out() 补齐 strided output backing。", [["alloc", "main"], ["_ensure_out", "main"], ["no capture", "guard"]]),
        F("Capture", "只记录既有地址操作", "capture_hook 连接 runner output 与 epilogue scatter/accept/commit 操作，不做首次动态分配。", [["graph ops", "cuda"], ["stable ptr", "data"]]),
        F("Replay N/N+1", "begin_step → replay → read", "每轮先覆盖 lens/gate；缺 begin_step 时 N+1 会沿用 N 的 verify_lens/inject_gate。", [["begin_step", "main"], ["replay", "cuda"], ["read_accept", "main"]])
      ],
      delta: [
        ["buffer address", "构造后固定", "replay 中不变", "CUDA Graph 地址稳定要求"],
        ["verify_lens_buf content", "N 的 lens", "begin_step 后 N+1 lens", "每轮必须覆盖"],
        ["inject_gate", "N 的 armed", "N+1 的 armed", "决定图内结果是否可信/是否 commit"],
        ["accept buffer authority", "图中可能已写", "folded_accept=true 才采用", "执行与采用是两层 guard"]
      ],
      wfb: [
        ["为什么", "静态地址让 CUDA Graph 可重放；per-step copy 让同一图承载不同请求长度与 guard。", "main"],
        ["失败", "省略 begin_step 最先造成上一轮 verify_lens/inject_gate 被复用，scatter、accept 或 KV 写入范围错误。", "fail"],
        ["边界", "本页是静态源码路径核验，不等价于 GPU capture/replay 运行时已验证。", "guard"]
      ],
      evidence: [
        E("dspark_verify.py:453–655 · DsparkVerifyEpilogue", "构造 buffers、capture_hook、begin_step、read_accept、_ensure_out", "static epilogue lifecycle", "Worker"),
        E("decode_cuda_graph_runner.py:920–989,1072–1288", "warm-up/capture/replay 与 load_batch", "static Target graph execution", "epilogue hook"),
        E("dspark_worker_v2.py:570–659", "每轮 arm epilogue 并决定是否采用 accept/commit", "fallback/rejoin", "GenerationBatchResult")
      ],
      faq: [
        ["_ensure_out 为什么不能在 capture 内首次执行？", "首次分配会引入动态地址/allocator 行为；warm-up 先建立稳定 backing，capture 只记录对既有 buffer 的操作。"],
        ["静态 buffer 是否意味着 lens 每轮相同？", "地址相同，内容不同；begin_step 每轮把当前 verify_lens/armed 写入同一地址。"],
        ["缺 begin_step 会读到什么？", "最直接是上一轮 N 的 verify_lens 与 inject_gate，导致 N+1 的图内边界错误。"],
        ["folded_accept=false 为什么图还可能执行 epilogue？", "capture/replay 的操作集合是静态的；Worker 的采用 guard 是更高层权威边界，可忽略图内 accept buffer并走 fallback。"]
      ]
    }
  };

  function esc(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function section(className, id, kicker, title, intro = "") {
    const node = document.createElement("section");
    node.className = `section ${className}`;
    node.id = id;
    node.innerHTML = `<div class="section__head"><p class="section__kicker">${kicker}</p><h2>${title}</h2>${intro ? `<p class="section__intro">${intro}</p>` : ""}</div>`;
    return node;
  }

  function tokenMarkup(tokens) {
    return tokens.map(([label, kind]) => `<span class="example-token" data-kind="${kind || "data"}">${esc(label)}</span>`).join("");
  }

  function mountLesson() {
    const data = lessons[document.body.dataset.page];
    const main = document.querySelector("[data-dspark-main]");
    const hero = main?.querySelector(".hero");
    if (!data || !main || !hero) return;

    const sourceTarget = "lesson-source-main";

    const outline = document.createElement("nav");
    outline.className = "lesson-outline";
    outline.setAttribute("aria-label", "本页九段教学结构");
    outline.innerHTML = [
      ["lesson-conclusion", "01 直接结论", "main"], ["lesson-intuition", "02 前置直觉", "data"],
      ["lesson-objects", "03 组件/对象", "data"], ["lesson-example", "04 具体样例", "main"],
      [sourceTarget, "05 源码主链", "main"], ["lesson-delta", "06 Before/After", "data"],
      ["lesson-wfb", "07 Why/Failure", "fail"], ["lesson-evidence", "08 源码证据", "guard"],
      ["lesson-faq", "09 FAQ/自测", "guard"]
    ].map(([id, label, kind]) => `<a href="#${id}" data-kind="${kind}">${label}</a>`).join("");

    const conclusion = document.createElement("section");
    conclusion.id = "lesson-conclusion";
    conclusion.className = "lesson-conclusion";
    conclusion.innerHTML = `<div class="lesson-conclusion__label">01 · 直接结论</div><p>${data.conclusion}</p><div class="lesson-conclusion__scope"><span data-lesson-route></span><br>${data.scope}<br>源码核对：2026-08-26 · ${HEAD}</div>`;

    const lenses = document.createElement("div");
    lenses.className = "lesson-lenses";
    lenses.setAttribute("aria-label", "选择阅读视角");
    lenses.innerHTML = `<b>阅读视角</b>${[["all","全部"],["object","只看对象"],["state","只看状态"],["failure","只看失败边界"],["evidence","只看证据"]].map(([value, label]) => `<button type="button" data-learning-choice="${value}" aria-pressed="${value === "all"}">${label}</button>`).join("")}`;

    const primer = section("lesson-primer", "lesson-intuition", "02 · 前置直觉", "先建立三个不会错位的心智支点", "它们不是摘要，而是阅读后续图、代码与状态表时的坐标系。");
    primer.insertAdjacentHTML("beforeend", `<div class="intuition-grid">${data.intuition.map(([title, text, kind]) => `<article class="intuition-card" data-kind="${kind}"><b>${title}</b><p>${text}</p></article>`).join("")}</div>`);

    const objects = section("lesson-object-map", "lesson-objects", "03 · 组件 / 对象地图", "点击对象，查看所有权、输入、输出和下一消费者", "图中的箭头表示本页主讲解顺序；所有权与运行时调用身份都使用当前源码实例名。");
    objects.insertAdjacentHTML("beforeend", `<div class="object-explorer"><div class="object-map">${data.objects.map((item, index) => `<button type="button" class="object-map__node" data-kind="${item.kind}" data-object-index="${index}" aria-pressed="${index === 0}"><span class="instance">${item.instance}</span><span class="class-name">${item.type}</span><span class="role">${item.role}</span></button>`).join("")}</div><aside class="object-inspector" aria-live="polite"><b>当前对象</b><div class="object-inspector__name"></div><dl><dt>所有者</dt><dd data-object-field="owner"></dd><dt>输入</dt><dd data-object-field="input"></dd><dt>输出/写入</dt><dd data-object-field="output"></dd><dt>下一消费者</dt><dd data-object-field="next"></dd></dl></aside></div>`);

    const example = section("lesson-example", "lesson-example", "04 · 具体样例", "用 Before → 执行中 → After 走一遍真实对象状态", "三步不是装饰动画；每一步都改变对象、张量或可见边界。");
    example.insertAdjacentHTML("beforeend", `<div class="example-lab"><div class="example-stepper">${data.frames.map((frame, index) => `<button type="button" data-example-index="${index}" aria-pressed="${index === 0}">${frame.label}</button>`).join("")}</div><div class="example-stage" aria-live="polite"><div class="example-stage__meta"></div><h3></h3><p></p><div class="example-stage__tokens"></div></div></div>`);

    hero.after(outline, conclusion, lenses, primer, objects, example);

    const delta = section("lesson-delta", "lesson-delta", "06 · Before / After 状态", "精确到字段：什么变了，什么没有变", "把临时视图、物理 backing、有效长度和跨轮 durable 状态分开看。");
    delta.insertAdjacentHTML("beforeend", `<div class="lesson-table-scroll"><table class="state-delta-table"><thead><tr><th>对象 / 字段</th><th>Before</th><th>After</th><th>为什么在这里改变</th></tr></thead><tbody>${data.delta.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);

    const wfb = section("lesson-wfb", "lesson-wfb", "07 · Why / Failure / Boundary", "为什么必须这样；省略后最先坏在哪里；结论到哪里为止");
    wfb.insertAdjacentHTML("beforeend", `<div class="wfb-grid">${data.wfb.map(([title, text, kind]) => `<article class="wfb-card" data-kind="${kind}"><b>${title}</b><p>${text}</p></article>`).join("")}</div>`);

    const evidence = section("lesson-evidence", "lesson-evidence", "08 · 当前源码证据", "沿 action → changed state → next consumer 复核", `源码核对基线：2026-08-26，<code>${HEAD}</code>。2026-09-09 整理导航与内容归属，保留原案例和证据范围；静态源码与页面交互不代表 GPU 运行时验证。`);
    evidence.insertAdjacentHTML("beforeend", `<div class="lesson-table-scroll"><table class="evidence-table"><thead><tr><th>当前源码锚点</th><th>执行动作</th><th>改变/产出</th><th>下一消费者</th></tr></thead><tbody>${data.evidence.map((item) => `<tr><td><code>${item.anchor}</code></td><td>${item.action}</td><td>${item.state}</td><td>${item.next}</td></tr>`).join("")}</tbody></table></div>`);

    const faq = section("lesson-faq", "lesson-faq", "09 · FAQ / 自测", "先回答，再展开核对", "展开过的问题会计入本页进度；答案强调能改变结论的边界。");
    faq.insertAdjacentHTML("beforeend", `<div class="faq-grid">${data.faq.map(([q, a]) => `<details class="faq-card"><summary>${q}</summary><p>${a}</p></details>`).join("")}</div><div class="faq-progress" aria-live="polite"></div>`);

    const tailAnchor = main.querySelector(":scope > .pager") || main.querySelector(":scope > footer");
    if (tailAnchor) tailAnchor.before(delta, wfb, evidence, faq);
    else main.append(delta, wfb, evidence, faq);

    function updateObject(index) {
      const item = data.objects[index];
      objects.querySelectorAll("[data-object-index]").forEach((node) => node.setAttribute("aria-pressed", String(Number(node.dataset.objectIndex) === index)));
      objects.querySelector(".object-inspector__name").textContent = `${item.instance} · ${item.type}`;
      ["owner", "input", "output", "next"].forEach((field) => { objects.querySelector(`[data-object-field="${field}"]`).textContent = item[field]; });
    }

    function updateExample(index) {
      const frame = data.frames[index];
      example.querySelectorAll("[data-example-index]").forEach((node) => node.setAttribute("aria-pressed", String(Number(node.dataset.exampleIndex) === index)));
      example.querySelector(".example-stage__meta").textContent = `${index + 1} / ${data.frames.length} · ${frame.label}`;
      example.querySelector("h3").textContent = frame.title;
      example.querySelector(".example-stage > p").textContent = frame.text;
      example.querySelector(".example-stage__tokens").innerHTML = tokenMarkup(frame.tokens);
    }

    function updateFaqProgress() {
      const cards = [...faq.querySelectorAll("details")];
      cards.forEach((card) => { card.dataset.opened = String(card.open); });
      faq.querySelector(".faq-progress").textContent = `已核对 ${cards.filter((card) => card.open).length} / ${cards.length} 个问题`;
    }

    function updateRoute() {
      const route = window.DSparkSite?.route || "compact-sampling";
      const label = { "static-greedy": "STATIC · 贪心", "compact-sampling": "COMPACT · 随机采样", "folded-greedy": "FOLDED · 贪心" }[route] || route;
      conclusion.querySelector("[data-lesson-route]").textContent = `当前路线：${label}`;
    }

    objects.addEventListener("click", (event) => {
      const button = event.target.closest("[data-object-index]");
      if (button) updateObject(Number(button.dataset.objectIndex));
    });
    example.addEventListener("click", (event) => {
      const button = event.target.closest("[data-example-index]");
      if (button) updateExample(Number(button.dataset.exampleIndex));
    });
    lenses.addEventListener("click", (event) => {
      const button = event.target.closest("[data-learning-choice]");
      if (!button) return;
      const choice = button.dataset.learningChoice;
      lenses.querySelectorAll("[data-learning-choice]").forEach((node) => node.setAttribute("aria-pressed", String(node === button)));
      if (choice === "all") delete main.dataset.learningLens;
      else main.dataset.learningLens = choice;
    });
    faq.addEventListener("toggle", updateFaqProgress, true);
    addEventListener("dspark:route", updateRoute);

    updateObject(0);
    updateExample(0);
    updateFaqProgress();
    updateRoute();
  }

  // 九段式课程骨架先挂载；dspark-source-studio.js 随后把第 05 章插入
  // 04/06 之间，并只将原始大图移动到编号补图区。
  window.DSparkLessonData = lessons;
  mountLesson();
})();
