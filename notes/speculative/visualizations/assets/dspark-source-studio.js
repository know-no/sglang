(() => {
  "use strict";

  const lessons = window.DSparkLessonData;
  if (!lessons) return;

  const T = (name, shape, dtype, value, note = "") => ({ name, shape, dtype, value, note });
  const C = (value, kind = "read", note = "") => ({ value, kind, note });
  const R = (name, note, cells) => ({ name, note, cells });
  const S = (key, label, title, source, call, inputs, equation, calc, outputs, rails, ledger, why, failure, boundary) => ({
    key, label, title, source, call, inputs, equation, calc, outputs, rails, ledger, why, failure, boundary
  });

  const draftConstruct = S(
    "draft-block", "构造 Draft 输入", "anchor 只写第 0 列；其余列是 MASK，不是已经采样出的 draft",
    "python/sglang/srt/speculative/dspark_components/dspark_draft.py:361–412",
    "DraftBlockProposer._run_forward()",
    [
      T("draft_input.bonus_tokens", "[B]=[2]", "int64 · CUDA", "[X0, X1]", "上一轮 bonus；本轮真实 anchor"),
      T("batch.seq_lens", "[2]", "int64 · CUDA", "[22, 10]", "进入 block 前已提交长度"),
      T("mask_token_id", "scalar", "int", "M")
    ],
    "draft_block_ids = full((B,G), M)\ndraft_block_ids[:,0] = bonus_tokens\npositions = seq_lens[:,None] + arange(G)",
    ["G=4，所以每请求只构造 4 个 Draft Transformer 输入行", "req0 positions = 22 + [0,1,2,3] = [22,23,24,25]", "seq_lens_cpu = [22,10] + 4 = [26,14]；batch.seq_lens 本身不原地改变"],
    [
      T("draft_block_ids", "[2,4]", "int64 · CUDA", "[[X0,M,M,M],\n [X1,M,M,M]]"),
      T("draft_positions", "[8]", "int64 · CUDA", "[22,23,24,25, 10,11,12,13]"),
      T("draft_seq_lens_cpu", "[2]", "int64 · CPU", "[26,14]", "host planner 的 prefix+G 临时视图")
    ],
    [
      R("req0 Draft 输入", "position 22…25", [C("X0","write","anchor"),C("M","off"),C("M","off"),C("M","off")]),
      R("req1 Draft 输入", "position 10…13", [C("X1","write","anchor"),C("M","off"),C("M","off"),C("M","off")])
    ],
    ["input_ids", "未构造", "[X0,M,M,M,X1,M,M,M]", "轮内 Tensor；不跨轮", "next: Draft ModelRunner"],
    "并行 Transformer 需要一次性获得 G 行固定形状输入；后置 token 的依赖由 Markov head 补，而不是先把 a/b/c 填进输入。",
    "若把 [X0,a,b,c] 当作 Draft Transformer 输入，就把尚未采样的 token 提前写进因果上下文，模型与真实实现完全不同。",
    "符号 X0/X1/M 是教学 token id；shape、索引和写入关系来自当前源码，不是 GPU Tensor dump。"
  );

  const draftForward = S(
    "draft-forward", "一次 Draft Transformer", "8 个输入行一次并行 forward，产生 8 个 hidden；没有为 a→b→c→d 重跑 Transformer",
    "python/sglang/srt/speculative/dspark_components/dspark_draft.py:398–425",
    "draft_out = draft_model_runner.forward(draft_forward_batch)",
    [
      T("ForwardBatch.input_ids", "[B×G]=[8]", "int64 · CUDA", "[X0,M,M,M, X1,M,M,M]"),
      T("ForwardBatch.positions", "[8]", "int64 · CUDA", "[22,23,24,25, 10,11,12,13]"),
      T("ForwardBatch.out_cache_loc", "[8]", "int64 · CUDA", "[L0_22…L0_25, L1_10…L1_13]")
    ],
    "raw_hidden = draft_model_runner.forward(FB).logits_output.hidden_states\ndraft_hidden_3d = raw_hidden.view(B,G,-1)\nbase_logits = compute_base_logits(raw_hidden).view(B,G,V)",
    ["raw_hidden rows = len(input_ids) = 8", "reshape 只改变视图：[8,Hd] → [2,4,Hd]", "shared LM head 对每个 hidden 投影到 vocab 维；得到 [2,4,V]"],
    [
      T("raw_hidden", "[8,Hd]", "model dtype · CUDA", "[hX0,hM01,hM02,hM03, hX1,hM11,hM12,hM13]"),
      T("draft_hidden_3d", "[2,4,Hd]", "view · CUDA", "同一 backing 的三维视图"),
      T("base_logits", "[2,4,V]", "model dtype · CUDA", "L[b,step,vocab]")
    ],
    [
      R("Transformer 输入", "8 行同时进入", [C("X0","read"),C("M","read"),C("M","read"),C("M","read")]),
      R("Transformer hidden", "一次 forward 同时返回", [C("h0","cuda"),C("h1","cuda"),C("h2","cuda"),C("h3","cuda")])
    ],
    ["raw_hidden", "不存在", "[B×G,Hd]", "设备 Tensor", "next: base logits / Markov"],
    "固定 G 行让 Draft 主干并行；串行依赖只落在较小的 Markov logits 修正与采样上。",
    "若为每个 draft token 重跑一次 Transformer，会把当前实现的 1 次 G 行 forward 错画成 G 次自回归 forward。",
    "Hd/V 随模型配置变化；页面展示轴语义而不虚构具体 hidden 数值。"
  );

  const markovSample = S(
    "markov", "Markov 串行采样", "每一步读取上一步 token，修正对应 hidden 的 base logits，再采出当前 draft",
    "python/sglang/srt/models/dspark.py:34–64; dspark_draft.py:177–238",
    "draft_tokens, corrected_logits = markov_head.sample_block(...) ",
    [
      T("base_logits", "[2,4,V]", "model dtype · CUDA", "L[:,0:4,:]"),
      T("first_prev_tokens", "[2]", "int64 · CUDA", "[X0,X1]"),
      T("draft_hidden", "[2,4,Hd]", "model dtype · CUDA", "H[:,0:4,:]")
    ],
    "prev = anchor\nfor i in 0..3:\n  step_logits = head.apply_step_logits(L[:,i,:], token_ids=prev, hidden_states=H[:,i,:])\n  token[:,i] = sampler(step_logits)\n  prev = token[:,i]",
    ["i=0：prev=[X0,X1] → sample=[a,e]", "i=1：prev=[a,e] → sample=[b,f]", "i=2：prev=[b,f] → sample=[c,g]", "i=3：prev=[c,g] → sample=[d,h]"],
    [
      T("draft_tokens", "[2,4]", "int64 · CUDA", "[[a,b,c,d],\n [e,f,g,h]]"),
      T("corrected_logits", "[2,4,V]", "model dtype · CUDA", "真实 proposal q 的 logits"),
      T("temperatures / greedy_mask", "[2] / [2]", "float32 / bool", "用于 sampling verify 重建真实 P_D")
    ],
    [
      R("prev token", "串行依赖", [C("X0","read"),C("a","read"),C("b","read"),C("c","read")]),
      R("sampled draft", "每步只产生一个", [C("a","write"),C("b","write"),C("c","write"),C("d","write")]),
      R("对应 hidden", "早已并行产生", [C("h0","cuda"),C("h1","cuda"),C("h2","cuda"),C("h3","cuda")])
    ],
    ["proposal token chain", "只有 anchor", "[anchor,a,b,c,d]", "draft_tokens 是轮内候选", "next: confidence / Target candidates"],
    "MASK hidden 给每个 slot 提供位置/前缀特征；Markov bias 用 prev token 补上后置 token 的串行依赖。",
    "若丢掉 corrected_logits，只保留 token id，sampling verify 无法恢复当时真实 proposal 分布 q，拒绝采样会失真。",
    "greedy 的 sampler 是 argmax；sampling 的 sampler 使用 temperature 后 softmax/multinomial，二者共用同一串行循环。"
  );

  const currentSchedule = S(
    "current-schedule", "当前 survival 分配", "历史预算只给出 budget=2；当前 confidence 再决定这 2 个额外槽给谁",
    "python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py:49–115",
    "verify_lens = ScheduleVerifyLensTopk.execute(confidence=C_t, budget=2, cfg=cfg)",
    [
      T("current confidence C_t", "[2,4]", "float32 · CUDA", "req0 [.95,.90,.70,.20]\nreq1 [.99,.95,.92,.85]"),
      T("budget", "Python int", "host scalar", "2", "不包含 mandatory anchor floor")
    ],
    "survival = cumprod(C_t, dim=1)\norder = argsort(survival.flatten(), descending=True)\nselected_extra.scatter_add_(request_id[top2], 1)\nverify_lens = clamp(min_len + selected_extra)",
    ["req0 survival=[.95000,.85500,.59850,.11970]", "req1 survival=[.99000,.94050,.86526,.73547]", "全局 top-2：req1/pos0=.99，req0/pos0=.95", "selected_extra=[1,1]；floor=[1,1] → verify_lens=[2,2]"],
    [
      T("current survival", "[2,4]", "float32 · CUDA", "[[.950,.855,.599,.120],\n [.990,.941,.865,.735]]"),
      T("selected_extra", "[2]", "int64 · CUDA", "[1,1]"),
      T("verify_lens", "[2]", "int32 · CUDA", "[2,2]", "每个 2 都是 anchor + 1 draft")
    ],
    [
      R("req0 survival", "top-k 选择", [C(".950","write","选中"),C(".855","off"),C(".599","off"),C(".120","off")]),
      R("req1 survival", "top-k 选择", [C(".990","write","选中"),C(".941","off"),C(".865","off"),C(".735","off")])
    ],
    ["verify_lens", "未产生", "[2,2]", "本轮 CUDA int32 Tensor", "next: RaggedVerifyLayout"],
    "全局 top-k 把有限 Target 计算分给预计最可能继续被接受的请求/深度；mandatory floor 保证每请求至少验证 anchor。",
    "把 budget=2 直接解释成每请求验证 2 个 draft，会漏掉 batch-global 分配和 anchor floor 两层语义。",
    "一般只保证 sum(verify_lens-floor) ≤ budget；survival_eps 过滤后 budget 不一定用满。"
  );

  const compactWindow = S(
    "compact-window", "压紧 Target 输入", "从完整 2×5 候选矩阵中只 gather 两个请求的连续前缀，再补 1 个 graph-tier padding",
    "python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py:76–127",
    "ragged_window = BuildRaggedVerifyWindow.execute(...) ",
    [
      T("verify_ids_2d", "[2,5]", "int64 · CUDA", "[[X0,a,b,c,d],\n [X1,e,f,g,h]]"),
      T("verify_lens", "[2]", "int32 · CUDA", "[2,2]"),
      T("prefix_lens", "[2]", "int64 · CUDA", "[22,10]")
    ],
    "qo_indptr = prefix_sum([0] + verify_lens) = [0,2,4]\nlogical_rows = sum(verify_lens) = 4\ngraph_num_tokens = round_up_grid(4) = 5\n(req,within) = compact_row_index(verify_lens, 5)",
    ["row0/1 → req0 within0/1 → [X0,a]", "row2/3 → req1 within0/1 → [X1,e]", "row4 → synthetic padding；position/cache_loc 都写 0"],
    [
      T("ragged_window.verify_ids", "[5]", "int64 · CUDA", "[X0,a,X1,e,0]"),
      T("ragged_window.positions", "[5]", "int64 · CUDA", "[22,23,10,11,0]"),
      T("ragged_window.verify_cache_loc", "[5]", "int64 · CUDA", "[L0_22,L0_23,L1_10,L1_11,0]")
    ],
    [
      R("完整 candidates", "req0", [C("X0","read"),C("a","read"),C("b","off"),C("c","off"),C("d","off")]),
      R("完整 candidates", "req1", [C("X1","read"),C("e","read"),C("f","off"),C("g","off"),C("h","off")]),
      R("compact flat rail", "qo=[0,2,4]", [C("X0","write","r0"),C("a","write","r1"),C("X1","write","r2"),C("e","write","r3"),C("0","off","padding")])
    ],
    ["Target token rows", "完整矩形 10", "逻辑 4 / 物理 5", "本轮 compact backing", "next: DFlashVerifyInput"],
    "compact 节省的是 Target 真实候选行；graph tier 仍要求固定物理行数，所以逻辑行和物理行必须分开看。",
    "若把 padding row 当成 req1 的第三个真实 token，就会错误推进 attention/KV/accept 边界。",
    "不存在一个单独的 compact_flat_buffer；token、position、KV-location 是三条严格对齐的 Tensor rail。"
  );

  const targetForward = S(
    "target-forward", "Target 计算与 attention 几何", "ForwardBatch 同时保留真实请求数 B=2 和物理 token 行数 5；attention 用 indptr 解释每行属于谁",
    "python/sglang/srt/speculative/dflash_info.py:53–141; dspark_verify.py:320–387",
    "target_worker.forward_batch_generation(forward_batch=verify_forward_batch, is_verify=True)",
    [
      T("ForwardBatch.input_ids", "[5]", "int64 · CUDA", "[X0,a,X1,e,0]"),
      T("ForwardBatch.seq_lens", "[2]", "int64 · CUDA", "[22,10]", "未加 verify block 的 committed prefix"),
      T("layout.qo_indptr_device", "[3]", "int32 · CUDA", "[0,2,4]")
    ],
    "paged_kernel_lens = prefix_lens + verify_lens = [24,12]\ncum_kv_seq_len[1:] = cumsum([24,12]) = [24,36]\ncu_seqlens_q = [0,2,4]\nmodel.forward(input_ids, positions, forward_batch)",
    ["batch_size=2：真实请求数", "len(input_ids)=5：graph-tier 物理行数", "有效 query rows=qo_indptr[-1]=4", "FULL hidden 让每个实际 Target 输入行同时返回 auxiliary hidden"],
    [
      T("next_token_logits", "[5,V]", "model dtype · CUDA", "rows 0..3 有真实请求语义；row4 为 padding backing"),
      T("hidden_states", "[5,Haux]", "model dtype · CUDA", "与 5 个物理输入行对齐"),
      T("can_run_cuda_graph", "Python bool", "host", "取决于当前 runner/grid 是否命中")
    ],
    [
      R("query row", "由 qo_indptr 分段", [C("r0:req0/X0","cuda"),C("r1:req0/a","cuda"),C("r2:req1/X1","cuda"),C("r3:req1/e","cuda"),C("r4:pad","off")]),
      R("KV 可见长度", "prefix + verify", [C("24","read","req0"),C("12","read","req1"),C("—","off"),C("—","off"),C("0","off")])
    ],
    ["Target outputs", "无", "logits[5,V] + hidden[5,Haux]", "轮内设备 Tensor", "next: scatter / accept"],
    "Target 一次 forward 就能对 anchor+draft 的每个输入位置计算 next-token 分布；attention metadata 保证每行只看到自己的 prefix 与线性 verify 前缀。",
    "只看 batch_size=2 而忽略 input_ids 有 5 行，会误判 kernel 形状；只看 5 行又会把 padding 当真实请求。",
    "CUDA Graph、eager 的调度位置不同，但这里展示的 ForwardBatch 输入/输出合同相同。"
  );

  const scatterOutput = S(
    "scatter", "散布回固定宽度", "公共 accept 接口按 [B,W] 读取，因此 4 个逻辑结果要写回 10 个 strided 槽，未验证列填 0",
    "python/sglang/srt/speculative/dspark_components/dspark_verify.py:387–425",
    "strided = ScatterCompactToStrided.execute(compact, layout, fill_value=0.0, stride=W)",
    [
      T("compact_logits", "[5,V]", "model dtype · CUDA", "[L_X0,L_a,L_X1,L_e,L_pad]"),
      T("compact_hidden", "[5,Haux]", "model dtype · CUDA", "[hX0,ha,hX1,he,hpad]"),
      T("verify_lens", "[2]", "int32 · CUDA", "[2,2]")
    ],
    "dst_row = request_id * W + within\nreq0 rows 0,1 → dst 0,1\nreq1 rows 2,3 → dst 5,6\n其余 dst 2,3,4,7,8,9 = 0",
    ["compact row0 X0 → strided[0]", "compact row1 a  → strided[1]", "compact row2 X1 → strided[5]", "compact row3 e  → strided[6]", "compact padding row4 不获得真实 strided 语义"],
    [
      T("logits_output.next_token_logits", "[B×W,V]=[10,V]", "model dtype · CUDA", "req0 [L_X0,L_a,0,0,0]\nreq1 [L_X1,L_e,0,0,0]"),
      T("hidden_strided", "[10,Haux]", "model dtype · CUDA", "req0 [hX0,ha,0,0,0]\nreq1 [hX1,he,0,0,0]")
    ],
    [
      R("compact row", "source", [C("X0","read"),C("a","read"),C("X1","read"),C("e","read"),C("pad","off")]),
      R("req0 strided", "dst 0…4", [C("X0","write"),C("a","write"),C("0","off"),C("0","off"),C("0","off")]),
      R("req1 strided", "dst 5…9", [C("X1","write"),C("e","write"),C("0","off"),C("0","off"),C("0","off")])
    ],
    ["logits layout", "compact [5,V]", "strided [10,V]", "统一 accept 读取形状", "next: AcceptGreedy/Sampling"],
    "scatter 让 compact/non-compact 在 accept 边界共享固定 [B,W] 索引公式；hidden_strided 同时保留给后续 KV commit。",
    "若直接把 compact row index 当作 request*W+column，req1 会从 row2 被误当成 req0 column2。",
    "填 0 的列不是概率分布，也不是已验证 logits；cutoff verify_lens 阻止 accept 读取这些列。"
  );

  const acceptGreedy = S(
    "accept", "连续接受与边界 bonus", "Target 每行预测下一个 token；比较从第一个 draft 开始，第一次不匹配即停止",
    "python/sglang/kernels/ops/speculative/dspark/dspark_accept.py:558–673",
    "correct_len, bonus, cap_trim = AcceptGreedy.execute(candidates, target_logits, cutoff_verify_lens)",
    [
      T("candidates", "[2,5]", "int64 · CUDA", "req0 [X0,a,b,c,d]\nreq1 [X1,e,f,g,h]"),
      T("target_predict", "[2,5]", "int64 · CUDA", "req0 [a,Y0,?,?,?]\nreq1 [Y1,?,?,?,?]"),
      T("cutoff_verify_lens", "[2]", "int32 · CUDA", "[2,2]")
    ],
    "req0: target_predict[col0]=a == draft[col1]=a → accept 1\n      target_predict[col1]=Y0 != draft[col2]=b → stop\nreq1: target_predict[col0]=Y1 != draft[col1]=e → stop\nbonus = target_predict[row, correct_len]",
    ["req0 correct_len=1，bonus=target_predict[0,1]=Y0", "req1 correct_len=0，bonus=target_predict[1,0]=Y1", "比较是连续前缀；后面偶然相等也不能越过第一次拒绝"],
    [
      T("correct_len", "[2]", "int64/int32 · CUDA", "[1,0]", "只数接受的 draft，不含 anchor/bonus"),
      T("bonus", "[2]", "int64 · CUDA", "[Y0,Y1]"),
      T("cap_trim_lens", "[2]", "int32 · CUDA", "[0,0]")
    ],
    [
      R("Target 输入", "col0…4", [C("X0","read"),C("a","read"),C("b","off"),C("c","off"),C("d","off")]),
      R("Target next 预测", "向右错一位", [C("a","write","匹配 d1"),C("Y0","op","bonus"),C("?","off"),C("?","off"),C("?","off")]),
      R("用户候选", "从 d1 开始比", [C("a","write","接受"),C("b","fail","拒绝"),C("c","off"),C("d","off")])
    ],
    ["accept result", "未决", "correct=[1,0], bonus=[Y0,Y1]", "轮内决定", "next: FinalizeAcceptLens"],
    "Target input rail 与 next-token prediction rail 天生错一位；bonus 来自最后接受边界那一行的 Target 分布。",
    "若拒绝 b 后继续比较 c，c 那行 Target hidden 已经看过未接受的 b，不再对应真实生成前缀。",
    "sampling 路径不用 argmax 比较，而是用实际 p/q 做拒绝采样；06A 单独展开数值。"
  );

  const finalizeOutput = S(
    "finalize", "长度与输出 backing", "把接受数量变成正式长度，并只在 correct_len 列覆盖 bonus；固定宽度尾部仍留在 backing 中",
    "python/sglang/kernels/ops/speculative/dspark/dspark_accept.py:718–730; dspark_verify_window.py:810–827",
    "commit_lens = correct_len.int() + 1; new_seq_lens = prefix_lens + commit_lens; out_tokens.scatter_(1, correct_len, bonus)",
    [
      T("correct_len", "[2]", "int", "[1,0]"),
      T("prefix_lens", "[2]", "int64", "[22,10]"),
      T("draft_tokens", "[2,4]", "int64", "req0 [a,b,c,d]\nreq1 [e,f,g,h]")
    ],
    "commit_lens = [1,0] + 1 = [2,1]\nnew_seq_lens = [22,10] + [2,1] = [24,11]\nbacking 先复制 drafts，再在 col=correct_len 写 bonus",
    ["req0 [a,b,c,d,0]，在 col1 写 Y0 → [a,Y0,c,d,0]", "req1 [e,f,g,h,0]，在 col0 写 Y1 → [Y1,f,g,h,0]", "有效输出由 accept_lens=[2,1] 截取，不是消费整个 W 列"],
    [
      T("commit_lens", "[2]", "int32 · CUDA", "[2,1]"),
      T("new_seq_lens", "[2]", "int64 · CUDA", "[24,11]"),
      T("out_tokens backing", "[2,5]", "int64 · CUDA", "req0 [a,Y0,c,d,0]\nreq1 [Y1,f,g,h,0]"),
      T("用户有效输出", "ragged lengths [2,1]", "token ids", "req0 [a,Y0]\nreq1 [Y1]")
    ],
    [
      R("req0 backing", "accept_lens=2", [C("a","write","有效"),C("Y0","write","有效"),C("c","off","尾部"),C("d","off","尾部"),C("0","off")]),
      R("req1 backing", "accept_lens=1", [C("Y1","write","有效"),C("f","off","尾部"),C("g","off"),C("h","off"),C("0","off")])
    ],
    ["正式长度", "prefix=[22,10]", "new=[24,11]", "跨轮 durable", "next: KV commit + Scheduler"],
    "固定 shape 便于 batch/kernel/graph 复用；commit_lens 才是用户输出、长度推进和 KV 写入共同使用的有效边界。",
    "把 out_tokens 整个 [B,W] 返回给用户会泄漏未消费的 c/d/f/g/h backing。",
    "commit_lens 数值等于‘有效输出长度’，但 KV 提交的 token 内容与用户输出错一位。"
  );

  const kvCommit = S(
    "kv-commit", "用 commit mask 写 Draft KV", "Target hidden 与输入行对齐；只把 anchor + accepted drafts 的 hidden 投影为 Draft K/V",
    "python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py:105–157",
    "kv_injector.inject_ragged(hidden_strided, commit_lens, layout, batch)",
    [
      T("hidden_strided", "[2,5,Haux]", "model dtype · CUDA", "req0 [hX0,ha,0,0,0]\nreq1 [hX1,he,0,0,0]"),
      T("commit_lens", "[2]", "int32 · CUDA", "[2,1]"),
      T("positions", "[2,5]", "int64 · CUDA", "req0 [22,23,24,25,26]\nreq1 [10,11,12,13,14]")
    ],
    "committed_mask[b,col] = col < commit_lens[b]\nreq0 mask=[1,1,0,0,0]\nreq1 mask=[1,0,0,0,0]\nDraftModel.write_target_hidden_kv(hidden, positions, cache_locs)",
    ["req0 写 hX0@22、ha@23；不写 hb/hc/hd", "req1 只写 hX1@10；he 对应被拒绝的 e，不提交", "每层实际 K/V 投影由当前 draft_model.write_target_hidden_kv 实现；不是简单复制 hidden"],
    [
      T("Draft KV 新增", "ragged [2,1] positions", "per-layer K/V", "req0 K/V(X0@22), K/V(a@23)\nreq1 K/V(X1@10)"),
      T("未提交 suffix", "物理 backing 可能存在", "不可见", "req0 b/c/d；req1 e/f/g/h", "不写、不清零，由长度保持不可见")
    ],
    [
      R("req0 commit mask", "commit_len=2", [C("X0","write"),C("a","write"),C("b","off"),C("c","off"),C("d","off")]),
      R("req1 commit mask", "commit_len=1", [C("X1","write"),C("e","off"),C("f","off"),C("g","off"),C("h","off")])
    ],
    ["Draft KV visible prefix", "到旧 prefix-1", "新增 anchor + accepted drafts", "持久设备状态", "next: next_draft_input"],
    "下一轮 Draft 不只需要 token id，还需要已确认历史在 Draft 层的 K/V；Target auxiliary hidden 是投影源。",
    "把用户输出 [a,Y0] 当成 KV 写入内容会错位：Y0 尚未作为 Transformer 输入，没有自己的 hidden/K/V。",
    "dense/MLA 具体 K/V 写函数不同；共同合同是 positions/cache_loc + commit_lens 控制 prefix-valid 写入。"
  );

  const nextRound = S(
    "next-round", "形成下一轮输入", "本轮 bonus 没有自己的 KV；它只随 new_seq_lens 跨轮保存，下一轮才写入 Draft block 第 0 列",
    "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:684–700; dspark_draft.py:158–163,375–380",
    "next_draft_input = make_next_draft_input(bonus_tokens=accept.bonus, new_seq_lens=accept.new_seq_lens)",
    [
      T("accept.bonus", "[2]", "int64 · CUDA", "[Y0,Y1]"),
      T("accept.new_seq_lens", "[2]", "int64 · CUDA", "[24,11]"),
      T("Draft KV committed", "req0 +2, req1 +1", "device state", "不包含 K/V(Y0) 或 K/V(Y1)")
    ],
    "DFlashDraftInputV2(bonus_tokens=[Y0,Y1], new_seq_lens=[24,11])\n下一轮：full((B,G),M); ids[:,0]=bonus_tokens\npositions = new_seq_lens[:,None] + arange(G)",
    ["req0 下一轮 Draft 输入 [Y0,M,M,M]，positions [24,25,26,27]", "req1 下一轮 Draft 输入 [Y1,M,M,M]，positions [11,12,13,14]", "Y0/Y1 在下一轮经过 Transformer 后才获得自己的 hidden/KV"],
    [
      T("next_draft_input.bonus_tokens", "[2]", "int64", "[Y0,Y1]"),
      T("next_draft_input.new_seq_lens", "[2]", "int64", "[24,11]"),
      T("next draft_block_ids", "[2,4]", "int64", "[[Y0,M,M,M],\n [Y1,M,M,M]]")
    ],
    [
      R("本轮用户输出", "next-token rail", [C("a","write"),C("Y0","write"),C("—","off"),C("—","off")]),
      R("本轮 Draft KV", "input rail", [C("X0","write"),C("a","write"),C("—","off"),C("—","off")]),
      R("下一轮输入", "rolling anchor", [C("Y0","write"),C("M","off"),C("M","off"),C("M","off")])
    ],
    ["跨轮 carrier", "bonus=[X0,X1]", "bonus=[Y0,Y1]", "DFlashDraftInputV2", "next: DraftBlockProposer"],
    "rolling anchor 补上自回归的一位错位：上轮只预测出的 bonus，在下一轮作为真实输入被计算。",
    "若直接把整个 DFlashVerifyInput 跨轮保存，会携带过期 candidates、positions、layout 与 cache loc。",
    "overlap/non-overlap 的 carrier/提交时点不同，但下一轮重建 [bonus,MASK,…] 的语义相同。"
  );

  const isolationSteps = [
    S("stable", "稳定视图", "ScheduleBatch 进入 isolation 前持有跨轮状态与当前 decode 输入引用",
      "python/sglang/srt/managers/scheduler.py:3326–3367", "snapshot = (batch.input_ids, batch.spec_info, batch.forward_mode, ...)",
      [T("batch.input_ids", "Tensor ref T_old", "int64", "[X0,X1]"),T("batch.spec_info", "object ref S_old", "DFlashDraftInputV2", "bonus=[X0,X1], new_seq_lens=[22,10]"),T("batch.forward_mode", "enum", "CPU", "DECODE")],
      "snapshot 保存字段引用，不复制 Tensor 数据\nT_old 与 S_old 的 backing/对象身份保持不变",
      ["保存的是 Python 引用关系", "此时 batch.seq_lens=[22,10] 是 durable committed state"],
      [T("snapshot.input_ids", "same ref", "Python", "T_old → [X0,X1]"),T("snapshot.spec_info", "same ref", "Python", "S_old" )],
      [R("ScheduleBatch fields", "Before", [C("T_old","read"),C("S_old","read"),C("DECODE","read")])],
      ["对象视图", "DECODE", "已快照", "CPU 临时状态", "next: 临时重绑"],
      "isolation 的目标是同一个 ScheduleBatch 可复用，而不是深拷贝一个新批次。", "不保存旧引用，finally 无法恢复 Scheduler 的稳定视图。", "这里没有复制 GPU Tensor；生命周期仍由已有引用/stream 规则负责。"),
    S("rebind", "临时重绑", "把同一个 ScheduleBatch 的字段指向本轮 verify Tensor/对象",
      "python/sglang/srt/speculative/dflash_info.py:53–77", "batch.input_ids=T_verify; batch.spec_info=verify_input; batch.forward_mode=TARGET_VERIFY",
      [T("T_verify candidates", "[5]", "int64 · CUDA", "[X0,a,X1,e,0]"),T("verify_input.positions", "[5]", "int64 · CUDA", "[22,23,10,11,0]"),T("batch.seq_lens_cpu", "[2]", "int64 · CPU", "临时 [24,12]")],
      "batch.input_ids = T_verify\nbatch.spec_info = DFlashVerifyInput(T_verify, positions, layout)\nbatch.forward_mode = TARGET_VERIFY",
      ["旧 T_old/S_old 仍由 snapshot 持有", "重绑只改变 ScheduleBatch 字段指向，不修改 T_old 内容"],
      [T("batch.input_ids", "ref", "int64", "T_verify → [X0,a,X1,e,0]"),T("batch.spec_info", "object", "DFlashVerifyInput", "本轮 candidates/layout"),T("batch.forward_mode", "enum", "CPU", "TARGET_VERIFY")],
      [R("字段引用", "临时", [C("T_verify","op"),C("DFlashVerifyInput","op"),C("TARGET_VERIFY","op")])],
      ["ScheduleBatch", "DECODE refs", "TARGET_VERIFY refs", "只在 yield 范围内", "next: ForwardBatch.init_new"],
      "ForwardBatch.init_new 必须从正确的 candidates/spec metadata 装配执行输入。", "不重绑会把 [X0,X1] 和旧 DFlashDraftInputV2 当成 Target verify 输入，行数与 attention geometry 全错。", "本轮 verify_input 绑定当前 layout/cache loc，不能成为下一轮 durable spec_info。"),
    S("capture", "ForwardBatch 捕获执行引用", "构造完成后，ForwardBatch 自己持有 T_verify 和临时 host length；后续不再重新读取 ScheduleBatch.input_ids",
      "python/sglang/srt/model_executor/forward_batch_info.py:684–942", "verify_forward_batch = ForwardBatch.init_new(batch, model_runner, capture_hidden_mode=FULL)",
      [T("batch.input_ids", "T_verify [5]", "int64", "[X0,a,X1,e,0]"),T("batch.seq_lens_cpu", "[2]", "int64 CPU", "[24,12]"),T("batch.spec_info", "object", "DFlashVerifyInput", "positions/layout")],
      "ForwardBatch.input_ids ← T_verify reference\nForwardBatch.seq_lens_cpu ← current [24,12] view\nForwardBatch.spec_info ← current verify_input",
      ["这是新的执行容器，不是读取 batch 的惰性代理", "Tensor backing 由 ForwardBatch 引用保活"],
      [T("verify_forward_batch.input_ids", "[5]", "same Tensor ref", "T_verify"),T("verify_forward_batch.seq_lens_cpu", "[2]", "captured value/ref", "[24,12]"),T("verify_forward_batch.spec_info", "object ref", "DFlashVerifyInput", "S_verify")],
      [R("引用所有权", "ForwardBatch", [C("T_verify","write"),C("[24,12]","write"),C("S_verify","write")])],
      ["执行输入", "只在 ScheduleBatch 临时字段", "ForwardBatch 已持有", "保活到 Target forward 完成", "next: restore + Target"],
      "这解释了为何 restore ScheduleBatch 不会让正在运行的 Target 丢输入。", "若 ForwardBatch 只保存对 ScheduleBatch 的惰性 getter，restore 后确实会读回旧数据；当前实现不是这样。", "seq_lens_cpu 的具体复制/引用细节不改变关键事实：ForwardBatch 已经装配完成。"),
    S("restore", "恢复与 durable commit", "finally 先恢复旧字段；Target 完成后只提交 new_seq_lens 与 next_draft_input",
      "python/sglang/srt/managers/scheduler.py:3326–3367,3653–3673", "finally: restore(snapshot); process_batch_result(batch, GenerationBatchResult)",
      [T("snapshot", "refs", "Python", "T_old, S_old, DECODE"),T("GenerationBatchResult.new_seq_lens", "[2]", "int64", "[24,11]"),T("next_draft_input.bonus_tokens", "[2]", "int64", "[Y0,Y1]")],
      "restore: batch.input_ids=T_old; spec_info=S_old; mode=DECODE\ncommit: batch.seq_lens=[24,11]; batch.spec_info=next_draft_input; input_ids=None/next-round rebuilt",
      ["临时 verify fields 不跨轮", "durable state 只保留新长度与 bonus"],
      [T("batch.seq_lens", "[2]", "int64", "[24,11]"),T("batch.spec_info", "object", "DFlashDraftInputV2", "bonus=[Y0,Y1]"),T("batch.forward_mode", "enum", "CPU", "DECODE")],
      [R("ScheduleBatch fields", "After", [C("durable len","write"),C("next draft input","write"),C("DECODE","write")])],
      ["跨轮状态", "旧 [22,10]/X", "新 [24,11]/Y", "Scheduler durable", "next scheduling round"],
      "restore 清理临时视图，commit 选择性推进正确状态；两者目的不同、顺序不能合并。", "忘记恢复 forward_mode，Scheduler.process_batch_result 最先按错误的 Prefill/extend 分支分派结果。", "GPU 工作读取的是 verify_forward_batch，不是恢复后的 ScheduleBatch。")
  ];

  const prefillSteps = [
    S("prefill-target", "Target Prefill", "真实 prompt token 进入 Target；最后一个 hidden 产生下一 token X，但 X 本身尚未进入 Transformer",
      "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:379–425", "batch_output = target_worker.forward_batch_generation(batch, capture_hidden_mode=FULL)",
      [T("prompt input_ids", "[20]", "int64 · CUDA", "[p0,p1,…,p19]"),T("positions", "[20]", "int64 · CUDA", "[0,1,…,19]"),T("out_cache_loc", "[20]", "int64 · CUDA", "[L0,L1,…,L19]")],
      "H = TargetTransformer(prompt)              # [20,Haux]\nlogits_t = LMHead(H[t])                     # predicts token at t+1\nX = sample/argmax(logits_19)",
      ["H[0]…H[19] 与真实输入 p0…p19 对齐", "X 是 logits(H[19]) 的输出，不存在 H[X]"],
      [T("logits_output.hidden_states", "[20,Haux]", "model dtype · CUDA", "[hp0,…,hp19]"),T("next_token_ids", "[1]", "int64 · CUDA", "[X]"),T("Target KV valid", "positions 0…19", "per-layer K/V", "包含 prompt")],
      [R("Target 输入", "真实 token", [C("p0","read"),C("…","read"),C("p18","read"),C("p19","read"),C("X","off","只是预测")]),R("Target hidden", "与输入对齐", [C("h0","cuda"),C("…","cuda"),C("h18","cuda"),C("h19","cuda"),C("hX?","off","不存在")])],
      ["Target hidden", "无", "[20,Haux]", "本轮轮内，随后注入", "next: TargetHiddenKvInjector"],
      "FULL hidden 为每个真实 prompt 位置提供 Draft KV 投影源。", "把 X 画进本轮 hidden/KV 等于假造一个从未执行的 Transformer 行。", "有 prefix-cache 命中时只处理 extend suffix；固定例假设完整 20-token prefill。"),
    S("prefill-inject", "投影并写 Draft KV", "positions/cache_loc 决定写哪里；target_hidden 决定写什么，模型专属函数完成每层 K/V 投影与 RoPE",
      "python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py:31–78", "draft_model.write_target_hidden_kv(target_hidden, pool, positions, cache_loc)",
      [T("target_hidden", "[20,Haux]", "model dtype", "[hp0,…,hp19]"),T("positions", "[20]", "int64", "[0,…,19]"),T("cache_loc", "[20]", "int64", "[L0,…,L19]")],
      "n_real = len(positions) = 20\ntarget_hidden = target_hidden[:n_real]\nfor t,j: DraftKV[j, cache_loc[t]] ← Project_j(target_hidden[t], position[t])",
      ["logical position 用于 RoPE/位置语义", "physical cache_loc 用于 KV pool 写地址", "Project_j 的具体矩阵随 dense/MLA 模型实现变化"],
      [T("Draft layer K/V", "20 valid token slots × layers", "KV dtype", "slot L0…L19 写入 K/V(prompt)"),T("Draft KV(X)", "不存在", "—", "X 尚未作为输入")],
      [R("Draft KV 可见", "prefix-valid", [C("p0","write"),C("…","write"),C("p18","write"),C("p19","write"),C("X","off")])],
      ["Draft KV", "未 bootstrap", "prompt positions 0…19", "持久设备状态", "next: decode anchor"],
      "Draft 模型下一轮需要与 Target prompt 语义对齐的 prefix K/V，而不只是 token ids。", "位置和 cache_loc 任一错位都会把正确 K/V 写进错误 token slot。", "本页不伪造模型专属投影矩阵；源码合同明确委托给 draft_model.write_target_hidden_kv。"),
    S("prefill-next", "保存首个 anchor", "next_token_ids=[X] 与 seq_lens=[20] 进入 DFlashDraftInputV2；X 下一轮才成为真实输入 position 20",
      "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:426–432", "next_draft_input = make_next_draft_input(bonus_tokens=next_token_ids, new_seq_lens=batch.seq_lens)",
      [T("next_token_ids", "[1]", "int64", "[X]"),T("batch.seq_lens", "[1]", "int64", "[20]"),T("Draft KV valid", "0…19", "device", "无 KV@20")],
      "next_draft_input = { bonus_tokens:[X], new_seq_lens:[20] }\nnext round positions = 20 + arange(G)\ndraft_block_ids = [X,M,M,M]",
      ["输出 X 不推进到 KV@20", "下一轮 ForwardBatch 才计算 hidden(X@20)"],
      [T("DFlashDraftInputV2.bonus_tokens", "[1]", "int64", "[X]"),T("DFlashDraftInputV2.new_seq_lens", "[1]", "int64", "[20]"),T("next Draft input", "[4]", "int64", "[X,M,M,M]")],
      [R("时间轴", "prefill→decode", [C("p19","write"),C("predict X","op"),C("X@20","read","下一轮"),C("K/V(X)","off","下一轮才写")])],
      ["rolling anchor", "无", "X@20", "跨轮 durable", "next: DraftBlockProposer"],
      "预测与输入错一位是整个 rolling-anchor/KV commit 语义的根。", "若把 seq_lens 写成 21，就会让下一轮跳过 position20 的 X 计算。", "new_seq_lens 仍是已具有 KV 的输入长度 20；bonus 单独保存。")
  ];

  const historyBudget = S(
    "history-budget", "历史 survival → budget", "两轮前 confidence 估算收益；SPS 表提供成本，最终只返回一个 batch-global scalar",
    "python/sglang/srt/speculative/dspark_components/dspark_planner.py:954–991,1047–1118",
    "decision = compute_verify_token_budget(history_survival_probs, sps_table, cfg)",
    [T("C_(t-2)", "[2,4]", "float32 · CPU", "req0 [.90,.80,.50,.20]\nreq1 [.70,.60,.50,.40]"),T("SPS probes", "2 arrays", "CPU", "tokens [1,5] → SPS [1.0,.5]")],
    "S = cumprod(C_(t-2), dim=1)\nsorted = sort(S.flatten(), descending=True)\ntau_star(k)=B+sum(sorted[:k])\ntheta(k)=tau_star(k)×SPS(B+k)\nbudget=argmax(theta)",
    ["S0=[.900,.720,.360,.072]；S1=[.700,.420,.210,.084]", "sorted=[.900,.720,.700,.420,.360,.210,.084,.072]", "theta: k0=2.000, k1=2.900, k2=3.620, k3=2.160, …", "最大 k=2 → budget=2"],
    [T("history survival", "[2,4]", "float32 CPU", "[[.900,.720,.360,.072],\n [.700,.420,.210,.084]]"),T("VerifyBudgetDecision.budget", "scalar", "Python int", "2"),T("predicted_theta", "scalar", "float", "3.620")],
    [R("k / theta", "SPS 在 B+k=5 降档", [C("0/2.000","read"),C("1/2.900","read"),C("2/3.620","write","最大"),C("3/2.160","off"),C("4/2.370","off"),C("5/2.550","off"),C("6/2.655","off"),C("7/2.697","off"),C("8/2.733","off")])],
    ["verify budget", "None/历史未决", "2", "batch-global host scalar", "next: current top-k"],
    "历史 survival 表示多验证一个 slot 的期望收益；SPS 把额外 Target 行的吞吐成本纳入同一目标。",
    "用当前 confidence 算总预算会混淆时间职责；当前 C_t 是下一步‘分给谁’。",
    "generation guard 不匹配时 survival 回退为 ones；默认 lag 是稳态两轮，不是无条件拿物理前两次 batch。"
  );

  const samplingSteps = [
    S("pq", "实际 p 与 q", "temperature/top-k/top-p 先塑造真实 proposal q 和 Target p；verify 必须使用这两个实际分布",
      "python/sglang/srt/speculative/dspark_components/dspark_verify.py:679–696; dspark_accept.py:80–107", "q=SoftmaxTemp(corrected_logits); p=build_target_probs(target_logits)",
      [T("draft corrected logits → q", "[4]", "probability", "A .50, B .20, C .20, D .10"),T("target logits → p", "[4]", "probability", "A .20, B .45, C .25, D .10"),T("draft token", "scalar", "int64", "A")],
      "q = [.50,.20,.20,.10]\np = [.20,.45,.25,.10]\naccept_prob(A)=min(1,p(A)/q(A))=.20/.50=.40",
      ["q/p 都已包含采样配置影响", "draft 在 A 上给出 .50，Target 只给 .20"],
      [T("accept probability", "scalar", "float", ".40"),T("reject probability", "scalar", "float", ".60")],
      [R("q", "Draft", [C("A .50","read"),C("B .20","read"),C("C .20","read"),C("D .10","read")]),R("p", "Target", [C("A .20","read"),C("B .45","read"),C("C .25","read"),C("D .10","read")])],
      ["draft A", "已提案", "以 .4 接受 / .6 拒绝", "随机决策", "next: residual"],
      "接受步骤保留 p 与 q 的重叠概率质量。", "用未经 temperature/top-k/top-p 的 logits 重建 q，会与真实 proposal 分布不一致。", "固定 4-token vocab 是教学分布，不是实际模型 vocab dump。"),
    S("residual", "拒绝后的 residual", "拒绝表示 Draft 在 A 上多给的质量被扣掉；Target 比 Draft 多的 B/C 质量再被归一化",
      "python/sglang/kernels/ops/speculative/reject_sampling.py:101–145", "residual = clamp(p-q,min=0); residual /= residual.sum()",
      [T("p-q", "[4]", "float", "[-.30, +.25, +.05, 0]"),T("positive part", "[4]", "float", "[0,.25,.05,0]")],
      "mass = .25 + .05 = .30\nresidual = [0,.25/.30,.05/.30,0]\n         = [0,.8333,.1667,0]",
      ["A 的负差被 clamp 为 0", "B/C 是拒绝后仍需补回的 Target 质量"],
      [T("residual", "[4]", "probability", "[0,.8333,.1667,0]"),T("CDF", "[4]", "float", "[0,.8333,1.0,1.0]")],
      [R("max(p-q,0)", "未归一化", [C("0","off"),C(".25","op"),C(".05","op"),C("0","off")]),R("residual", "归一化", [C("0","off"),C(".8333","write"),C(".1667","write"),C("0","off")])],
      ["拒绝后分布", "p/q", "residual", "轮内概率 Tensor", "next: CDF sample"],
      "residual 只补 Target 超出 Draft 的部分，和前面的接受质量合起来恢复 Target 边际分布。", "拒绝后直接令分布=p 会重复计算已由接受步骤贡献的共同质量。", "源码对 q=NaN 的退化行按 0 处理；一般数值例不触发。"),
    S("sample", "CDF 随机抽一个 bonus", "多个 residual 概率为正只代表多个可能结果；一次 kernel 仍只写一个 final_token",
      "python/sglang/kernels/ops/speculative/reject_sampling.py:121–160", "target_u = u * norm_sum; final_token = first(cumsum > target_u)",
      [T("residual", "[4]", "probability", "B .8333, C .1667"),T("u", "scalar", "uniform[0,1)", ".90")],
      "CDF(B)=.8333\nCDF(C)=1.0\n默认 u=.90 > .8333 → final_token=C",
      ["若 u≤.8333，抽到 B", "若 u>.8333，抽到 C", "Target argmax=B 不能替代这次随机抽样"],
      [T("bonus", "[1]", "int64", "C"),T("correct_len", "[1]", "int", "0（A 被拒绝）")],
      [R("CDF 区间", "u=.90", [C("A","off"),C("B 0→.8333","read"),C("C .8333→1","write"),C("D","off")])],
      ["replacement token", "未决", "C", "单个随机样本", "next: finalize"],
      "随机选一个 residual token 才保持 sampling 语义。", "改取 Target argmax B 会把拒绝后的 B 概率从 .8333 强行变成 1。", "全体 drafts 都接受时，最终 bonus 直接从最后一行 Target p 采样，不用 residual。")
  ];

  const cudaSteps = [
    S("alloc", "构造静态 buffers", "max_bs/stride 决定固定地址 backing；strided 输出在非 capture warm-up 时按实际 hidden/vocab 宽度分配",
      "python/sglang/srt/speculative/dspark_components/dspark_verify.py:453–492,538–556", "epilogue = DsparkVerifyEpilogue(max_bs, W, device); _ensure_out(compact)",
      [T("max_bs", "scalar", "int", "例如 8"),T("stride=W", "scalar", "int", "5"),T("compact_logits.shape[1]", "scalar", "int", "V")],
      "verify_lens_buf = zeros([max_bs], int64)\ndraft_tokens_buf = zeros([max_bs×G], int64)\nout_tokens_buf = zeros([max_bs,W], int64)\nstrided_logits = empty([max_bs×W,V])  # warm-up",
      ["地址在 capture/replay 生命周期保持稳定", "_ensure_out 断言 capture 中不得首次分配"],
      [T("verify_lens_buf", "[8]", "int64 CUDA", "[0,0,0,0,0,0,0,0]"),T("inject_gate_buf", "[1]", "int32 CUDA", "[0]"),T("strided_logits", "[40,V]", "model dtype", "已分配 backing")],
      [R("静态地址", "warm-up", [C("lens ptr","cuda"),C("gate ptr","cuda"),C("logits ptr","cuda"),C("hidden ptr","cuda"),C("accept ptr","cuda")])],
      ["graph buffers", "未分配/部分分配", "固定最大 shape", "跨轮复用设备状态", "next: capture"],
      "CUDA Graph 重放依赖稳定地址；每轮变化的是 buffer 内容，不是地址。", "capture 内首次 allocator 分配会破坏可重放地址合同，源码直接 assert。", "页面不声称实测指针值，只展示当前分配 shape 与生命周期。"),
    S("capture", "Capture 记录算子", "capture_hook 把 compact output 接到 scatter→accept→可选 commit；记录的是对既有 buffer 的设备操作",
      "python/sglang/srt/speculative/dspark_components/dspark_verify.py:493–510,558–576", "capture_hook(...): epilogue(compact_logits, compact_hidden, input_ids, seq_lens, ...)",
      [T("compact_logits", "[graph_tokens,V]", "static CUDA", "graph output buffer"),T("verify_lens_buf", "[max_bs]", "static CUDA", "graph 每次从同一地址读取"),T("accept buffers", "fixed max shapes", "static CUDA", "graph 写入")],
      "capture records:\ncompact → scatter(strided)\nstrided + candidates → accept\ncorrect_len → finalize/out_tokens\n[optional] hidden → Draft KV commit",
      ["Python 控制流不在每轮 replay 中重新执行", "设备算子集合固定，输入内容由 begin_step/load_batch 更新"],
      [T("captured graph", "static op DAG", "CUDA Graph", "scatter→accept→optional commit"),T("buffer addresses", "unchanged", "device ptr", "全部既有")],
      [R("graph DAG", "capture", [C("Target","cuda"),C("scatter","cuda"),C("accept","cuda"),C("finalize","cuda"),C("commit?","cuda")])],
      ["execution placement", "Python launches", "captured DAG", "可跨轮 replay", "next: begin_step"],
      "把稳定尾处理捕获进图能减少逐 kernel Python launch。", "把 capture 看成‘保存上一轮结果’会混淆操作图与每轮数据内容。", "epilogue 仅在当前支持的 NVIDIA CUDA 路径成立。"),
    S("begin", "每轮覆盖 lens/gate", "N+1 复用同一地址前，begin_step 必须把当轮 verify_lens 与 armed 覆盖进去",
      "python/sglang/srt/speculative/dspark_components/dspark_verify.py:511–520", "epilogue.begin_step(layout.verify_lens, armed=fold_eligible)",
      [T("N buffer content", "lens[8]+gate[1]", "static CUDA", "lens=[2,2,0…], gate=[1]"),T("N+1 inputs", "lens[2]+bool", "CUDA/host", "lens=[3,1], armed=0")],
      "verify_lens_buf[:2].copy_([3,1])\nverify_lens_buf[2:].zero_()\ninject_gate_buf.fill_(0)",
      ["地址不变", "内容从 N 的 [2,2]/1 变成 N+1 的 [3,1]/0"],
      [T("verify_lens_buf after", "[8]", "int64 CUDA", "[3,1,0,0,0,0,0,0]"),T("inject_gate_buf after", "[1]", "int32 CUDA", "[0]")],
      [R("跨轮 buffer", "N → N+1", [C("2→3","write","req0"),C("2→1","write","req1"),C("1→0","write","gate"),C("tail→0","write")])],
      ["static buffer content", "N lens/gate", "N+1 lens/gate", "每轮 mutable state", "next: replay"],
      "begin_step 是静态地址与动态批次之间的写屏障。", "省略后 N+1 会按 N 的 verify_lens scatter/accept，甚至沿用上一轮 inject_gate。", "armed 控制图内 commit 是否生效；图内算子可能仍被 replay。"),
    S("replay", "Replay 与结果采用", "graph 写 accept buffers；Worker 仍用 folded_accept/folded_commit 决定是否把这些 buffer 当权威结果",
      "python/sglang/srt/speculative/dspark_components/dspark_verify.py:521–537,578–655; dspark_worker_v2.py:627–659", "graph.replay(); accept = epilogue.read_accept(bs) if folded_accept else fallback_accept(...) ",
      [T("verify_lens_buf", "[max_bs]", "int64", "当前轮内容"),T("strided logits/hidden", "[max_bs×W,*]", "CUDA", "graph 写入"),T("folded_accept", "bool", "host", "fold_eligible && compact && graph_hit")],
      "graph replay writes correct_len_buf/bonus_buf/commit_lens_buf/out_tokens_buf\nif folded_accept: read_accept(bs)\nelse: ignore graph accept backing and run normal accept\nif !folded_commit: run normal commit_hidden",
      ["执行过 ≠ 一定采用", "fallback 与 folded 路径最终返回相同 AcceptOuts 合同"],
      [T("AcceptOuts view", "slices [:bs]", "views of static buffers", "correct_len/bonus/commit_lens/out_tokens"),T("GenerationBatchResult", "dynamic object", "Python", "采用 guard 后的权威结果")],
      [R("结果来源", "guard", [C("graph wrote","cuda"),C("folded_accept?","op"),C("read buffer","write"),C("fallback","read"),C("common tail","write")])],
      ["authority", "graph backing 可能被写", "guard 真才采用", "Worker semantic state", "next round"],
      "Graph 优化执行位置，Worker guard 仍保护算法语义和 fallback。", "看到 accept buffer 被写就无条件读取，会在未武装或不匹配 route 下采用错误结果。", "folded、COMPACT、CUDA Graph 命中是三个不同条件。")
  ];

  const pages = {
    "00": { index:"00 · 全流程", title:"一轮 dSpark 不是对象接力，而是 11 组 Tensor/状态连续变形", sample:"B=2 · G=4 · W=5 · prefix=[22,10] · budget=2", steps:[draftConstruct,draftForward,markovSample,currentSchedule,compactWindow,targetForward,scatterOutput,acceptGreedy,finalizeOutput,kvCommit,nextRound] },
    "01": { index:"01 · Scheduler 隔离", title:"临时重绑改变的是字段引用；ForwardBatch 捕获执行 Tensor 后，restore 与 durable commit 分开发生", sample:"T_old/S_old → T_verify/S_verify → restore → new durable state", steps:isolationSteps },
    "01A": { index:"01A · overlap 时间线", title:"FutureMap 改变值何时可用；Tensor 内容、generation guard 与引用寿命必须分别追踪", sample:"C_t + generation → FutureKey → resolve → C_(t-2)", steps:[
      S("publish","发布 Future 值","overlap 把未完成 confidence/length 以 key 交给后续轮；non-overlap 同位置已经是实 Tensor",
        "python/sglang/srt/managers/overlap_utils.py:84–116,153–229","future_map.publish(key, tensor); batch.field = key",
        [T("confidence_t","[2,4]","float32 CUDA","本轮 C_t"),T("generation_t","[2]","int64","[7,12]")],
        "non-overlap: field = Tensor C_t\noverlap: field = FutureKey(K_t), backing C_t 由 producer/event 负责",
        ["key 不是 Tensor 数据，只是值可用性的句柄"],
        [T("FutureMap[K_t]","future entry","Python + event","→ (C_t,generation_t)"),T("batch field","key or Tensor","union state","K_t / C_t")],
        [R("value state","publish",[C("C_t","cuda"),C("K_t","op"),C("event","cuda"),C("consumer","off")])],
        ["forward input","Tensor","FutureKey(等待中)","overlap transport","next: resolve"],
        "FutureMap 让 CPU 调度不必立刻等待设备结果。","把 FutureKey 当成 Tensor 传入 planner 会在 shape/dtype 访问处失败。","pure decode 中字段已是 Tensor 时 resolve_forward_inputs 被调用但 guard-only no-op。"),
      S("resolve","消费前解析","下一消费者在真正读 shape/value 前把 key 替换为 Tensor，并等待相应完成事件",
        "python/sglang/srt/managers/overlap_utils.py:84–116","resolved = future_map.resolve(K_t); batch.field = resolved",
        [T("batch.field","FutureKey","Python","K_t"),T("FutureMap[K_t]","entry","event-backed","C_t [2,4]")],
        "if field is FutureKey: wait/resolve → Tensor\nelse: return field unchanged",
        ["non-overlap 已是 Tensor，guard 不查 map", "overlap 在此获得真实 shape=[2,4]"],
        [T("batch.field","[2,4]","float32 Tensor","C_t"),T("generation","[2]","int64","[7,12]")],
        [R("resolve","key→value",[C("K_t","read"),C("wait event","cuda"),C("C_t","write")])],
        ["consumer input","key/real value","统一 Tensor","当前轮临时字段","next: generation guard"],
        "值的发布时间与消费时间解耦，但消费前必须恢复普通 Tensor 合同。","跳过 resolve 会把 FutureKey 送进 cumprod/top-k。","是否阻塞取决于 event 是否已完成，不等于每次调用都等待。"),
      S("lag","generation guard + 两轮 lag","planner 不是盲取数组上一格；只有 generation 对得上才使用历史 confidence，否则 survival 回退为 1",
        "python/sglang/srt/speculative/dspark_components/dspark_planner.py:1087–1118","fresh=(current_gen>=1)&(lagged_generation==current_gen); where(fresh,cumprod(C),ones)",
        [T("lagged_confidence","[2,4]","float32 CPU","C_(t-2)"),T("lagged_generation","[2]","int64","[7,11]"),T("current_generation","[2]","int64","[7,12]")],
        "survival=cumprod(lagged_confidence,dim=1)\nfresh=[7==7,11==12]=[1,0]\nresult=req0 survival; req1 ones",
        ["row0 使用历史 survival", "row1 request-pool row 已换代，阻断旧请求信息"],
        [T("history_survival","[2,4]","float32 CPU","req0 cumprod(C); req1 [1,1,1,1]")],
        [R("generation guard","两请求",[C("req0 fresh","write"),C("req1 stale","fail"),C("fallback ones","op")])],
        ["history input","可能陈旧","generation-safe survival","budget planner state","next: SPS budget"],
        "generation guard 防止 request-pool 行复用后读取上一个请求的 confidence。","只按 row index 取历史会跨请求串线。","默认两轮 lag 是稳态时间语义；warm-up/非 decode/复用时使用 fallback。"),
      S("lifetime","引用保活","FutureMap 管值，batch_record_buf/record_stream 管 backing 生命周期；二者不是同一问题",
        "python/sglang/srt/managers/scheduler.py:3608–3621","batch.seq_lens.record_stream(current_stream); batch_record_buf.append(batch)",
        [T("Tensor backing","device allocation","CUDA","仍被异步 kernel 读取"),T("Python batch ref","object","host","可能离开当前栈帧")],
        "record_stream marks allocator use on CUDA stream\nrecord buffer retains Python owners until safe reuse boundary",
        ["值已经 publish 不代表 backing 可以立刻复用"],
        [T("retained refs","queue/deque","Python","保活 previous batch/tensors"),T("safe reuse","after stream/event","lifecycle","可释放/覆盖")],
        [R("lifetime","CPU/CUDA",[C("publish","op"),C("kernel reads","cuda"),C("retain","write"),C("event done","cuda"),C("reuse","write")])],
        ["backing lifetime","当前栈","延长到设备完成","scheduler runtime","next batch"],
        "FutureMap 解决‘值何时可读’，引用保活解决‘内存何时可复用’。","只 publish 不保活可能让设备继续读已复用 backing。","non-overlap 也要遵守 CUDA stream 生命周期，只是值不经 FutureMap 延迟发布。")
    ]},
    "02": { index:"02 · Prefill bootstrap", title:"从 prompt hidden 到 Draft K/V：X 只是预测结果，本轮没有 hidden(X) 或 KV(X)", sample:"seq_len=20 · prompt positions 0…19 · next anchor X@20", steps:prefillSteps },
    "03": { index:"03 · Draft proposal", title:"[anchor,MASK,MASK,MASK] 如何变成 hidden、base logits、串行 draft_tokens 与真实 proposal 分布", sample:"B=2 · G=4 · Transformer 8 rows · Markov 4 steps", steps:[draftConstruct,draftForward,markovSample] },
    "03A": { index:"03A · Markov Head", title:"同一个 [B,G,V] base logits，Vanilla/Gated/RNN 用不同 Tensor 计算偏置，但 token 链都逐步推进", sample:"B=2 · G=4 · rank=R · hidden=Hd · vocab=V", steps:[draftForward,
      S("vanilla","Vanilla 偏置","上一 token 先查低秩 embedding，再投影成 vocab bias 与当前 base logits 相加",
        "python/sglang/srt/models/dspark.py:67–130","bias=W2(W1(prev_token)); step_logits=base_logits+bias",
        [T("prev_tokens","[B]","int64","[X0,X1] / [a,e] / …"),T("markov_w1","[V,R]","parameter","embedding table"),T("base_logits[:,i,:]","[B,V]","model dtype","第 i 个 MASK hidden 的 base logits")],
        "prev_emb = Embedding(V,R)[prev]          # [B,R]\nbias = Linear(R,V)(prev_emb)              # [B,V]\nstep_logits = base_logits[:,i,:] + bias",
        ["每步只对 [B,R]→[B,V] 做低秩投影", "采出 token 后成为下一步 prev"],
        [T("step_logits","[B,V]","model dtype","L_i + bias(prev)"),T("next_tokens","[B]","int64","sampler(step_logits)")],
        [R("token chain","serial",[C("X→a","write"),C("a→b","write"),C("b→c","write"),C("c→d","write")])],
        ["Markov state","prev token","next token","每步更新","next Markov step"],
        "低秩 W1/W2 用很小的串行头补 token 依赖。","把 bias 一次按 anchor 广播到四步，会让 b/c/d 都只依赖 X。","Vanilla 不读取 hidden_states，但 base logits 本身仍来自各位置 hidden。"),
      S("gated","Gated 偏置","当前 hidden 与 prev embedding 共同计算 gate，再调制低秩 token embedding",
        "python/sglang/srt/models/dspark.py:133–162","gate=sigmoid(gate_proj(cat(hidden,prev_emb))); bias=W2(gate*prev_emb)",
        [T("hidden_i","[B,Hd]","model dtype","H[:,i,:]"),T("prev_emb","[B,R]","model dtype","W1(prev)"),T("gate_proj","[Hd+R → R]","parameter","linear")],
        "gate_input = cat([hidden_i,prev_emb],-1)  # [B,Hd+R]\ngate = sigmoid(proj(gate_input))          # [B,R]\nbias = W2(gate * prev_emb)                # [B,V]",
        ["hidden 决定 token embedding 的哪些 rank 分量生效", "仍然逐步更新 prev token"],
        [T("gate","[B,R]","model dtype","0…1"),T("step_logits","[B,V]","model dtype","base+bias")],
        [R("数据依赖","current step",[C("hidden_i","read"),C("prev token","read"),C("gate","op"),C("next token","write")])],
        ["bias source","prev token","prev + hidden gate","轮内","next step"],
        "gate 让每个位置的 hidden 参与决定 Markov 修正强度。","遗漏 hidden 会触发当前源码 ValueError，不是退化成 Vanilla。","gate 不让 Transformer 重新看到已采 token；依赖仍在 head 内。"),
      S("rnn","RNN 状态","RNN 额外携带 [B,R] state，使更早 token 影响后续所有步骤",
        "python/sglang/srt/models/dspark.py:164–263","state,bias=_rnn_step(state,prev_emb,hidden_i); logits=base+bias",
        [T("state_i","[B,R]","model dtype","i=0 时 zeros"),T("prev_emb","[B,R]","model dtype","W1(prev)"),T("hidden_i","[B,Hd]","model dtype","H[:,i,:]")],
        "z=cat([state,prev_emb,hidden_i])          # [B,2R+Hd]\ngate,candidate,out = joint_proj(z).chunk(3)\nstate'=gate*state+(1-gate)*tanh(candidate)\nbias=W2(tanh(out)); logits=base+bias",
        ["state 在 i=0…3 串行更新", "next token 与 state 同时进入下一步依赖"],
        [T("state_(i+1)","[B,R]","model dtype","persistent within block"),T("step_logits","[B,V]","model dtype","base+bias")],
        [R("RNN block","serial",[C("s0=0","read"),C("s1","write"),C("s2","write"),C("s3","write"),C("s4","write")])],
        ["recurrent state","zeros","updated R state","仅本 block","next step"],
        "RNN state 表达比单 prev token 更长的 block 内历史。","每一步重新把 state 置零会退化并丢掉跨 draft 的累积信息。","RNN 串行的是小头，不是 Draft Transformer。")
    ]},
    "04": { index:"04 · Confidence/预算/Layout", title:"同一个 cumprod 出现两次：历史 Tensor 决定总数，当前 Tensor 决定分配，layout 再决定物理形状", sample:"C_(t-2) → budget=2 · C_t → verify_lens=[2,2] · graph=5", steps:[historyBudget,currentSchedule,compactWindow] },
    "04A": { index:"04A · SPS 预算", title:"把历史 confidence 的每次数值计算展开到 tau_star、SPS 档位与 argmax budget", sample:"历史 2×4 survival · k=0…8 · 最大 theta 在 k=2", steps:[historyBudget] },
    "04B": { index:"04B · Top-k/Layout", title:"当前 confidence 如何按全局排名变成 verify_lens、indptr 和 graph-tier padding", sample:"raw budget=2 · 默认 [2,2]/padding1 · alignment 后 [2,3]/padding0", steps:[currentSchedule,compactWindow] },
    "05": { index:"05 · Target Verify", title:"完整候选如何压成 4 个逻辑行、补到 5 个物理行、计算 logits/hidden，再散布回 10 个接受槽", sample:"2×5 candidates → 4 logical → 5 physical → 10 strided", steps:[compactWindow,targetForward,scatterOutput] },
    "06": { index:"06 · Accept/Finalize", title:"从 Target 每行分布到 correct_len、bonus、commit_lens 与有效输出，每个 Tensor 的索引都可手算", sample:"req0 accept 1 · req1 accept 0 · output lengths [2,1]", steps:[acceptGreedy,finalizeOutput] },
    "06A": { index:"06A · Sampling 修正", title:"把 p、q、接受率、positive residual 与 CDF 的每个数值展开；bonus 是一次随机样本，不是 argmax", sample:"q=[.50,.20,.20,.10] · p=[.20,.45,.25,.10] · u=.90", steps:samplingSteps },
    "07": { index:"07 · Hidden/KV/Anchor", title:"同一轮的三套 token 内容：Target 输入 [X,a]、用户输出 [a,Y]、Draft KV 提交 [X,a]", sample:"prefix=20 · correct_len=1 · commit_len=2 · next anchor=Y@22", steps:[
      S("alignment","hidden/预测错位","hidden 与输入 token 对齐；LM head 的预测在下一 token rail，因此用户输出和 KV rail 永远错一位",
        "python/sglang/srt/speculative/dspark_components/dspark_verify.py:287–318","hidden = logits_output.hidden_states.view(B,W,-1)",
        [T("Target inputs","[1,5]","int64","[X,a,b,c,d]"),T("Target hidden","[1,5,Haux]","model dtype","[hX,ha,hb,hc,hd]"),T("Target predict","[1,5]","int64","[a,Y,?,?,?]")],
        "LMHead(hX) → a\nLMHead(ha) → Y\naccept a; reject b → correct_len=1; bonus=Y",
        ["hX 支持 d1=a", "ha 产生边界 bonus Y", "Y 没有对应 hY"],
        [T("correct_len","[1]","int","[1]"),T("bonus","[1]","int64","[Y]"),T("commit_lens","[1]","int32","[2]")],
        [R("Target 输入","hidden rail",[C("X","read"),C("a","read"),C("b","off"),C("c","off"),C("d","off")]),R("next 预测","output rail",[C("a","write"),C("Y","write"),C("?","off"),C("?","off"),C("?","off")])],
        ["边界", "inputs [X,a,…]", "output [a,Y]", "自回归错一位", "next: commit mask"],
        "一个 hidden 只能代表它实际消费的输入 token；next-token 是该 hidden 的预测结果。","用 ha 伪装成 hY 会把 Y 的语义与 position 错位。","这是 token 对齐事实，不依赖 compact/non-compact。"),
      S("single-kv","提交 [X,a]","commit_len=2 在输入/hidden rail 上打开前两列，因此 Draft KV 写 X@20 与 a@21",
        "python/sglang/srt/speculative/dspark_components/dspark_kv_inject.py:31–157","inject_target_hidden(target_hidden, cache_loc_2d, positions, commit_lens=[2])",
        [T("hidden","[1,5,Haux]","model dtype","[hX,ha,hb,hc,hd]"),T("positions","[1,5]","int64","[20,21,22,23,24]"),T("commit_lens","[1]","int32","[2]")],
        "mask = arange(5) < 2 = [1,1,0,0,0]\nwrite Project(hX,pos20) → Draft KV slot L20\nwrite Project(ha,pos21) → Draft KV slot L21",
        ["hb/hc/hd 即使算过也不提交", "suffix 不写、不清零，由 new_seq_lens 隐藏"],
        [T("Draft KV writes","2 token positions","per-layer K/V","X@20, a@21"),T("new_seq_lens","[1]","int64","[22]")],
        [R("commit mask","input rail",[C("X@20","write"),C("a@21","write"),C("b@22","off"),C("c@23","off"),C("d@24","off")])],
        ["Draft KV", "valid through 19", "新增 X20/a21", "持久状态", "next: Y@22"],
        "Draft 下一轮需要已确认输入历史的 K/V。","把输出 [a,Y] 写 KV 会漏 X 并提前写不存在的 Y。","物理 suffix backing 可残留旧值；逻辑长度 22 阻止读取。"),
      S("rolling","Y 下一轮才计算","用户本轮看见 [a,Y]；KV 本轮写 [X,a]；Y 作为 bonus_tokens 跨轮到 position22",
        "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:684–700","next_draft_input={bonus_tokens:[Y],new_seq_lens:[22]}",
        [T("用户有效输出","length 2","tokens","[a,Y]"),T("Draft KV 新增","length 2","inputs","[X,a]"),T("next bonus","[1]","int64","[Y]")],
        "next position = new_seq_lens = 22\nnext draft input=[Y,M,M,M]\nTransformer(Y@22) → hY → then K/V(Y) can be written",
        ["本轮没有 hY", "下一轮补算 Y"],
        [T("DFlashDraftInputV2","object","durable","bonus=[Y], new_seq_lens=[22]"),T("next Draft positions","[4]","int64","[22,23,24,25]")],
        [R("五条同步 rail","两轮",[C("input X","read"),C("predict a","op"),C("input a","read"),C("predict Y","op"),C("next input Y","write")])],
        ["rolling anchor", "X@20", "Y@22", "跨轮 durable", "next round"],
        "rolling anchor 正好填补 bonus 尚无 hidden/KV 的空缺。","若下一轮从 position23 开始，Y@22 永远没有被模型消费。","overlap 只改变 publish/resolve，token/position 对齐不变。")
    ]},
    "08": { index:"08 · Folded 快路", title:"Folded 改变的是这些 Tensor 算子在哪执行；静态 buffer 内容与采用 guard 才是正确性核心", sample:"static buffers → capture DAG → begin_step → replay/read/fallback", steps:cudaSteps },
    "08A": { index:"08A · CUDA Graph", title:"逐轮观察同一静态地址的内容变化：warm-up 分配、capture 记录、N/N+1 覆盖、replay 与结果采用", sample:"N lens=[2,2],gate=1 → N+1 lens=[3,1],gate=0", steps:cudaSteps }
  };

  const routeNames = { "static-greedy":"STATIC · 贪心", "compact-sampling":"COMPACT · 随机采样", "folded-greedy":"FOLDED · 贪心" };
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (value) => String(value).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
  const clone = (value) => JSON.parse(JSON.stringify(value));

  function initialStep(max) {
    const match = location.hash.match(/^#step-(\d+)$/);
    return Math.min(max, Math.max(0, match ? Number(match[1]) : 0));
  }

  function tensorMarkup(items, failure) {
    return items.map((item) => `<article class="tensor-item"><div class="tensor-item__name">${esc(item.name)}</div><div class="tensor-item__meta">${esc(item.shape)} · ${esc(item.dtype)}</div><div class="tensor-item__value">${esc(failure ? `[未正确产生] ${item.value}` : item.value)}</div>${item.note ? `<div class="tensor-item__note">${esc(item.note)}</div>` : ""}</article>`).join("");
  }

  function specialLab(pageId, state) {
    if (pageId === "04B") return `<div class="tensor-lab"><button type="button" data-align aria-pressed="${state.align}">${state.align ? "关闭" : "开启"}图档位对齐</button><span>对应 graph-tier alignment；默认 padding=1，开启后 allocator allowance=3，把 padding 换成真实 req1/position1</span><output>${state.align ? "verify_lens=[2,3] · padding=0" : "verify_lens=[2,2] · padding=1"}</output></div>`;
    if (pageId === "06A") return `<div class="tensor-lab"><label for="residual-u">residual CDF 随机数 u</label><input id="residual-u" type="range" min="0" max="1" step="0.01" value="${state.u}" data-u><output>u=${Number(state.u).toFixed(2)} → ${Number(state.u) <= .8333 ? "B" : "C"}</output></div>`;
    if (pageId === "08A") return `<div class="tensor-lab"><button type="button" data-skip aria-pressed="${state.skip}">${state.skip ? "恢复" : "省略"} begin_step()</button><span>比较 N+1 是否覆盖静态 verify_lens_buf / inject_gate_buf</span><output>${state.skip ? "错误：仍是 N 的 [2,2]/1" : "正确：N+1 为 [3,1]/0"}</output></div>`;
    return "";
  }

  function dynamicStep(pageId, raw, state) {
    const step = clone(raw);
    if (pageId === "04B" && state.align && step.key === "current-schedule") {
      step.calc.push("alignment allowance=graph tier 5 - B 2 = 3；第三名 req1/pos1=.9405 被选中");
      step.outputs.find((x) => x.name === "selected_extra").value = "[1,2]";
      step.outputs.find((x) => x.name === "verify_lens").value = "[2,3]";
      step.rails[1].cells[1] = C(".941","write","padding→真实验证");
    }
    if (pageId === "04B" && state.align && step.key === "compact-window") {
      step.inputs.find((x) => x.name === "verify_lens").value = "[2,3]";
      step.equation = "qo_indptr = [0,2,5]\nlogical_rows = 5\ngraph_num_tokens = 5\npadding = 0";
      step.outputs[0].value = "[X0,a,X1,e,f]";
      step.outputs[1].value = "[22,23,10,11,12]";
      step.outputs[2].value = "[L0_22,L0_23,L1_10,L1_11,L1_12]";
      step.rails[1].cells[2] = C("f","read","alignment extra");
      step.rails[2].cells = [C("X0","write"),C("a","write"),C("X1","write"),C("e","write"),C("f","write","替代 padding")];
    }
    if (pageId === "06A" && step.key === "sample") {
      const token = Number(state.u) <= .8333 ? "B" : "C";
      step.inputs.find((x) => x.name === "u").value = Number(state.u).toFixed(2);
      step.equation = `CDF(B)=.8333\nCDF(C)=1.0\nu=${Number(state.u).toFixed(2)} → final_token=${token}`;
      step.outputs.find((x) => x.name === "bonus").value = token;
      step.ledger[2] = token;
    }
    if (pageId === "08A" && state.skip && step.key === "begin") {
      step.equation = "# begin_step() 被省略\nverify_lens_buf 仍为 N 的 [2,2,0…]\ninject_gate_buf 仍为 N 的 [1]";
      step.outputs[0].value = "错误：[2,2,0,0,0,0,0,0]";
      step.outputs[1].value = "错误：[1]";
      step.rails[0].cells = [C("2 stale","fail"),C("2 stale","fail"),C("gate 1 stale","fail"),C("tail stale","fail")];
    }
    return step;
  }

  const sceneKinds = {
    matrix: new Set(["draft-block", "compact-window", "scatter", "accept", "finalize"]),
    markov: new Set(["markov", "vanilla", "gated", "rnn"]),
    rank: new Set(["history-budget", "current-schedule"]),
    probability: new Set(["pq", "residual", "sample"]),
    memory: new Set(["kv-commit", "single-kv", "prefill-inject", "alloc", "begin"]),
    lanes: new Set(["next-round", "alignment", "rolling", "prefill-target", "prefill-next", "stable", "rebind", "restore", "publish", "resolve", "lag", "lifetime", "capture", "replay"])
  };

  function sceneKind(pageId, step) {
    if (pageId === "01" && step.key === "capture") return "lanes";
    for (const [kind, keys] of Object.entries(sceneKinds)) if (keys.has(step.key)) return kind;
    return "flow";
  }

  function shapeMarkup(step) {
    const inputs = step.inputs.slice(0, 3).map((item) => `<span class="execution-shape" data-role="read">${esc(item.name)} ${esc(item.shape)}</span>`);
    const outputs = step.outputs.slice(0, 3).map((item) => `<span class="execution-shape" data-role="write">${esc(item.name)} ${esc(item.shape)}</span>`);
    return `${inputs.join('<span class="execution-shape-arrow">＋</span>')}<span class="execution-shape-arrow">→</span><span class="execution-shape" data-role="op">${esc(step.label)}</span><span class="execution-shape-arrow">→</span>${outputs.join('<span class="execution-shape-arrow">＋</span>')}`;
  }

  function displayKind(cell, failure) {
    return failure && cell.kind === "write" ? "fail" : cell.kind;
  }

  function matrixRowsMarkup(rails, failure, cols) {
    return rails.map((rail) => {
      const cells = rail.cells.concat(Array.from({ length: cols - rail.cells.length }, () => C("—", "off")));
      return `<div class="viz-matrix-row" style="--scene-cols:${cols}"><div class="viz-matrix-label">${esc(rail.name)}<small>${esc(rail.note)}</small></div>${cells.map((cell, index) => `<div class="viz-cell" style="--cell-i:${index}" data-kind="${displayKind(cell, failure)}"><span class="viz-cell__index">${index}</span><span>${esc(cell.value)}</span>${cell.note ? `<small>${esc(cell.note)}</small>` : ""}</div>`).join("")}</div>`;
    }).join("");
  }

  function remapScene(step, failure) {
    const sourceCount = step.key === "compact-window" ? 2 : 1;
    const sources = step.rails.slice(0, sourceCount);
    const destinations = step.rails.slice(sourceCount);
    const sourceCols = Math.max(...sources.map((rail) => rail.cells.length), 1);
    const destinationCols = Math.max(...destinations.map((rail) => rail.cells.length), 1);
    const tokens = step.key === "compact-window" ? ["X0", "a", "X1", "e"] : ["X0", "a", "X1", "e"];
    const op = step.key === "compact-window" ? "gather\nverify_lens=[2,2]\nrow4=padding" : "scatter\ndst=req×W+within\nfill suffix=0";
    return `<p class="scene-primer"><b>映射关系：</b>${step.key === "compact-window" ? "从每个请求只取连续前缀，拼成 compact flat rows；物理第 5 行只是 graph-tier padding。" : "compact row 先恢复 request/within 坐标，再写入固定宽度 [B,W]；不是按 compact 下标直接顺排。"}</p><div class="viz-remap"><div class="viz-remap-stack"><b>源 Tensor</b><div class="viz-matrix">${matrixRowsMarkup(sources, false, sourceCols)}</div></div><div class="viz-remap-center"><div class="viz-remap-operator">${esc(op)}</div><div class="viz-remap-packets">${tokens.map((token, index) => `<span style="--packet-i:${index}">${esc(token)}</span>`).join("")}</div></div><div class="viz-remap-stack"><b>目标 Tensor</b><div class="viz-matrix">${matrixRowsMarkup(destinations, failure, destinationCols)}</div></div></div>`;
  }

  function matrixScene(step, failure) {
    if (step.key === "compact-window" || step.key === "scatter") return remapScene(step, failure);
    const cols = Math.max(...step.rails.map((rail) => rail.cells.length), 1);
    const rows = matrixRowsMarkup(step.rails, failure, cols);
    return `<p class="scene-primer"><b>读图：</b>每一列是一个确定的 token/row/内存槽；播放时依次点亮读取、算子和最终写入。灰色槽仍有物理 backing，但不具备本轮有效语义。</p><div class="viz-matrix">${rows}</div>`;
  }

  function flowScene(step, failure) {
    const inputs = step.inputs.slice(0, 4).map((item) => `<div class="viz-port"><b>${esc(item.name)}</b>${esc(item.shape)}<br>${esc(item.value)}</div>`).join("");
    const outputs = step.outputs.slice(0, 4).map((item) => `<div class="viz-port" data-role="write"><b>${esc(item.name)}</b>${esc(item.shape)}<br>${esc(failure ? "未正确写入" : item.value)}</div>`).join("");
    const opLine = step.equation.split("\n").slice(0, 2).join(" · ");
    return `<p class="scene-primer"><b>数据不是“经过一个方块”：</b>左侧是算子真正读取的 shape/value，中间是索引或 Tensor 运算，右侧是新 backing、视图或跨轮状态。</p><div class="viz-flow"><div class="viz-port-cluster">${inputs}</div><div class="viz-edge"><span>▶</span></div><div class="viz-operator"><b>${esc(step.call)}</b><code>${esc(opLine)}</code></div><div class="viz-edge"><span>▶</span></div><div class="viz-port-cluster">${outputs}</div></div>`;
  }

  function markovScene(step, failure) {
    const prev = ["X", "a", "b", "c"];
    const next = ["a", "b", "c", "d"];
    const cards = next.map((token, index) => `<div class="viz-markov-step"><b>i=${index} · slot hidden 已存在</b><div class="viz-markov-hidden">H[:,${index},:] → base_logits[:,${index},:]</div><div class="viz-markov-token"><span>${prev[index]}</span><span>＋ bias / gate</span><strong>${failure ? "?" : token}</strong></div><small>采样结果${index < 3 ? "成为下一步 prev" : "写入 draft_tokens"}</small></div>`).join("");
    return `<p class="scene-primer"><b>并行/串行边界：</b>四个 hidden 已由同一次 Draft Transformer forward 同时得到；动画只让小型 Markov 头按 X→a→b→c 推进，不重复 Transformer。</p><div class="viz-markov">${cards}</div>`;
  }

  function lastNumber(value) {
    const matches = String(value).match(/-?(?:\d+\.?\d*|\.\d+)/g);
    return matches?.length ? Number(matches[matches.length - 1]) : 0;
  }

  function rankScene(step, failure) {
    const entries = step.rails.flatMap((rail) => rail.cells.map((cell, index) => ({ rail, cell, index, value: lastNumber(cell.value) })));
    const max = Math.max(...entries.map((entry) => entry.value), 1);
    const rows = entries.map((entry) => {
      const selected = entry.cell.kind === "write" && !failure;
      const width = Math.max(2, Math.min(100, entry.value / max * 100));
      return `<div class="viz-rank-row" data-selected="${selected}"><b>${esc(entry.rail.name)} · ${entry.index}</b><div class="viz-rank-track"><div class="viz-rank-fill" style="--bar:${width.toFixed(1)}%"></div></div><span class="viz-rank-value">${esc(entry.cell.value)}${selected ? " ✓" : ""}</span></div>`;
    }).join("");
    const conclusion = step.key === "history-budget" ? "历史 survival 与 SPS 共同决定最大 theta 的 k=2" : "当前 survival 全局排序，绿色项得到两个额外验证槽";
    return `<p class="scene-primer"><b>排名不是装饰：</b>${conclusion}。条长表示比较值，绿色表示本轮真正改变 budget/verify_lens 的入选项。</p><div class="viz-rank">${rows}</div>`;
  }

  const samplingDistribution = {
    q: [.50, .20, .20, .10],
    p: [.20, .45, .25, .10],
    r: [0, .8333, .1667, 0]
  };

  function probabilityScene(step, state, failure) {
    const tokens = ["A", "B", "C", "D"];
    const series = step.key === "pq" ? ["q", "p"] : ["r"];
    const cards = tokens.map((token, index) => `<div class="viz-prob-token"><b>${token}</b>${series.map((name) => {
      const value = samplingDistribution[name][index];
      return `<div class="viz-prob-row" data-series="${name}"><span>${name}</span><div class="viz-prob-track"><div class="viz-prob-fill" style="--prob:${value * 100}%"></div></div><span>${value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}</span></div>`;
    }).join("")}</div>`).join("");
    const u = Number(state.u);
    const sampled = u <= .8333 ? "B" : "C";
    const cdf = step.key === "sample" ? `<div class="viz-cdf" style="--cdf-cols:83.33fr 16.67fr"><div class="viz-cdf-segment">B · 0→.8333</div><div class="viz-cdf-segment">C · .8333→1</div><div class="viz-cdf-marker" data-u="${u.toFixed(2)}" style="--u:${u * 100}%"></div></div><div class="viz-prob-result" data-kind="${failure ? "fail" : "write"}">一次抽样只落入一个区间：u=${u.toFixed(2)} → bonus=${failure ? "错误/未采样" : sampled}；Target argmax=B 不能替代这一步。</div>` : "";
    return `<p class="scene-primer"><b>面积就是概率质量：</b>${step.key === "pq" ? "蓝 q 是真实 Draft proposal，绿 p 是 Target；A 的接受率=min(1,.20/.50)=.40。" : "拒绝 A 后，只保留 max(p−q,0)；归一化得到 B/C 两个互斥区间。"}</p><div class="viz-prob">${cards}</div>${cdf}`;
  }

  function memoryScene(step, failure) {
    const cols = Math.max(...step.rails.map((rail) => rail.cells.length), 1);
    const rows = step.rails.map((rail) => {
      const cells = rail.cells.concat(Array.from({ length: cols - rail.cells.length }, () => C("—", "off")));
      return `<div class="viz-memory-row" style="--memory-cols:${cols}"><div class="viz-memory-label">${esc(rail.name)}<br>${esc(rail.note)}</div>${cells.map((cell, index) => `<div class="viz-cell" style="--cell-i:${index}" data-kind="${displayKind(cell, failure)}"><span class="viz-cell__index">slot ${index}</span><span>${esc(cell.value)}</span>${cell.note ? `<small>${esc(cell.note)}</small>` : ""}</div>`).join("")}</div>`;
    }).join("");
    return `<p class="scene-primer"><b>地址与内容分开看：</b>边框单元代表已存在的物理槽；绿色只表示本轮有效写入，灰色 suffix/padding 不写也不清零，靠长度保持不可见。</p><div class="viz-memory">${rows}</div>`;
  }

  function laneScene(step, failure) {
    const inputNodes = step.inputs.slice(0, 4).map((item) => `<div class="viz-lane-node">${esc(item.name)}<br>${esc(item.value)}</div>`).join("");
    const opNodes = step.calc.slice(0, 4).map((line) => `<div class="viz-lane-node" data-kind="op">${esc(line)}</div>`).join("");
    const outputNodes = step.outputs.slice(0, 4).map((item) => `<div class="viz-lane-node" data-kind="${failure ? "fail" : "write"}">${esc(item.name)}<br>${esc(failure ? "未正确提交" : item.value)}</div>`).join("");
    return `<p class="scene-primer"><b>沿时间向右读：</b>三条泳道分别展示旧状态、当轮操作与新状态；同一个字段名出现两次时，比较的是引用或内容在 Before/After 的变化。</p><div class="viz-lanes"><div class="viz-lane"><b>读取 / 上一轮</b><div class="viz-lane-track">${inputNodes}</div></div><div class="viz-lane"><b>当前执行</b><div class="viz-lane-track">${opNodes}</div></div><div class="viz-lane"><b>写入 / 下一轮</b><div class="viz-lane-track">${outputNodes}</div></div></div>`;
  }

  function renderScene(pageId, step, state, failure) {
    const kind = sceneKind(pageId, step);
    const renderers = {
      matrix: () => matrixScene(step, failure),
      flow: () => flowScene(step, failure),
      markov: () => markovScene(step, failure),
      rank: () => rankScene(step, failure),
      probability: () => probabilityScene(step, state, failure),
      memory: () => memoryScene(step, failure),
      lanes: () => laneScene(step, failure)
    };
    const names = { matrix: "矩阵 / 槽位映射", flow: "张量算子图", markov: "串行 token 依赖", rank: "数值排名与资源分配", probability: "概率质量与 CDF 抽样", memory: "静态地址与有效写入", lanes: "跨阶段状态泳道" };
    return { kind, title: names[kind], html: renderers[kind]() };
  }

  function setup() {
    const pageId = document.body.dataset.page;
    const page = pages[pageId];
    const lesson = lessons[pageId];
    const main = document.querySelector("[data-dspark-main]");
    const hero = main?.querySelector(".hero");
    if (!page || !lesson || !main || !hero) return;

    const generatedLessonSelector = ".lesson-outline,.lesson-conclusion,.lesson-lenses,.lesson-primer,.lesson-object-map,.lesson-example,.lesson-delta,.lesson-wfb,.lesson-evidence,.lesson-faq";
    const archiveNodes = [...main.children].filter((node) => node !== hero && !node.matches(`.pager,.site-footer,${generatedLessonSelector}`));
    const state = { step: initialStep(page.steps.length - 1), failure: false, timer: null, align: false, u: .90, skip: false };
    const studio = document.createElement("section");
    studio.className = "tensor-studio section lesson-source-main";
    studio.id = "lesson-source-main";
    studio.dataset.failure = "false";
    studio.innerHTML = `
      <div class="tensor-studio__head"><div><p>05 · 当前源码主链 · ${esc(page.index)}</p><h2>${esc(page.title)}</h2></div><div class="tensor-studio__sample"><b>本章固定教学数据</b>${esc(page.sample)}<br>承接 04 的样例，逐步执行当前源码；结果交给 06 的状态总表</div></div>
      <div class="tensor-subsection"><b>05.1 · 源码定位</b><span>先确认当前 receiver、方法和源码锚点</span></div>
      <div class="tensor-call-ribbon"><span class="tensor-call-ribbon__label">当前调用</span><code data-call></code><span class="tensor-call-ribbon__route" data-route></span></div>
      <div class="tensor-subsection"><b>05.2 · 选择执行步骤</b><span>从左向右形成同一数据的连续变换；播放不会跳过中间状态</span></div>
      <div class="tensor-stagebar"><div class="tensor-stage-tabs" role="tablist" aria-label="选择张量变化步骤">${page.steps.map((s,i) => `<button type="button" role="tab" data-step="${i}" data-no="${String(i+1).padStart(2,"0")}" aria-selected="${i===0}">${esc(s.label)}</button>`).join("")}</div><div class="tensor-controls"><button type="button" data-action="prev" aria-label="上一步">←</button><button type="button" data-action="play">播放</button><button type="button" data-action="next" aria-label="下一步">→</button><button type="button" data-failure aria-pressed="false">模拟省略/错误</button></div></div>
      <div class="tensor-stage-title"><div><div class="tensor-stage-title__kicker" data-progress></div><h3 data-title></h3></div><div class="tensor-stage-title__source" data-source></div></div>
      <section class="execution-figure" data-figure><div class="execution-figure__head"><b><em>05.3</em><span data-figure-title></span></b><span data-figure-kind></span></div><div class="execution-shapes" data-shapes></div><div class="execution-viewport"><div class="execution-scene" data-scene></div></div><div class="execution-caption"><div><b>发生了什么</b><span data-caption="change"></span></div><div><b>改变落在哪里</b><span data-caption="where"></span></div><div data-caption-box="failure"><b>省略 / 误算</b><span data-caption="failure"></span></div></div></section>
      <details class="tensor-evidence"><summary><b>05.7 · 展开逐字段源码证据</b><span>精确 shape / dtype / equation / Before / After</span></summary><div class="tensor-transform">
          <section class="tensor-column" data-role="read"><div class="tensor-column__head"><b>读取的张量 / 状态</b><span>改动前</span></div><div class="tensor-list" data-inputs></div></section>
          <section class="tensor-column" data-role="operation"><div class="tensor-column__head"><b>源码算子、索引与数值演算</b><span>执行中</span></div><pre class="tensor-equation" data-equation></pre><div class="tensor-calc-list" data-calc></div><p class="tensor-boundary" data-boundary></p></section>
          <section class="tensor-column" data-role="write"><div class="tensor-column__head"><b>写出的张量 / 持久状态</b><span>改动后</span></div><div class="tensor-list" data-outputs></div></section>
      </div></details>
      <section class="tensor-rails"><div class="tensor-rails__head"><b>05.4 · 逐槽位索引 / 内存可见性</b><div class="tensor-color-legend" aria-label="槽位颜色含义"><span data-kind="read"><i></i>读取</span><span data-kind="op"><i></i>计算 / guard</span><span data-kind="write"><i></i>真实写入</span><span data-kind="cuda"><i></i>设备执行</span><span data-kind="off"><i></i>padding / 不可见</span><span data-kind="fail"><i></i>错误</span></div></div><div class="tensor-rail-scroll" data-rails></div></section>
      <div class="tensor-ledger"><div class="tensor-ledger__title"><b>05.5 · 当前变更对象</b><span data-ledger="field"></span></div><div><b>改动前</b><span data-ledger="before"></span></div><div><b>改动后</b><span data-ledger="after"></span></div><div><b>存活范围</b><span data-ledger="life"></span></div><div><b>下一消费者</b><span data-ledger="next"></span></div></div>
      <div class="tensor-proof"><div><b>05.6 · 为什么这样算 / 写</b><p data-why></p></div><div data-proof="failure"><b>若省略或误算，最先坏在哪里</b><p data-fail></p></div></div>
      <div data-special></div>
      <div class="tensor-source-handoff"><b>本章交出</b><span>把本步的 Tensor、字段和失败后果交给 06 Before/After 汇总，再由 07–09 解释、举证和自测。</span></div>`;

    const evidenceBlock = studio.querySelector(".tensor-evidence");
    const specialBlock = studio.querySelector("[data-special]");
    if (evidenceBlock && specialBlock) specialBlock.after(evidenceBlock);

    const sourceInsertion = main.querySelector(".lesson-example") || hero;
    sourceInsertion.after(studio);
    const archiveDetails = document.createElement("details");
    archiveDetails.className = "studio-archive";
    archiveDetails.innerHTML = `<summary><b>展开编号补图与完整调用材料</b><span>补图用于拓展 03/05/06；每张图标明它关联的主线章节</span></summary><div class="studio-archive__body"></div>`;
    const archive = archiveDetails.querySelector(".studio-archive__body");
    let supplementIndex = 0;
    archiveNodes.forEach((node) => {
      if (node.matches(".section")) {
        supplementIndex += 1;
        const supplementNo = String(supplementIndex).padStart(2, "0");
        const kicker = node.querySelector(":scope > .section__head .section__kicker");
        if (kicker) kicker.textContent = `补图 ${supplementNo} · ${kicker.textContent}`;
        node.querySelectorAll(".visual__bar strong").forEach((label, visualIndex) => {
          label.textContent = `补图 ${supplementNo}.${visualIndex + 1} · ${label.textContent}`;
        });
        const relationMap = {
          architecture: ["lesson-objects", "03 组件/对象"],
          routes: ["lesson-source-main", "05 源码主链"],
          lifecycle: ["lesson-delta", "06 Before/After"],
          questions: ["lesson-faq", "09 FAQ/自测"],
          stages: ["lesson-source-main", "05 源码主链"]
        };
        const [relationTarget, relationLabel] = relationMap[node.id] || ["lesson-source-main", "05 源码主链"];
        const relation = document.createElement("a");
        relation.className = "supplement-relation";
        relation.href = `#${relationTarget}`;
        relation.textContent = `↖ 关联主线 ${relationLabel}`;
        node.querySelector(":scope > .section__head")?.append(relation);
      }
      archive.append(node);
    });
    const archiveInsertion = main.querySelector(".lesson-faq") || studio;
    archiveInsertion.after(archiveDetails);
    const chapterSpec = [
      ["lesson-conclusion", "01", "直接结论", "先知道这一页最终要建立什么结论"],
      ["lesson-intuition", "02", "前置直觉", "用三个心智支点约束后续理解"],
      ["lesson-objects", "03", "组件 / 对象地图", "把直觉落到实际实例、所有权和消费者"],
      ["lesson-example", "04", "具体样例", "给对象代入固定数据，形成可追踪现场"],
      ["lesson-source-main", "05", "当前源码主链", "沿源码重放同一组数据的计算与写入"],
      ["lesson-delta", "06", "Before / After", "汇总哪些字段改变、哪些 backing 仍不可见"],
      ["lesson-wfb", "07", "Why / Failure / Boundary", "解释为何必须如此以及省略后的首个故障点"],
      ["lesson-evidence", "08", "源码证据", "用当前 checkout 的 action、write 和 consumer 复核"],
      ["lesson-faq", "09", "FAQ / 自测", "确认能否独立推导本页结论"]
    ];
    const chapterNodes = chapterSpec.map(([id]) => document.getElementById(id)).filter(Boolean);
    chapterSpec.forEach(([id, number, title, purpose], index) => {
      const node = document.getElementById(id);
      if (!node) return;
      node.dataset.chapter = number;
      const prev = chapterSpec[index - 1];
      const next = chapterSpec[index + 1];
      const path = document.createElement("nav");
      path.className = "chapter-handoff";
      path.setAttribute("aria-label", `${number} ${title} 与相邻章节的关系`);
      path.innerHTML = `${prev ? `<a href="#${prev[0]}"><small>上一章</small><b>← ${prev[1]} ${prev[2]}</b></a>` : `<span class="chapter-handoff__edge"><small>起点</small><b>先读直接结论</b></span>`}<span class="chapter-handoff__current"><small>当前 ${number}</small><b>${esc(title)}</b><em>${esc(purpose)}</em></span>${next ? `<a href="#${next[0]}"><small>下一章</small><b>${next[1]} ${next[2]} →</b></a>` : `<span class="chapter-handoff__edge"><small>终点</small><b>完成本页自测</b></span>`}`;
      node.after(path);
    });
    const outline = main.querySelector(".lesson-outline");
    if (outline && "IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        outline.querySelectorAll("a").forEach((link) => link.toggleAttribute("aria-current", link.getAttribute("href") === `#${visible.target.id}`));
      }, { rootMargin: "-18% 0px -66% 0px", threshold: [0, .12, .35] });
      chapterNodes.forEach((node) => observer.observe(node));
    }
    document.body.dataset.studioReady = "true";
    main.closest(".dspark-lab").dataset.studioReady = "true";

    function stop() {
      if (state.timer) clearInterval(state.timer);
      state.timer = null;
      studio.querySelector("[data-action='play']").textContent = "播放";
    }

    function render(writeHash = true) {
      const raw = page.steps[state.step];
      const step = dynamicStep(pageId, raw, state);
      const effectiveFailure = state.failure || (pageId === "08A" && state.skip && step.key === "begin");
      studio.dataset.failure = String(effectiveFailure);
      studio.querySelectorAll("[data-step]").forEach((button) => {
        const i = Number(button.dataset.step);
        button.setAttribute("aria-selected", String(i === state.step));
        button.dataset.past = String(i < state.step);
      });
      studio.querySelector("[data-call]").textContent = `${step.call}  ←  ${step.source}`;
      studio.querySelector("[data-route]").textContent = routeNames[window.DSparkSite?.route || "compact-sampling"] || "当前路线";
      studio.querySelector("[data-progress]").textContent = `步骤 ${state.step + 1} / ${page.steps.length} · ${step.label}`;
      studio.querySelector("[data-action='prev']").disabled = state.step === 0;
      studio.querySelector("[data-action='next']").disabled = state.step === page.steps.length - 1;
      studio.querySelector("[data-action='play']").disabled = page.steps.length < 2;
      studio.querySelector("[data-title]").textContent = step.title;
      studio.querySelector("[data-source]").textContent = step.source;
      const scene = renderScene(pageId, step, state, effectiveFailure);
      studio.querySelector("[data-figure-title]").textContent = scene.title;
      studio.querySelector("[data-figure-kind]").textContent = `图形执行舞台 · ${step.label}`;
      studio.querySelector("[data-shapes]").innerHTML = shapeMarkup(step);
      studio.querySelector("[data-scene]").innerHTML = scene.html;
      studio.querySelector("[data-caption='change']").textContent = `${step.ledger[1]} → ${effectiveFailure ? "错误/未产生" : step.ledger[2]}`;
      studio.querySelector("[data-caption='where']").textContent = `${step.ledger[0]} · ${step.ledger[3]} · ${step.ledger[4]}`;
      studio.querySelector("[data-caption='failure']").textContent = step.failure;
      const figure = studio.querySelector("[data-figure]");
      figure.classList.remove("is-animating");
      void figure.offsetWidth;
      if (!reduceMotion) figure.classList.add("is-animating");
      studio.querySelector("[data-inputs]").innerHTML = tensorMarkup(step.inputs, false);
      studio.querySelector("[data-outputs]").innerHTML = tensorMarkup(step.outputs, effectiveFailure);
      studio.querySelector("[data-equation]").textContent = effectiveFailure ? `# 此步骤被省略/误算\n${step.equation}` : step.equation;
      studio.querySelector("[data-calc]").innerHTML = step.calc.map((line,i) => `<div class="tensor-calc-row"><b>${i+1}</b><span>${esc(line)}</span></div>`).join("");
      studio.querySelector("[data-boundary]").textContent = step.boundary;
      const maxCells = Math.max(...step.rails.map((r) => r.cells.length), 1);
      studio.querySelector("[data-rails]").innerHTML = step.rails.map((rail) => `<div class="tensor-rail" style="--rail-cols:${maxCells}"><div class="tensor-rail__name">${esc(rail.name)}<small>${esc(rail.note)}</small></div>${rail.cells.concat(Array.from({length:maxCells-rail.cells.length},() => C("—","off"))).map((cell) => `<div class="tensor-cell" data-kind="${effectiveFailure && cell.kind === "write" ? "fail" : cell.kind}">${esc(cell.value)}${cell.note ? `<small>${esc(cell.note)}</small>` : ""}</div>`).join("")}</div>`).join("");
      ["field","before","after","life","next"].forEach((key,i) => { studio.querySelector(`[data-ledger='${key}']`).textContent = effectiveFailure && key === "after" ? `错误/未写入：${step.ledger[i]}` : step.ledger[i]; });
      studio.querySelector("[data-why]").textContent = step.why;
      studio.querySelector("[data-fail]").textContent = effectiveFailure ? step.failure : `切换“模拟省略/错误”可查看：${step.failure}`;
      studio.querySelector("button[data-failure]").setAttribute("aria-pressed", String(state.failure));
      studio.querySelector("[data-special]").innerHTML = specialLab(pageId, state);
      if (writeHash) history.replaceState(null,"",`${location.pathname}${location.search}#step-${state.step}`);
    }

    function setStep(next) {
      state.step = Math.max(0, Math.min(page.steps.length - 1, next));
      render();
      if (state.step === page.steps.length - 1) stop();
    }

    studio.addEventListener("click", (event) => {
      const stepButton = event.target.closest("[data-step]");
      if (stepButton) { stop(); setStep(Number(stepButton.dataset.step)); return; }
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (action === "prev") { stop(); setStep(state.step - 1); }
      if (action === "next") { stop(); setStep(state.step + 1); }
      if (action === "play") {
        if (state.timer) { stop(); return; }
        if (reduceMotion) { setStep(state.step === page.steps.length - 1 ? 0 : state.step + 1); return; }
        if (state.step === page.steps.length - 1) state.step = 0;
        studio.querySelector("[data-action='play']").textContent = "暂停";
        render();
        state.timer = setInterval(() => setStep(state.step + 1), 2400);
      }
      if (event.target.closest("button[data-failure]")) { state.failure = !state.failure; render(); }
      if (event.target.closest("[data-align]")) { state.align = !state.align; render(); }
      if (event.target.closest("[data-skip]")) { state.skip = !state.skip; render(); }
    });

    studio.addEventListener("input", (event) => {
      if (!event.target.matches("[data-u]")) return;
      state.u = Number(event.target.value);
      render(false);
    });
    addEventListener("dspark:route", () => render(false));
    render(false);
  }

  setup();
})();
