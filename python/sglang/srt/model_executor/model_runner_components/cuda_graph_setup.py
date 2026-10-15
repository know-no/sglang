from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    prealloc_symmetric_memory_pool,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import XPUGraphRunner
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner import (
    EagerRunner,
    PrefillCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_loader.utils import resolve_language_model
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

# Measured immediately before prefill graph construction, after model weights,
# KV cache, and the eager runner's static buffers have been allocated.  Below
# this budget, compiling/capturing a multi-bucket prefill graph is likely to
# OOM or make no forward progress for large models.  Keep an explicitly chosen
# backend untouched: an operator may intentionally trade KV capacity for it.
_MIN_AUTO_PREFILL_CUDA_GRAPH_FREE_MEMORY_GB = 4.0


def should_skip_auto_prefill_cuda_graph_for_memory(
    available_memory_gb: float,
    cuda_graph_config_locked: set[tuple[str, str]],
) -> bool:
    """Return whether an auto-selected prefill graph lacks capture headroom."""
    return (
        (Phase.PREFILL, "backend") not in cuda_graph_config_locked
        and available_memory_gb < _MIN_AUTO_PREFILL_CUDA_GRAPH_FREE_MEMORY_GB
    )


class DecodeGraphCapture(msgspec.Struct, frozen=True, kw_only=True):
    runner: Optional[BaseRunner]
    graph_mem_usage: float


class CudaGraphsCapture(msgspec.Struct, frozen=True, kw_only=True):
    eager_runner: EagerRunner
    prefill_runner: Optional[BaseRunner]
    decode: DecodeGraphCapture
#    | 路径 | 档位主要表示什么 | 捕获时的请求数 | 普通 attention |
#| Prefill BCG | 总 token 数 `B` | dummy `bs=1` | 在图段之间 eager 执行，使用真实请求边界 |
def capture_cuda_graphs(
    *, model_runner: ModelRunner, capture_decode_cuda_graph: bool = True
) -> CudaGraphsCapture:
    """Capture cuda graphs. Requires init_attention_backends() to have run.
#| Prefill Full | 总 token 数 `B` | 固定请求容量 `R` | 捕获进图，真实请求数必须 `≤R` |
    Spec draft runners pass capture_decode_cuda_graph=False
    because they capture their own decode-style graphs separately.
#| **Decode Full** | **请求数档位 `C`** | **`bs=C`** | 捕获进图 |
    """
#| **Decode BCG** | **同样是请求数档位 `C`** | **同样是 `bs=C`** | **不会因为选择 BCG 就自动移到图外** |
    model_runner.graph_shared_output = GraphSharedOutput.create_for_model_runner(
        model_runner
    )

    # The eager (no-cuda-graph) phase runner, built AFTER the attention
    # backend so its __init__ can warm up kernels (run-once) and allocate the
    # fixed-max static buffer — both before the cuda-graph runners, so that
    # buffer is canonical in the shared pool and the cg runners coalesce onto
    # it. Always built: it serves both the fully-disabled case (decode/prefill
    # runners point at it) and the eager fallback when a cg runner can't run a
    # batch.
    eager_runner = EagerRunner(model_runner)

    # cuda-graph capture: prefill before decode, so both coalesce onto the
    # eager buffer allocated above. (capture_prefill_graph routes prefill
    # to the eager runner when the prefill graph is disabled.)
    prefill_runner = capture_prefill_graph(
        model_runner=model_runner, eager_runner=eager_runner
    )

    decode = DecodeGraphCapture(runner=None, graph_mem_usage=0)
    if capture_decode_cuda_graph:
        if model_runner.device in ("cuda", "musa", "cpu", "npu", "xpu"):
            decode = capture_decode_graph(model_runner=model_runner)
        elif (
            current_platform.is_out_of_tree() and current_platform.support_cuda_graph()
        ):
            decode = capture_decode_graph(model_runner=model_runner)
    else:
        decode = DecodeGraphCapture(runner=eager_runner, graph_mem_usage=0)

    # Register forward hooks AFTER cuda-graph capture so their tensor ops are
    # not traced into any captured graph — capture stays hook-free and hooks
    # fire only on the eager forward path (capture replay never runs Python
    # hooks anyway).
    if model_runner.server_args.forward_hooks:
        register_forward_hooks(
            model_runner.model, model_runner.server_args.forward_hooks
        )

    prealloc_symmetric_memory_pool(
        is_draft_worker=model_runner.is_draft_worker,
        enable_symm_mem=model_runner.server_args.enable_symm_mem,
        device=model_runner.device,
        forward_stream=model_runner.forward_stream,
    )
# Canary 是可选的 KV 正确性检查机制；默认 kv_canary="none" 时 manager 为 None，跳过此分支。
    if model_runner.canary_manager is not None and not model_runner.is_draft_worker:
        model_runner.canary_manager.mark_init_finished() # 这里只标记非 Draft；EAGLE Draft 等专用路径会在自身捕获结束后另行调用 mark_init_finished()。
# 此处启用 Target Canary 的调用阶段断言和 KV 链 position 断言，不是直到这里才创建全部检查逻辑。
    return CudaGraphsCapture(
        eager_runner=eager_runner, prefill_runner=prefill_runner, decode=decode
    )


def capture_prefill_graph(
    *,
    model_runner: ModelRunner,
    eager_runner: EagerRunner,
    force_for_draft_worker: bool = False,
) -> Optional[BaseRunner]:
    """Initialize prefill CUDA graph runner."""

    if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
        logger.info(
            "Disable prefill CUDA graph because cuda_graph_config "
            "resolved prefill.backend='disabled' (e.g. via "
            "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
        )
        # Prefill cuda graph disabled: route eager prefill through the
        # EagerRunner (its can_run_graph returns False, so _forward_raw's
        # extend branch falls through to the eager path).
        if not model_runner.is_draft_worker:
            return eager_runner
        return None

    # Draft models skip here during __init__; the eagle worker calls
    # this method explicitly (force_for_draft_worker=True) after
    # init_lm_head so graphs capture the final embedding weights.
    if model_runner.is_draft_worker and not force_for_draft_worker:
        return None

    # Skip prefill CG for EAGLE target on tc_piecewise: that backend
    # captures CaptureHiddenMode.NULL while runtime requests FULL, so
    # the captured graph is dead, and capturing it perturbs FP4 /
    # TRTLLM-MoE state and corrupts decode replay (see #28386). BCG
    # captures FULL for EAGLE target in PrefillCudaGraphRunner.__init__
    # (restored from #25795), so it does NOT need this skip.
    if (
        model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and not model_runner.server_args.enable_return_hidden_states
        and not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    ):
        logger.info(
            "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
            "to avoid FP4/MoE decode-replay corruption (#28386)."
        )
        return eager_runner

    if (
        model_runner.server_args.enable_lora
        and not model_runner.lora_manager.supports_prefill_cuda_graph
    ):
        logger.warning(
            "Disable prefill CUDA graph because the current LoRA "
            "configuration does not support it (unsupported LoRA backend, "
            "MoE LoRA, or DP attention)."
        )
        return eager_runner

    # Resolve the decoder once. Some VLM wrappers (for example Kimi-VL)
    # expose it as ``language_model`` rather than ``model``.
    try:
        language_model = resolve_language_model(model_runner.model)
    except AttributeError:
        logger.warning(
            "Disable prefill CUDA graph because the model is not a language model"
        )
        return None

    # Disable prefill CUDA graph for non capture size
    if not model_runner.server_args.cuda_graph_config.prefill.bs:
        logger.warning("Disable prefill CUDA graph because the capture size is not set")
        return None

    # Collect attention layers and moe layers from the model. Keep a VLM
    # wrapper that exposes ``language_model`` unchanged: assigning it to
    # ``model`` would register a duplicate module alias and duplicate the
    # model's state-dict namespace.
    if hasattr(model_runner.model, "model"):
        model_runner.model.model = language_model

    # Find the module that owns the decoder `layers`. Models wrap it at
    # varying depths: a direct text model exposes `.layers`, a CausalLM
    # wraps it as `.model.layers`, and some multimodal models add another
    # level (e.g. DeepSeek-OCR: OCR wrapper -> Deepseek*ForCausalLM ->
    # text model -> `.layers`). Descend the `.model` chain until we find it.
    layer_model = language_model
    while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
        layer_model = layer_model.model

    if not hasattr(layer_model, "layers"):
        logger.warning(
            "Disable prefill CUDA graph because the model does not have a 'layers' attribute"
        )
        return None

    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = compute_attention_and_moe_layers(layer_model)

    if len(model_runner.attention_layers) < model_runner.model_config.num_hidden_layers:
        # TODO(yuwei): support Non-Standard GQA
        log_info_on_rank0(
            logger,
            "Disable prefill CUDA graph because some layers do not apply Standard GQA",
        )
        return None

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    prefill_backend = model_runner.server_args.cuda_graph_config.prefill.backend
    if should_skip_auto_prefill_cuda_graph_for_memory(
        before_mem,
        getattr(model_runner.server_args, "_cuda_graph_config_locked", set()),
    ):
        logger.warning(
            "Disabling auto-selected prefill CUDA graph: only %.2f GiB is free "
            "after model/KV/eager-buffer allocation; at least %.2f GiB is "
            "required for capture. Set an explicit prefill CUDA graph backend "
            "to override this safety gate.",
            before_mem,
            _MIN_AUTO_PREFILL_CUDA_GRAPH_FREE_MEMORY_GB,
        )
        return eager_runner

    role = "draft" if model_runner.is_draft_worker else "target"
    capture_name = f"{role} prefill"
    capture_num_tokens = sorted(model_runner.server_args.cuda_graph_config.prefill.bs)
    logger.info(
        f"Capture {capture_name} CUDA graph begin. "
        f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
        f"avail mem={before_mem:.2f} GB"
    )

    prefill_runner = PrefillCudaGraphRunner(model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {capture_name} CUDA graph end. "
        f"elapsed={time.perf_counter() - tic:.2f} s, "
        f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return prefill_runner


def capture_decode_graph(*, model_runner: ModelRunner) -> DecodeGraphCapture:
    """Capture device graphs."""
    no_capture = DecodeGraphCapture(runner=None, graph_mem_usage=0)

    if not model_runner.is_generation:
        # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
        return no_capture
    if model_runner.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
        return no_capture
    if model_runner.device != "cpu" and check_cuda_graph_backend(
        Phase.DECODE, Backend.DISABLED
    ):
        return no_capture
    if model_runner.device == "cpu" and not get_flags().capture.enable_torch_compile:
        return no_capture

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_backend = defaultdict(
        lambda: f"{current_platform.device_name} graph",
        {
            "cuda": "CUDA graph",
            "musa": "CUDA graph",
            "cpu": "CPU graph",
            "npu": "NPU graph",
            "xpu": "XPU graph",
        },
    )
    role = "draft" if model_runner.is_draft_worker else "target" # 非 Draft 的主模型统一记作 target，包括非投机的普通模型；身份来自 ModelRunner.is_draft_worker。
    if model_runner.spec_algorithm.is_speculative(): # 这是通用 decode 图的投机分支；dSpark 的 Target 和 Draft 都会进入。
        capture_name = f"{role} verify"              # draft verify 是这里的日志名；dSpark Draft 复用 TARGET_VERIFY 执行模式生成 proposal，并非替 Target 验收。
        num_tokens_per_req = model_runner.decode_num_tokens_per_req() # dSpark：令 N=speculative_num_draft_tokens，则 Target 宽度为 N，Draft 为 N-1（即 gamma）；见 spec_info.py 的角色分支。
    else:
        capture_name = f"{role} decode"
        num_tokens_per_req = 1
    capture_bs, _ = get_batch_sizes_to_capture(model_runner, num_tokens_per_req)
    decode_backend = model_runner.server_args.cuda_graph_config.decode.backend
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} begin. "
        f"backend={decode_backend}, num_tokens_per_req={num_tokens_per_req}, "
        f"bs={capture_bs}, avail mem={before_mem:.2f} GB"
    )

    if current_platform.is_out_of_tree():
        GraphRunnerCls = current_platform.get_graph_runner_cls()
        runner = GraphRunnerCls(model_runner)
    else:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        graph_runners = defaultdict(
            lambda: DecodeCudaGraphRunner, # 非 OOT 分支中，设备名未命中显式映射时返回此类，再在下方实例化；常规 CUDA/ROCm 走这里。
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
                "xpu": XPUGraphRunner,
            },
        )
        runner = graph_runners[model_runner.device](model_runner)

    after_mem = get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {capture_name} {graph_backend[model_runner.device]} end. "
        f"elapsed={time.perf_counter() - tic:.2f} s, "
        f"mem usage={graph_mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return DecodeGraphCapture(runner=runner, graph_mem_usage=graph_mem_usage)
