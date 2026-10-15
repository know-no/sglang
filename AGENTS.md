# Project Agent Instructions

## SGLang source-study additions

The global source-study rules in `/Users/magician/.codex/AGENTS.md` apply. Add these
SGLang-specific checks:

1. In scheduler explanations, explicitly distinguish `Req`, accepted request lists,
   `ScheduleBatch`, `batch_to_run`, `running_batch`, and `last_batch`, and name the
   exact `Scheduler` field or queue that receives a state transition.
2. Always check speculative decoding, including dFlash, dSpark, and EAGLE, plus the
   relevant DP/PP/CP and Mixed-execution boundary; do not infer one algorithm's
   behavior from the ordinary decode path.
3. The user reset the teaching collection on 2026-09-10 to reduce information
   overload. Cross-subsystem reading starts at the comprehensive module-and-topic
   panorama in `notes/index.html`, linked from `notes/README.md`. Keep important
   modules and topic names visible and searchable, with source, documentation and
   related-module links; do not limit this map to the current lesson. Preserve the
   speculative collection; add other teaching notes only when requested for the topic being
   studied. Execution, attention, I/O and distributed content does not belong under
   Scheduler. Treat an audited primary note as authoritative over comparisons.
4. Before presenting a runtime call chain, state the assumed hardware backend and
   verify the worker/backend dispatch that selects it. At minimum, distinguish the
   generic CUDA/ROCm `TpModelWorker` path from the MLX override when `use_mlx()` can
   change the callee; mention NPU/XPU or another backend when it changes the current
   conclusion. An IDE jump to an override is not proof of the runtime branch.
5. Inside `ModelRunner`, do not collapse execution into a vague `forward()` step.
   Check and name the selected runner boundary: eager execution, decode CUDA Graph,
   prefill CUDA Graph, or split prefill. State where backend-specific attention
   metadata is planned on that branch, and re-check speculative pre-planning through
   `ForwardBatch.needs_forward_metadata_init()`.
6. Choose the runtime-diagram form that best exposes the SGLang flow being taught;
   a compact call tree may be clearer for nested dispatch, while a sequence diagram,
   swimlane, or data-flow view may be clearer for ownership, concurrency, or tensors.
   When the selected view crosses model-execution boundaries, annotate the decisive
   receiver/instance and concrete class for the worker, `ModelRunner`, graph/eager
   runner, attention backend, or speculative-input object where that identity matters;
   label material calls, mode/configuration guards, and handoffs/returns involving
   `ScheduleBatch`, `ForwardBatch`, attention metadata, logits, or hidden states.
   Split only runtime targets relevant to the stated backend/configuration instead of
   expanding unrelated implementations or forcing every node into one label template.
7. When a source lesson crosses methods, objects, processes, threads, or device
   streams, supplement the basic control/data-flow diagram with the most useful
   additional view: a swimlane call-flow diagram and/or a sequence diagram. Label
   lanes by the actual component, process, CPU thread, or CUDA stream; distinguish a
   Python call from asynchronously enqueued device work; and show decisive returns,
   state writes, waits, events, and synchronization points. Avoid redundant diagrams
   for a trivial linear call, but prefer multiple complementary views when they make
   ownership, concurrency, or timing materially easier to understand.
8. Treat `notes/speculative/visualizations/` as the durable delivery
   directory for this project's speculative-decoding HTML visualizations. Temporary
   or inline versions may be developed in Codex-owned space, but copy or overwrite
   the corresponding standalone HTML in that project directory before handoff so it
   can be opened independently from other tools.

9. The former full course in `notes/course/` has been retired. Retained speculative
   course authoring lives in `notes/speculative/course/`; follow its build entry
   point to regenerate `notes/speculative/visualizations/sglang-course-v2/` and
   related reading pages. Update source and rebuild rather than maintaining
   generated HTML as separate teaching prose. Old speculative-path redirects are
   compatibility entries, not additional lessons. Do not recreate the retired full
   course or its chapter-count navigation while maintaining the retained material.
