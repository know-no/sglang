# 投机解码交互学习入口

[笔记总入口](../../index.html) · [投机笔记](../README.md) · [浏览器入口](index.html)

## 选择阅读路线

| 想解决的问题 | 从这里进入 | 接下来 |
|---|---|---|
| 先理解 proposal / verify / accept 的共同合同 | [全家族课程](sglang-course-v2/index.html) | 分别进入 EAGLE、dFlash、dSpark 和边界矩阵 |
| 追踪 dSpark 的具体对象、张量和下一轮状态 | [dSpark 完整执行地图](dspark-complete-execution-flow.html) | 按 00–08 主阶段和就近专题逐步播放 |
| 核对 compact 行、accept lane 与 rolling anchor | [Target Verify](dspark-verify-step-through.html) → [Accept](dspark-accept-finalize.html) → [Hidden / KV](dspark-target-hidden-kv-explainer.html) | 对照 [Verify 笔记](../sglang-dspark-verify-visual-guide.md) |
| 比较调度时序或 CUDA Graph | [Overlap 时间线](dspark-overlap-nonoverlap-timeline.html) · [Folded capture/replay](dspark-folded-capture-replay.html) | 对照 [dSpark 主文](../sglang-dspark-single-round-decode.md) |

## 内容在哪维护

| 页面类型 | 内容源 | 维护方法 |
|---|---|---|
| dSpark 15 个深入页面 | `sources/*.html.jinja`、`assets/dspark-learning-layer.js`、`assets/dspark-source-studio.js` | 页面补图、九章教学数据和张量显微镜一起核对；`build_lab_shell.py` 生成同级 HTML，复用课程外壳；实验内导航在 `assets/dspark-site.js` |
| 全家族课程 `sglang-course-v2/` | [投机课程源](../course/README.md)的 content、course.yaml 与模板 | 执行 `python3 notes/speculative/course/build.py`；避免手工维护生成正文 |
| 本入口与各主题导航 | 对应 README.md | 执行 `python3 notes/speculative/course/build.py` 生成浏览器入口 |
| 2026-08-15 legacy | `legacy/2026-08-15/` | 历史快照，不作为当前教学主版本 |

两套现行页面具有不同阅读职责：家族课程串联算法和跨阶段实验；深入站提供 dSpark 分阶段推演。相同源码结论链接对应主文核对；使用真实 q 的采样模型与实际实现的 sampling 输入需要明确区分。

深入站与家族课程的目录和搜索仅包含投机解码内容，共用明暗主题。离线分享时保留整个 `notes/speculative/` 目录（包含实验 `assets/`、`course/assets/` 与家族课程的本地搜索数据）；返回全景图需要上级 `notes/index.html`。所有教学页面保留自己的核对快照；浏览器播放成功不能证明 GPU 执行结果。

## 历史快照

仅供比较：[2026-08-15 dSpark 地图](legacy/2026-08-15/dspark-complete-execution-flow.html)。新入口不把 legacy 纳入默认下一页路线。
