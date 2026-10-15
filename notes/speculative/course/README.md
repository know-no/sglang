# 投机解码资料生成源

这是保留下来的投机解码资料的维护目录，不是额外学习路线。

- `content/`：8 章家族课程 Markdown 源文；章节与索引由 `course.yaml` 维护。
- `templates/`、`assets/`：课程、深入笔记与实验共用的页面外壳、本地脚本、样式和数学渲染资源。
- `../visualizations/sources/*.html.jinja`：15 个 dSpark 实验源文；同级 `assets/` 维护交互数据。
- `../*.md`、`../visualizations/README.md`：投机主文和两个阅读入口。

重建：

```bash
python3 -m pip install -r notes/speculative/course/requirements.txt
python3 notes/speculative/course/build.py
python3 notes/speculative/course/tools/verify_preserved.py
```

输出仍在 `notes/speculative/` 与 `notes/speculative/visualizations/` 原有位置；浏览器入口为[投机资料](../index.html)。课程源文改动应通过重建更新 HTML。普通 Markdown 阅读链接自动进入渲染页；页面上的“下载 Markdown 源文”保留下载功能。

只分享投机材料时保留整个 `notes/speculative/` 目录。站内图表、主题切换和实验无需联网；返回 SGLang 全景图需要上级 `notes/index.html`，源码定位链接需要仓库代码。源文保留原核对日期，重建不代表已重新执行 GPU 或端到端实验。
