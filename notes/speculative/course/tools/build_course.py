#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from markdown_it import MarkdownIt

from course_lib import (
    COURSE_ROOT,
    SITE_ROOT,
    REPO_ROOT,
    related_notes,
    chapter_links,
    ChapterLocation,
    extract_plain_text,
    flatten_chapters,
    load_course,
    parse_frontmatter,
    rel_url,
    slugify,
)


TEACHING_STAGE_SPECS = {
    "直接结论": {
        "kind": "verdict",
        "index": "01",
        "kicker": "先钉住结论",
        "guide": "先用一段连续文字回答本章到底发生了什么；后面的图和源码只负责证明它。",
        "roadmap": True,
    },
    "当前路径假设": {
        "kind": "prereq",
        "index": "02",
        "kicker": "前置直觉与运行坐标",
        "guide": "明确硬件、模式与暂时关闭的分支，避免把局部主线误读成全称结论。",
        "roadmap": True,
    },
    "输入、本页职责与输出": {
        "kind": "contract",
        "index": "03",
        "kicker": "组件合同",
        "guide": "先分清输入、拥有的状态、输出和下一消费者，再进入类名与调用细节。",
        "roadmap": False,
    },
    "对象与状态": {
        "kind": "objects",
        "index": "04",
        "kicker": "对象与所有权地图",
        "guide": "沿对象身份、持久字段和生命周期理解这一层真正拥有的状态。",
        "roadmap": True,
    },
    "连续例子": {
        "kind": "example",
        "index": "05",
        "kicker": "固定样例 · 连续过程",
        "guide": "围绕同一个请求、batch、slot、rank 或 tensor 连续推进，不用一轮快照代替过程。",
        "roadmap": True,
    },
    "当前源码调用链": {
        "kind": "source-flow",
        "index": "06",
        "kicker": "当前源码主链",
        "guide": "从真实 receiver 追到关键状态写入、返回对象与下一消费者。",
        "roadmap": True,
    },
    "为什么存在": {
        "kind": "why",
        "index": "07",
        "kicker": "为什么需要这一层",
        "guide": "解释它解决的约束，而不是把类名重新翻译一遍。",
        "roadmap": False,
    },
    "为什么必须在这里": {
        "kind": "why-here",
        "index": "08",
        "kicker": "时机、失败与权衡",
        "guide": "说明提前、延后或省略时，究竟会出现错误、陈旧状态、重复工作还是性能退化。",
        "roadmap": False,
    },
    "边界矩阵": {
        "kind": "boundary",
        "index": "09",
        "kicker": "分支与结论边界",
        "guide": "逐项标出哪些 guard 会改变当前结论，以及证据在哪一层停止。",
        "roadmap": True,
    },
    "source action / changed state / why": {
        "kind": "recap",
        "index": "10",
        "kicker": "Before / After 状态账本",
        "guide": "把源码动作、被改写的状态和必须这样做的原因放在同一条因果线上。",
        "roadmap": False,
    },
    "自测": {
        "kind": "self-check",
        "index": "13",
        "kicker": "自测与复盘",
        "guide": "不用背术语；尝试从 owner、state、time 和 next consumer 四个坐标重新推演。",
        "roadmap": True,
    },
}



DIAGRAM_KIND_SPECS = {
    "architecture": {
        "label": "架构 / 边界图",
        "guide": "先定子系统边界与 owner，再读入口、出口和跨边界 handoff。",
        "legend": (("blue", "入口 / 控制"), ("orange", "调度 / guard"), ("purple", "设备执行"), ("cyan", "跨进程 / 分布式")),
    },
    "component": {
        "label": "组件 / 所有权图",
        "guide": "先认实例与持久状态的 owner，再沿组合、复用和下一消费者读边。",
        "legend": (("blue", "入口 / 接口"), ("orange", "owner / 调度"), ("green", "持久状态"), ("purple", "执行组件")),
    },
    "sequence": {
        "label": "时序 / 泳道图",
        "guide": "从上到下区分 Python 调用、CPU 状态写入、异步 device enqueue、wait 与 return。",
        "legend": (("blue", "caller"), ("orange", "scheduler"), ("purple", "device work"), ("cyan", "IPC / collective")),
    },
    "state": {
        "label": "状态迁移图",
        "guide": "沿 guard 阅读状态迁移；每条边都应能回答谁改了什么、下一轮谁读取。",
        "legend": (("blue", "当前状态"), ("orange", "guard"), ("green", "已提交状态"), ("red", "失败 / 回收")),
    },
    "data": {
        "label": "数据 / Tensor 流",
        "guide": "沿对象或 tensor 身份、shape/索引和存储位置读，不把控制流误当数据复制。",
        "legend": (("blue", "输入对象"), ("green", "KV / tensor"), ("purple", "计算"), ("cyan", "传输")),
    },
    "code": {
        "label": "源码流程图",
        "guide": "沿 receiver、guard、callee、return、状态写入和下一消费者逐跳核对当前源码。",
        "legend": (("blue", "entry / return"), ("orange", "guard / policy"), ("green", "state write"), ("red", "fail / cleanup")),
    },
    "generic": {
        "label": "源码关系图",
        "guide": "先认 owner，再沿 guard、状态写入和下一消费者读箭头。",
        "legend": (("blue", "入口"), ("orange", "控制"), ("green", "状态"), ("purple", "执行")),
    },
}


def configure_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "linkify": False}).enable("table")
    default_fence = md.renderer.rules["fence"]

    # CommonMark deliberately excludes pipe tables.  The course uses precise
    # state/rank/field matrices heavily, so render them as semantic tables and
    # give only the table its own horizontal overflow boundary.
    md.renderer.rules["table_open"] = lambda *_args: (
        '<div class="table-scroll teaching-table"><table>\n'
    )
    md.renderer.rules["table_close"] = lambda *_args: "</table></div>\n"

    def render_fence(tokens, idx, options, env):
        token = tokens[idx]
        fence_info = token.info.strip().split() if token.info.strip() else []
        language = fence_info[0] if fence_info else ""
        if language == "mermaid":
            diagram_kind = fence_info[1].lower() if len(fence_info) > 1 else "generic"
            if diagram_kind == "generic":
                mermaid_grammar = token.content.lstrip().splitlines()[0].strip()
                if mermaid_grammar.startswith("sequenceDiagram"):
                    diagram_kind = "sequence"
                elif mermaid_grammar.startswith("stateDiagram"):
                    diagram_kind = "state"
                elif mermaid_grammar.startswith("classDiagram"):
                    diagram_kind = "component"
                elif mermaid_grammar.startswith(("flowchart", "graph")):
                    diagram_kind = "code"
            kind_spec = DIAGRAM_KIND_SPECS.get(
                diagram_kind, DIAGRAM_KIND_SPECS["generic"]
            )
            escaped = (
                token.content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            label = env.get("diagram_label", "源码关系图")
            # Prefer the closest authored subheading over a repeated chapter/stage
            # label.  The figure should answer a specific question, not merely say
            # that it is "a sequence diagram".
            local_heading = ""
            for cursor in range(idx - 1, -1, -1):
                if tokens[cursor].type == "heading_open" and cursor + 1 < idx:
                    local_heading = tokens[cursor + 1].content.strip()
                    break
            local_context = local_heading
            if not local_context:
                for cursor in range(idx - 1, 0, -1):
                    if (
                        tokens[cursor].type == "inline"
                        and tokens[cursor - 1].type == "paragraph_open"
                    ):
                        local_context = re.sub(
                            r"[`*_\[\]#]", "", tokens[cursor].content
                        ).strip()
                        local_context = re.split(r"[。！？]", local_context, maxsplit=1)[0]
                        break
            if local_context:
                if len(local_context) > 72:
                    local_context = local_context[:70].rstrip() + "…"
                chapter_label = label.split(" · ", maxsplit=1)[0]
                label = f"{chapter_label} · {local_context}"
            legend = "".join(
                '<span class="diagram-legend__item" '
                f'data-tone="{html.escape(tone, quote=True)}">'
                f'{html.escape(text)}</span>'
                for tone, text in kind_spec.get("legend", ())
            )
            return (
                '<figure class="diagram-frame" data-diagram-frame '
                f'data-diagram-kind="{html.escape(diagram_kind, quote=True)}">'
                '<div class="diagram-toolbar">'
                '<span class="diagram-toolbar__identity">'
                f'<b>{html.escape(kind_spec["label"])}</b>'
                f'<span>{html.escape(label)}</span></span>'
                '<span class="diagram-toolbar__status" data-diagram-status>'
                '正在计算可读尺寸…</span>'
                '<button type="button" class="diagram-open" data-diagram-open>'
                '沉浸阅读</button></div>'
                f'<div class="diagram-legend" aria-label="图内语义色图例">{legend}</div>'
                '<div class="diagram-frame__viewport" data-diagram-content tabindex="0" '
                'aria-label="图形阅读区，可横向滚动">'
                '<div class="diagram-frame__canvas">'
                f'<pre class="mermaid" aria-label="{html.escape(label, quote=True)}">{escaped}</pre>'
                '</div>'
                '</div>'
                f'<figcaption><b>读图方法：</b>{html.escape(kind_spec["guide"])}</figcaption>'
                '</figure>\n'
            )
        return default_fence(tokens, idx, options, env)

    md.renderer.rules["fence"] = render_fence
    return md


def _plain_html(value: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _walkthrough_html(
    md: MarkdownIt,
    tokens: list[Any],
    *,
    diagram_label: str,
) -> tuple[str, str, int]:
    """Render explicit semantic panels instead of guessing from browser DOM.

    An authored step-lab is a nested, richer experiment and stays outside the
    outer narrative stepper.  H3 sections form one panel each; otherwise each
    top-level Markdown block is one panel.  The resulting count is the same
    count the browser consumes and the verifier audits.
    """
    narrative: list[Any] = []
    authored_labs: list[Any] = []
    for token in tokens:
        if token.type == "html_block" and re.search(
            r'class=["\'][^"\']*\bstep-lab\b', token.content
        ):
            authored_labs.append(token)
        else:
            narrative.append(token)

    h3_positions = [
        index
        for index, token in enumerate(narrative)
        if token.type == "heading_open" and token.tag == "h3" and token.level == 0
    ]
    groups: list[list[Any]] = []
    if h3_positions:
        if h3_positions[0] > 0:
            groups.append(narrative[:h3_positions[0]])
        for ordinal, start in enumerate(h3_positions):
            end = h3_positions[ordinal + 1] if ordinal + 1 < len(h3_positions) else len(narrative)
            groups.append(narrative[start:end])
    else:
        current: list[Any] = []
        depth = 0
        for token in narrative:
            current.append(token)
            depth += token.nesting
            if depth == 0:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

    panels: list[str] = []
    for index, group in enumerate(groups, start=1):
        rendered = md.renderer.render(
            group,
            md.options,
            {"diagram_label": diagram_label},
        )
        if not _plain_html(rendered):
            continue
        label = ""
        if group and group[0].type == "heading_open" and len(group) > 1:
            label = group[1].content.strip()
        label_attr = (
            f' data-walk-label="{html.escape(label, quote=True)}"' if label else ""
        )
        panels.append(
            f'<section class="sf-walk-panel" data-walk-panel="{len(panels) + 1}"{label_attr}>'
            f"{rendered}</section>"
        )

    extras = md.renderer.render(
        authored_labs,
        md.options,
        {"diagram_label": diagram_label},
    )
    return "".join(panels), extras, len(panels)


def _source_excerpt(ref: dict[str, Any]) -> tuple[str, str]:
    source = REPO_ROOT / ref["path"]
    source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    focus_line = int(ref["line"])
    start_line = max(1, focus_line - 4)
    end_line = min(len(source_lines), focus_line + 4)
    excerpt = "\n".join(
        f"{'>' if number == focus_line else ' '} {number:5d} | {source_lines[number - 1]}"
        for number in range(start_line, end_line + 1)
    )
    return f"{ref['path']}:{focus_line}", excerpt



def source_microscope_html(refs: list[dict[str, Any]]) -> str:
    if not refs:
        return ""
    buttons: list[str] = []
    panels: list[str] = []
    for index, ref in enumerate(refs, start=1):
        value = f"source-{index}"
        symbol = html.escape(str(ref.get("symbol", "")))
        note = html.escape(str(ref.get("note", "")))
        level = html.escape(str(ref.get("level", "SOURCE_CURRENT")))
        source_label, excerpt = _source_excerpt(ref)
        next_symbol = (
            html.escape(str(refs[index].get("symbol", "")))
            if index < len(refs)
            else "回到本章结论"
        )
        buttons.append(
            f'<button class="button button--quiet" data-source-step="{value}">'
            f'{index:02d} · {symbol}</button>'
        )
        panels.append(
            f'<section class="source-microscope__panel" data-source-panel="{value}">'
            '<div class="source-microscope__lenses">'
            f'<div class="source-lens" data-lens="what"><b>Symbol / state</b><p><code>{symbol}</code></p></div>'
            f'<div class="source-lens" data-lens="how"><b>这条证据证明什么</b><p>{note}</p></div>'
            f'<div class="source-lens" data-lens="why"><b>证据等级</b><p><span class="evidence-badge" data-evidence="{level}">{level}</span></p></div>'
            f'<div class="source-lens" data-lens="failure"><b>下一核对点</b><p><code>{next_symbol}</code></p></div>'
            '</div>'
            f'<div class="source-block"><div class="source-block__head"><span>{html.escape(source_label)}</span>'
            f'<span>步骤 {index} / {len(refs)}</span></div>'
            f'<pre class="source-excerpt" aria-label="{html.escape(source_label, quote=True)} 附近源码"><code>{html.escape(excerpt)}</code></pre></div>'
            '</section>'
        )
    return (
        '<div class="source-microscope" data-source-microscope data-source-key="source" data-default-step="source-1">'
        '<div class="source-microscope__head"><h3>源码显微镜：receiver、状态写入与下一消费者</h3>'
        '<p>左侧按本章论证顺序定位决定性源码；每一跳必须回到正文判断它是 call、write、return 还是 next-consumer。相邻卡片本身不冒充直接调用边。</p></div>'
        f'<div class="source-microscope__tabs" data-source-tabs>{"".join(buttons)}</div>'
        f'<div class="source-microscope__stage">{"".join(panels)}</div>'
        '<div class="source-microscope__toolbar"><span class="source-microscope__status" data-source-status aria-live="polite"></span>'
        '<div class="source-microscope__actions"><button class="button button--quiet" data-source-action="prev">上一跳</button>'
        '<button class="button" data-source-action="next">下一跳</button>'
        '<button class="button button--quiet" data-source-action="reset">回到入口</button></div></div></div>'
    )


def render_markdown(
    md: MarkdownIt,
    body: str,
    *,
    chapter_title: str = "",
    source_refs: list[dict[str, Any]] | None = None,
    chapter_urls: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokens = md.parse(body)
    # Author links target sibling Markdown files so the source is readable in an
    # IDE. Rewrite parsed links only; code examples and remote URLs stay literal.
    for token in tokens:
        for child in token.children or []:
            if child.type != "link_open":
                continue
            parts = urlsplit(child.attrGet("href") or "")
            if parts.scheme or parts.netloc:
                continue
            path = Path(unquote(parts.path)).as_posix()
            if path in (chapter_urls or {}):
                child.attrSet("href", urlunsplit((
                    "", "", chapter_urls[path], parts.query, parts.fragment
                )))
    token_toc: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1])
        if level not in (2, 3):
            continue
        inline = tokens[index + 1]
        title = inline.content.strip()
        base = slugify(title)
        used[base] = used.get(base, 0) + 1
        anchor = base if used[base] == 1 else f"{base}-{used[base]}"
        token.attrSet("id", anchor)
        token_toc.append({"id": anchor, "level": level, "text": title, "token": index})

    h2_positions = [
        index for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.tag == "h2"
    ]
    if not h2_positions:
        return [
            {
                "kind": "reference",
                "index": "R",
                "kicker": "参考内容",
                "guide": "按需要查询，不替代主线课程。",
                "title": chapter_title or "参考内容",
                "id": "reference-content",
                "body_html": md.renderer.render(tokens, md.options, {"diagram_label": chapter_title}),
                "roadmap": False,
            }
        ], []

    stages: list[dict[str, Any]] = []
    for ordinal, start in enumerate(h2_positions):
        end = h2_positions[ordinal + 1] if ordinal + 1 < len(h2_positions) else len(tokens)
        title = tokens[start + 1].content.strip()
        anchor = tokens[start].attrGet("id") or slugify(title)
        inner_start = start + 3
        body_tokens = tokens[inner_start:end]
        spec = TEACHING_STAGE_SPECS.get(title, {})
        kind = spec.get("kind", "reference")
        body_html = md.renderer.render(
            body_tokens,
            md.options,
            {"diagram_label": f"{chapter_title} · {title}".strip(" ·")},
        )
        stage = {
            "kind": kind,
            "index": f"{ordinal + 1:02d}",
            "kicker": spec.get("kicker", "参考坐标"),
            "guide": spec.get("guide", "围绕当前主题查询精确映射与使用边界。"),
            "title": title,
            "id": anchor,
            "body_html": body_html,
            "roadmap": spec.get("roadmap", False),
            "toc_items": [
                {key: value for key, value in item.items() if key != "token"}
                for item in token_toc
                if start <= item["token"] < end
            ],
        }
        if kind == "example" and not re.search(r'class=[\"\'][^\"\']*\bstep-lab\b', body_html):
            walk_html, walk_extras_html, walk_step_count = _walkthrough_html(
                md,
                body_tokens,
                diagram_label=f"{chapter_title} · {title}".strip(" ·"),
            )
            stage["walk_html"] = walk_html
            stage["walk_extras_html"] = walk_extras_html
            stage["process_step_count"] = walk_step_count
        stages.append(stage)

    # Keep the author's causal order. Evidence is a lookup after the lesson;
    # generated prose must not restate authored tables or fabricate FAQs.
    if source_refs:
        stages.append({
            "kind": "evidence", "index": f"{len(stages) + 1:02d}",
            "kicker": "源码定位", "guide": "展开查看本章引用的源码上下文。",
            "title": "源码引用与上下文", "id": "source-evidence",
            "body_html": source_microscope_html(source_refs), "roadmap": False,
            "toc_items": [{"id": "source-evidence", "level": 2, "text": "源码引用与上下文"}],
        })

    toc: list[dict[str, Any]] = []
    for stage in stages:
        toc.extend(stage.get("toc_items", []))
    return stages, toc


def materialize_nav(course: dict[str, Any], current: Path) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section in course["sections"]:
        item = {
            "id": section["id"],
            "number": section["number"],
            "title": section["title"],
            "url": rel_url(current, SITE_ROOT / section["directory"] / "index.html"),
            "active": current.is_relative_to(SITE_ROOT),
            "chapters": [],
            "groups": [],
        }
        for chapter in section.get("chapters", []):
            target = SITE_ROOT / section["directory"] / chapter["path"]
            item["chapters"].append(
                {**chapter, "url": rel_url(current, target), "active": target == current}
            )
        for group in section.get("groups", []):
            group_item = {
                "id": group["id"],
                "title": group["title"],
                "url": rel_url(
                    current,
                    SITE_ROOT / section["directory"] / group["directory"] / "index.html",
                ),
                "active": group["directory"] in current.parts,
                "chapters": [],
            }
            for chapter in group.get("chapters", []):
                target = (
                    SITE_ROOT
                    / section["directory"]
                    / group["directory"]
                    / chapter["path"]
                )
                group_item["chapters"].append(
                    {**chapter, "url": rel_url(current, target), "active": target == current}
                )
            item["groups"].append(group_item)
        sections.append(item)
    return sections


def materialize_subsystem_overview(
    section: dict[str, Any],
    locations: list[ChapterLocation],
    current: Path,
    output_overrides: dict[str, Path] | None = None,
) -> dict[str, Any] | None:
    """Resolve authored subsystem-map route ids into local chapter links.

    The diagrams live with the section metadata, while URLs depend on the page
    being rendered.  Keeping that split lets the same speculative overview be
    reused by the durable standalone mirror without hard-coded paths.
    """

    authored = section.get("visual_overview")
    if not authored:
        return None

    overview = deepcopy(authored)
    by_id = {location.chapter["id"]: location for location in locations}
    section_locations = [
        location for location in locations if location.section["id"] == section["id"]
    ]
    section_order = {
        location.chapter["id"]: index
        for index, location in enumerate(section_locations, start=1)
    }
    for lens in overview.get("lenses", []):
        routes = []
        for chapter_id in lens.pop("route_ids", []):
            location = by_id.get(chapter_id)
            if location is None:
                raise SystemExit(
                    f"subsystem overview {section['id']} references unknown chapter "
                    f"{chapter_id}"
                )
            if location.section["id"] != section["id"]:
                raise SystemExit(
                    f"subsystem overview {section['id']} route {chapter_id} leaves its "
                    "section; cross-subsystem relationships belong in the diagram, "
                    "while drill-down links stay locally shareable"
                )
            routes.append(
                {
                    "id": chapter_id,
                    "number": section_order[chapter_id],
                    "title": location.chapter["title"],
                    "url": rel_url(
                        current,
                        (output_overrides or {}).get(chapter_id, location.output_path),
                    ),
                }
            )
        lens["routes"] = routes
    return overview


def subsystem_index_toc(overview: dict[str, Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if overview:
        items.append(
            {"id": "subsystem-overview", "text": "子系统三视角总图", "level": 2}
        )
        items.extend(
            {
                "id": f"subsystem-map-{lens['id']}",
                "text": lens["short_title"],
                "level": 3,
            }
            for lens in overview.get("lenses", [])
        )
    items.append({"id": "section-lessons", "text": "机制路线", "level": 2})
    return items

