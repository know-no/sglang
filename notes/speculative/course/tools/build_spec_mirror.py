#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from build_course import (
    configure_markdown,
    materialize_nav,
    materialize_subsystem_overview,
    render_markdown,
    subsystem_index_toc,
)
from course_lib import COURSE_ROOT, REPO_ROOT, chapter_links, flatten_chapters, load_course, parse_frontmatter, sha256, related_notes, rel_url


MIRROR_ROOT = (
    REPO_ROOT
    / "notes"
    / "speculative"
    / "visualizations"
    / "sglang-course-v2"
)


def navigation(course: dict, current: Path, active_id: str | None) -> list[dict]:
    sections = materialize_nav(course, current)
    spec = next(section for section in sections if section['id'] == 'speculative')
    spec.update(url='index.html', active=True)
    for chapter in spec['chapters']:
        chapter.update(url=chapter['path'], active=chapter['id'] == active_id)
    return sections


def write_manifest() -> None:
    manifest = MIRROR_ROOT / "MANIFEST.sha256"
    files = sorted(
        path
        for path in MIRROR_ROOT.rglob("*")
        if path.is_file() and path != manifest and "__pycache__" not in path.parts
    )
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(MIRROR_ROOT).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def build() -> None:
    course = load_course()
    section = next(item for item in course["sections"] if item["id"] == "speculative")
    locations = [
        location
        for location in flatten_chapters(course)
        if location.section["id"] == "speculative"
    ]
    if any(not location.content_path.is_file() for location in locations):
        missing = [
            location.content_path.name
            for location in locations
            if not location.content_path.is_file()
        ]
        raise SystemExit(f"spec mirror content incomplete: {', '.join(missing)}")

    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)
    mirror_assets = MIRROR_ROOT / "assets"
    if mirror_assets.exists():
        shutil.rmtree(mirror_assets)
    shutil.copytree(COURSE_ROOT / "assets", mirror_assets)
    shutil.copy2(COURSE_ROOT / "SOURCE_REVISION.txt", MIRROR_ROOT / "SOURCE_REVISION.txt")

    env = Environment(
        loader=FileSystemLoader(COURSE_ROOT / "templates"),
        autoescape=select_autoescape(("html",)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    md = configure_markdown()
    page_template = env.get_template("page.html")
    index_template = env.get_template("section-index.html")
    site = course["site"]
    chapters = [location.chapter for location in locations]
    search_records = []
    mirror_outputs = {
        location.chapter["id"]: MIRROR_ROOT / location.chapter["path"]
        for location in locations
    }

    for index, location in enumerate(locations):
        meta, body = parse_frontmatter(location.content_path)
        target = mirror_outputs[location.chapter["id"]]
        lesson_stages, toc = render_markdown(
            md,
            body,
            chapter_title=location.chapter["title"],
            source_refs=meta.get("source_refs", []),
            chapter_urls=chapter_links(flatten_chapters(course), target, mirror_outputs),
        )
        context = {
            "site": site,
            "related_notes": related_notes(location.chapter, target),
            "notes_home_url": "../index.html",
            "notes_home_title": "投机可视化总入口",
            "section": section,
            "group": None,
            "chapter": location.chapter,
            "content_html": "",
            "lesson_stages": lesson_stages,
            "toc": toc,
            "chapter_meta": meta,
            "breadcrumbs": [
                {"title": "Speculative Decoding 专题", "url": "index.html"},
                {"title": location.chapter["title"], "url": None},
            ],
            "prev_chapter": (
                {**locations[index - 1].chapter, "url": locations[index - 1].chapter["path"]}
                if index
                else None
            ),
            "next_chapter": (
                {**locations[index + 1].chapter, "url": locations[index + 1].chapter["path"]}
                if index + 1 < len(locations)
                else None
            ),
            "nav_sections": navigation(course, target, location.chapter["id"]),
            "asset_prefix": "assets",
            "root_prefix": rel_url(MIRROR_ROOT / "index.html", MIRROR_ROOT / "index.html"),
            "search_index_script": "assets/data/search-index.js",
            "source_snapshot": course["site"]["snapshot"],
        }
        target.write_text(page_template.render(**context), encoding="utf-8")
        search_records.append(
            {
                "id": location.chapter["id"],
                "title": location.chapter["title"],
                "summary": location.chapter["summary"],
                "section": section["title"],
                "group": "",
                "headings": [item["text"] for item in toc],
                "keywords": meta.get("keywords", []),
                "body": " ".join(body.split()),
                "url": rel_url(MIRROR_ROOT / "index.html", target),
            }
        )

    subsystem_overview = materialize_subsystem_overview(
        section,
        locations,
        MIRROR_ROOT / "index.html",
        {
            location.chapter["id"]: MIRROR_ROOT / location.chapter["path"]
            for location in locations
        },
    )
    index_context = {
        "site": site,
        "notes_home_url": "../index.html",
        "notes_home_title": "投机可视化总入口",
        "section": section,
        "group": None,
        "chapter": None,
        "toc": subsystem_index_toc(subsystem_overview),
        "content_html": "",
        "lesson_stages": [],
        "chapter_meta": {},
        "breadcrumbs": [],
        "prev_chapter": None,
        "next_chapter": None,
        "nav_sections": navigation(course, MIRROR_ROOT / "index.html", None),
        "asset_prefix": "assets",
        "root_prefix": rel_url(MIRROR_ROOT / "index.html", MIRROR_ROOT / "index.html"),
        "search_index_script": "assets/data/search-index.js",
        "source_snapshot": course["site"]["snapshot"],
        "section_cards": [{**chapter, "url": chapter["path"]} for chapter in chapters],
        "subsystem_overview": subsystem_overview,
        "render_mermaid": True,
    }
    (MIRROR_ROOT / "index.html").write_text(
        index_template.render(**index_context), encoding="utf-8"
    )
    search_index = MIRROR_ROOT / "assets" / "data" / "search-index.js"
    search_index.parent.mkdir(parents=True, exist_ok=True)
    search_index.write_text(
        "window.__SGL_SEARCH_INDEX__ = "
        + json.dumps(search_records, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (MIRROR_ROOT / "README.txt").write_text(
        "SGLang Speculative Decoding 源码专题\n"
        "=====================================\n\n"
        "双击 index.html 即可离线阅读。请分享整个 sglang-course-v2 目录，"
        "不要只发送单页 HTML；assets/ 包含本地样式、交互、搜索和 Mermaid。\n\n"
        "正文由 notes/speculative/course/content 与 course.yaml 生成；请勿直接编辑生成 HTML。\n"
        "深入笔记与全景图链接需要仓库中的相邻目录。源码原核对快照见 SOURCE_REVISION.txt。\n"
        "每章保留互补源码图：组件/所有权、运行时序，以及数据、状态、"
        "源码决策或算法边界中的一种。章首 Visual lenses 可直接跳到对应视图。\n",
        encoding="utf-8",
    )

    expected_html = {MIRROR_ROOT / "index.html"} | {
        MIRROR_ROOT / chapter["path"] for chapter in chapters
    }
    for stale in MIRROR_ROOT.glob("*.html"):
        if stale not in expected_html:
            stale.unlink()
    write_manifest()
    print(f"built speculative mirror: {len(chapters)} lessons at {MIRROR_ROOT}")


if __name__ == "__main__":
    build()
