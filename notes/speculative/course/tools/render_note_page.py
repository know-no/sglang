"""Render repository companions with the course's single navigation and shell."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from course_lib import COURSE_ROOT, REPO_ROOT, SITE_ROOT, load_course, rel_url
from build_course import materialize_nav


def shell_context(output: Path, title: str, *, kind: str, domain: str) -> dict:
    course = load_course()
    return {
        "site": {**course["site"], "evidence_notice": ""},
        "section": {"id": kind, "title": "深入源码笔记" if kind == "note" else "dSpark 交互实验", "domain": domain},
        "chapter": {"id": f"{kind}-{output.stem}", "title": title, "wide": True},
        "source_snapshot": None,
        "nav_sections": materialize_nav(course, output),
        # Use source assets: rebuilding a companion does not require copying site assets.
        "asset_prefix": rel_url(output, COURSE_ROOT / "assets"),
        "root_prefix": rel_url(output, SITE_ROOT / "index.html"),
        "search_index_script": rel_url(output, SITE_ROOT / "assets/data/search-index.js"),
        "notes_home_url": rel_url(output, REPO_ROOT / "notes/index.html"),
        "notes_home_title": "SGLang 全景图",
        "breadcrumbs": [
            {"title": "投机解码专题", "url": rel_url(output, SITE_ROOT / "index.html")},
            {"title": "SGLang 全景图", "url": rel_url(output, REPO_ROOT / "notes/index.html")},
            {"title": title},
        ],
        "toc": [],
    }


def render_shell(template: str, context: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(COURSE_ROOT / "templates"),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
    )
    return env.get_template(template).render(**context)


def render_note_page(source: Path, output: Path, title: str, body: str, headings: list, topic: Path) -> str:
    domain = source.relative_to(REPO_ROOT / "notes").parts[0]
    domain = {"attention": "memory", "io": "serving", "development": "engineering"}.get(domain, domain)
    context = shell_context(output, title, kind="note", domain=domain)
    context.update({
        "source_url": rel_url(output, source),
        "source_path": str(source.relative_to(REPO_ROOT)),
        "topic_url": rel_url(output, topic),
        "maintenance_prefix": rel_url(output, COURSE_ROOT / "assets" / "notes"),
        "content_html": body,
        "toc": [{"id": anchor, "text": label, "level": 2} for anchor, label in headings],
        "has_math": 'class="note-math"' in body,
    })
    return render_shell("note-page.html", context)
