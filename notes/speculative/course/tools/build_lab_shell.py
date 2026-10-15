#!/usr/bin/env python3
"""Wrap the single-source dSpark experiments in the shared course shell.

The .html.jinja files are preserved HTML source documents, not independently
rendered reading pages. Their body, inline scripts, IDs and asset URLs remain
verbatim, except that the inner main becomes a div inside the shell's main.
"""
from __future__ import annotations

import argparse
import hashlib
from html import unescape
from pathlib import Path
import re

from course_lib import REPO_ROOT, rel_url
from render_note_page import shell_context, render_shell

LAB_ROOT = REPO_ROOT / "notes/speculative/visualizations"
SOURCE_ROOT = LAB_ROOT / "sources"


def theme_style(css: str) -> str:
    """Retain semantic tints while replacing fixed dark surfaces/light text."""
    def surface(match: re.Match) -> str:
        red, green, blue = (int(match[i]) for i in (1, 2, 3))
        opacity = float(match[4])
        if 0 < max(red, green, blue) <= 32 and opacity >= 0.5:
            return "var(--panel)"
        return match[0]
    css = re.sub(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([.\d]+)\s*\)", surface, css)
    for color in ("#bdcbe0", "#c4d1e2", "#dce9f8", "#d9e4f2", "#cbd9e9", "#d8e5f4", "#d7e6f8"):
        css = css.replace(color, "var(--text-soft)")
    return css.replace("#c9b0e8", "var(--purple)").replace("#020711", "var(--code-bg)")


def scoped_style(css: str) -> str:
    # Local page styles contain coordinates/classes, never global root selectors.
    # Shared assets separately define the scoped reset and semantic tokens.
    return "@scope (.dspark-lab) {\n" + theme_style(css) + "\n}"


def render_lab(source: Path) -> tuple[Path, str]:
    original = source.read_text(encoding="utf-8")
    output = LAB_ROOT / source.name.removesuffix(".jinja")
    title_match = re.search(r"<title>(.*?)</title>", original, re.S | re.I)
    head_match = re.search(r"<head>(.*?)</head>", original, re.S | re.I)
    body_match = re.search(r'<body\b[^>]*data-page="([^"]+)"[^>]*>(.*?)</body>', original, re.S | re.I)
    if not title_match or not head_match or not body_match:
        raise ValueError(f"{source}: expected title, head and data-page body")
    title = unescape(title_match.group(1))
    page_id, body = body_match.groups()
    # Avoid nested main landmarks without reparsing or normalizing teaching HTML.
    body, count = re.subn(r'<main class="page page--wide">', '<div class="page page--wide" data-dspark-main>', body)
    if count != 1 or body.count("</main>") != 1:
        raise ValueError(f"{source}: expected one experiment main landmark")
    body = body.replace("</main>", "</div>")
    head = head_match.group(1)
    # Preserve CSP, local styles and all script order. Metadata belongs to the shell.
    extras = re.findall(r'<meta\b[^>]*http-equiv[^>]*>|<link\b[^>]*rel="stylesheet"[^>]*>|<style\b[^>]*>.*?</style>|<script\b[^>]*>.*?</script>', head, re.S | re.I)
    head = "\n".join(extras)
    head = re.sub(r"(<style\b[^>]*>)(.*?)(</style>)", lambda m: m[1] + scoped_style(m[2]) + m[3], head, flags=re.S | re.I)
    context = shell_context(output, title, kind="lab", domain="speculative")
    context.update({
        "lab_head": head,
        "lab_body": body,
        "lab_page_id": page_id,
        "source_url": rel_url(output, source),
        "source_path": str(source.relative_to(REPO_ROOT)),
        "source_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "lab_assets": rel_url(output, LAB_ROOT / "assets"),
        "breadcrumbs": [
            {"title": "投机解码专题", "url": context["root_prefix"]},
            {"title": "投机实验路线", "url": "index.html"},
            {"title": title},
        ],
    })
    return output, render_shell("lab-page.html", context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated pages are current without writing")
    args = parser.parse_args()
    sources = sorted(SOURCE_ROOT.glob("dspark-*.html.jinja"))
    if len(sources) != 15:
        raise ValueError(f"Expected 15 single-source dSpark experiments, found {len(sources)}")
    stale = []
    for source in sources:
        output, result = render_lab(source)
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != result:
                stale.append(str(output.relative_to(REPO_ROOT)))
        else:
            output.write_text(result, encoding="utf-8")
    if stale:
        raise SystemExit("Stale lab shells: " + ", ".join(stale))
    print(f'{"Verified" if args.check else "Rendered"} {len(sources)} dSpark lab shells from preserved HTML sources.')


if __name__ == "__main__":
    main()
