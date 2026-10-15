from __future__ import annotations

import hashlib
import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


COURSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COURSE_ROOT.parents[2]
SITE_ROOT = COURSE_ROOT.parent / "visualizations" / "sglang-course-v2"
# Both the course builders and the note renderer use this one Markdown route map.
from note_routes import rendered_path


@dataclass(frozen=True)
class ChapterLocation:
    section: dict[str, Any]
    group: dict[str, Any] | None
    chapter: dict[str, Any]
    output_path: Path

    @property
    def content_path(self) -> Path:
        return COURSE_ROOT / "content" / f"{self.chapter['id']}.md"


def load_course() -> dict[str, Any]:
    with (COURSE_ROOT / "course.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_chapters(course: dict[str, Any]) -> Iterable[ChapterLocation]:
    for section in course["sections"]:
        section_dir = SITE_ROOT / section["directory"]
        for chapter in section.get("chapters", []):
            yield ChapterLocation(section, None, chapter, section_dir / chapter["path"])
        for group in section.get("groups", []):
            group_dir = section_dir / group["directory"]
            for chapter in group.get("chapters", []):
                yield ChapterLocation(section, group, chapter, group_dir / chapter["path"])


def flatten_chapters(course: dict[str, Any]) -> list[ChapterLocation]:
    return list(iter_chapters(course))


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        _, raw_meta, body = text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(f"{path}: unclosed YAML frontmatter") from exc
    meta = yaml.safe_load(raw_meta) or {}
    return meta, body.strip() + "\n"


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "-", value, flags=re.UNICODE)
    return value.strip("-") or "section"


def rel_url(from_file: Path, to_file: Path) -> str:
    return Path(os.path.relpath(to_file, from_file.parent)).as_posix()


def chapter_links(
    locations: list[ChapterLocation],
    current: Path,
    output_overrides: dict[str, Path] | None = None,
) -> dict[str, str]:
    """Map sibling Markdown sources to this build's chapter destinations."""
    overrides = output_overrides or {}
    return {
        location.content_path.name: rel_url(
            current, overrides.get(location.chapter["id"], location.output_path)
        )
        for location in locations
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_plain_text(markdown: str) -> str:
    text = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[`*_#>|\[\]()]", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def source_ref_is_valid(ref: dict[str, Any]) -> tuple[bool, str]:
    raw_path = ref.get("path")
    line = ref.get("line")
    if not raw_path or not isinstance(line, int) or line < 1:
        return False, "source ref requires path and positive integer line"
    source_path = REPO_ROOT / raw_path
    if not source_path.is_file():
        return False, f"missing source file: {raw_path}"
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if line > len(lines):
        return False, f"line {line} exceeds {raw_path} ({len(lines)} lines)"
    symbol = str(ref.get("symbol", "")).strip()
    if symbol:
        start = max(0, line - 8)
        end = min(len(lines), line + 8)
        if not any(symbol in candidate for candidate in lines[start:end]):
            return False, f"symbol {symbol!r} not found near {raw_path}:{line}"
    return True, ""


def related_notes(chapter: dict[str, Any], current: Path) -> list[dict[str, Any]]:
    """Resolve canonical note links configured only in course.yaml."""
    result = []
    for note in chapter.get("related_notes", []):
        target = REPO_ROOT / note["path"]
        if not target.is_file():
            raise ValueError(f"{chapter['id']}: missing related note {note['path']}")
        url = rel_url(current, rendered_path(target) if target.suffix.lower() == ".md" else target)
        if note.get("anchor"):
            url += "#" + note["anchor"]
        result.append({**note, "url": url})
    return result


def allowed_companion_links(course: dict[str, Any]) -> set[Path]:
    """Allow explicit repository reading links; runtime assets stay inside each site."""
    paths = {REPO_ROOT / "notes" / "index.html",
             REPO_ROOT / "notes" / "speculative" / "visualizations" / "index.html"}
    for location in flatten_chapters(course):
        for note in location.chapter.get("related_notes", []):
            source = REPO_ROOT / note["path"]
            if not source.is_file():
                raise ValueError(f"{location.chapter['id']}: missing related note {note['path']}")
            paths.add(rendered_path(source) if source.suffix.lower() == ".md" else source)
    paths.add(SITE_ROOT / 'index.html')
    for section in course['sections']:
        paths.add(SITE_ROOT / section['directory'] / 'index.html')
        for group in section.get('groups', []):
            paths.add(SITE_ROOT / section['directory'] / group['directory'] / 'index.html')
    paths.update(loc.output_path for loc in flatten_chapters(course))
    return {path.resolve() for path in paths}
