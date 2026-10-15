"""Map preserved speculative Markdown to browser reading pages."""
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
import os
import yaml

COURSE = Path(__file__).resolve().parents[1]
SPECULATIVE = COURSE.parent
NOTES = SPECULATIVE.parent
REPO = NOTES.parent


def course_pages():
    course = yaml.safe_load((COURSE / 'course.yaml').read_text())
    target = SPECULATIVE / 'visualizations/sglang-course-v2'
    return {COURSE / 'content' / (chapter['id'] + '.md'): target / chapter['path']
            for section in course['sections'] for chapter in section['chapters']}


def rendered_path(source: Path) -> Path:
    source = source.resolve()
    old = NOTES / 'scheduler/speculative'
    if source.is_relative_to(old):
        source = SPECULATIVE / source.relative_to(old)
    if source.suffix.lower() != '.md' or not source.is_relative_to(NOTES):
        return source
    pages = course_pages()
    if source in pages:
        return pages[source]
    return source.with_name('index.html') if source.name == 'README.md' else source.with_suffix('.html')


def reading_url(url: str, source: Path, output: Path | None = None) -> str:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        return url
    target = (source.parent / unquote(parts.path)).resolve()
    if target.suffix.lower() != '.md' or not target.is_file() or not target.is_relative_to(NOTES):
        return url
    output = output or rendered_path(source)
    path = Path(os.path.relpath(rendered_path(target), output.parent)).as_posix()
    return urlunsplit(('', '', path, parts.query, parts.fragment))


def note_sources():
    return sorted([*SPECULATIVE.glob('*.md'), SPECULATIVE / 'visualizations/README.md', COURSE / 'README.md'])
