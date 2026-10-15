#!/usr/bin/env python3
"""Check local reading links/resources and the speculative-only search boundary."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re

SPEC = Path(__file__).resolve().parents[2]
NOTES = SPEC.parent
SITE = SPEC / 'visualizations/sglang-course-v2'

class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if attrs.get(key):
                self.links.append((attrs[key], 'download' in attrs))

pages = {p: Page(p) for p in SPEC.rglob('*.html') if 'templates' not in p.parts}
errors = []
links = 0
for path, page in pages.items():
    for url, download in page.links:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
        links += 1
        if not target.exists():
            errors.append(f'{path.relative_to(SPEC)}: missing {url}')
            continue
        if target.is_relative_to(NOTES) and not target.is_relative_to(SPEC) and target != NOTES / 'index.html':
            errors.append(f'{path.relative_to(SPEC)}: leaves speculative material: {url}')
        if target.suffix == '.md' and not download:
            errors.append(f'{path.relative_to(SPEC)}: raw Markdown reading link: {url}')
        if parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
            errors.append(f'{path.relative_to(SPEC)}: missing anchor {url}')
# Check CSS-loaded font files/images, including local KaTeX assets.
for path in SPEC.rglob('*.css'):
    for raw in re.findall(r'url\(([^)]+)\)', path.read_text()):
        url = raw.strip('\"\' ')
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        if not (path.parent / unquote(parts.path)).resolve().exists():
            errors.append(f'{path.relative_to(SPEC)}: missing CSS resource {url}')
index = SITE / 'assets/data/search-index.js'
records = json.loads(index.read_text().removeprefix('window.__SGL_SEARCH_INDEX__ = ').rstrip().removesuffix(';'))
if len(records) != 27:
    errors.append(f'Expected 8 chapters + 15 labs + 4 notes, got {len(records)} search entries')
for entry in records:
    target = (SITE / urlsplit(entry['url']).path).resolve()
    if target not in pages:
        errors.append(f'Bad search destination {entry["url"]}')
for line in (SITE / 'MANIFEST.sha256').read_text().splitlines():
    digest, filename = line.split('  ', 1)
    target = SITE / filename
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        errors.append(f'Manifest mismatch: {filename}')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'Passed: {len(pages)} HTML pages; {links} local links/resources; 27 speculative search entries; manifest.')
