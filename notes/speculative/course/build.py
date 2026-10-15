#!/usr/bin/env python3
"""Rebuild the preserved speculative course, labs, and rendered notes."""
from pathlib import Path
from html import unescape
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for script in ('build_spec_mirror.py', 'build_lab_shell.py', 'build_notes.py'):
    subprocess.run([sys.executable, str(ROOT / 'tools' / script)], check=True)

sys.path.insert(0, str(ROOT / 'tools'))
from build_spec_mirror import write_manifest
from course_lib import SITE_ROOT, rel_url
from note_routes import note_sources, rendered_path

# All generated shells use this one index, whose relative URLs start at SITE_ROOT.
index = SITE_ROOT / 'assets/data/search-index.js'
prefix = 'window.__SGL_SEARCH_INDEX__ = '
records = json.loads(index.read_text().removeprefix(prefix).rstrip().removesuffix(';'))
for source in sorted((ROOT.parent / 'visualizations/sources').glob('dspark-*.html.jinja')):
    content = source.read_text()
    title = unescape(re.search(r'<title>(.*?)</title>', content, re.S).group(1))
    target = source.parent.parent / source.name.removesuffix('.jinja')
    records.append({'id': source.stem, 'title': title, 'section': 'dSpark 交互实验',
                    'summary': '保留的 dSpark 张量、状态与时序实验。',
                    'body': title, 'url': rel_url(SITE_ROOT / 'index.html', target)})
for source in note_sources():
    if source.name == 'README.md':
        continue
    content = source.read_text()
    title = next((line.lstrip('# ') for line in content.splitlines() if line.startswith('# ')), source.stem)
    records.append({'id': source.stem, 'title': title, 'section': '投机解码深入笔记',
                    'summary': '保留的投机解码源码主文与配置参考。',
                    'body': ' '.join(content.split()),
                    'url': rel_url(SITE_ROOT / 'index.html', rendered_path(source))})
index.write_text(prefix + json.dumps(records, ensure_ascii=False, separators=(',', ':')) + ';\n')
write_manifest()
print(f'Indexed {len(records)} speculative chapters, labs, and notes.')
