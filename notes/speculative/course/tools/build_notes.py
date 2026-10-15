#!/usr/bin/env python3
"""Render Markdown reading pages and old-path redirects; never alter the Git index."""
from pathlib import Path
from html import escape, unescape
import json
import os
import re
import unicodedata
import sys

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from note_routes import NOTES, REPO, COURSE, note_sources, reading_url, rendered_path

from render_note_page import render_note_page


def slug(text):
    value = ''.join(c for c in text.lower().strip() if c in '-_ ' or unicodedata.category(c)[0] in 'LN')
    return re.sub(r'\s', '-', value)


def math_inline(state, silent):
    start = state.pos
    if state.src[start] != '$' or state.src.startswith('$$', start):
        return False
    if start + 1 >= state.posMax or state.src[start + 1].isspace():
        return False
    end = start + 1
    while True:
        end = state.src.find('$', end)
        if end < 0 or end >= state.posMax or '\n' in state.src[start:end]:
            return False
        if state.src[end - 1] != '\\':
            break
        end += 1
    if state.src[end - 1].isspace() or end == start + 1:
        return False
    if not silent:
        token = state.push('math_inline', 'span', 0)
        token.content = state.src[start + 1:end]
    state.pos = end + 1
    return True


def math_block(state, start_line, end_line, silent):
    if state.sCount[start_line] - state.blkIndent >= 4:
        return False
    line = state.src[state.bMarks[start_line] + state.tShift[start_line]:state.eMarks[start_line]].strip()
    if not line.startswith('$$'):
        return False
    content = [line[2:]]
    last = start_line
    if len(line) > 3 and line.endswith('$$'):
        content = [line[2:-2]]
    else:
        for last in range(start_line + 1, end_line):
            line = state.src[state.bMarks[last] + state.tShift[last]:state.eMarks[last]].strip()
            if line.endswith('$$'):
                content.append(line[:-2])
                break
            content.append(line)
        else:
            return False
    if silent:
        return True
    token = state.push('math_block', 'div', 0)
    token.block = True
    token.content = '\n'.join(content).strip()
    token.map = [start_line, last + 1]
    state.line = last + 1
    return True


def render_math(tokens, index, options, env):
    token = tokens[index]
    display = token.type == 'math_block'
    tag = 'div' if display else 'span'
    return f'<{tag} class="note-math" data-display="{str(display).lower()}" data-tex="{escape(token.content, quote=True)}">{escape(token.content)}</{tag}>'


def render_fence(tokens, index, options, env):
    token = tokens[index]
    language = token.info.strip().split()[0] if token.info.strip() else 'text'
    text = escape(token.content)
    if language == 'mermaid':
        return f'<figure class="note-diagram" data-diagram-frame><div class="diagram-canvas" data-diagram-content><div class="mermaid" data-source="{escape(token.content, quote=True)}">{text}</div></div><figcaption class="diagram-toolbar"><button type="button" data-diagram-open data-diagram-label="源码图解" disabled>放大图表</button><details><summary>图定义</summary><pre><code>{text}</code></pre></details></figcaption></figure>\n'
    try:
        text = highlight(token.content, get_lexer_by_name(language, stripnl=False, ensurenl=False), HtmlFormatter(nowrap=True))
    except ClassNotFound:
        pass
    return f'<pre class="codehilite"><code class="language-{escape(language)}">{text}</code></pre>\n'


def rewrite_html_links(content, source, output):
    def replace_tag(tag):
        value = tag.group(0)
        if re.search(r'\sdownload(?:\s*=|\s|>)', value, re.I):
            return value
        def replace_href(match):
            url = reading_url(unescape(match.group(2)), source, output)
            return f'href={match.group(1)}{escape(url, quote=True)}{match.group(1)}'
        return re.sub(r'''href\s*=\s*(["'])(.*?)\1''', replace_href, value, flags=re.I)
    return re.sub(r'<a\b[^>]*>', replace_tag, content, flags=re.I)


def render_note(source):
    output = rendered_path(source)
    md = MarkdownIt('commonmark', {'html': True}).enable('table').enable('strikethrough')
    md.inline.ruler.before('escape', 'math_inline', math_inline)
    md.block.ruler.before('fence', 'math_block', math_block, {'alt': ['paragraph', 'reference', 'blockquote', 'list']})
    md.renderer.rules.update(math_inline=render_math, math_block=render_math, fence=render_fence)
    tokens = md.parse(source.read_text())
    headings = []
    counts = {}
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            label = ''.join(t.content for t in tokens[i+1].children or [] if t.type in ('text', 'code_inline', 'math_inline'))
            anchor = slug(label)
            n = counts.get(anchor, 0)
            counts[anchor] = n + 1
            if n:
                anchor += f'-{n}'
            token.attrSet('id', anchor)
            if token.tag == 'h2':
                headings.append((anchor, label))
        if token.type in ('html_inline', 'html_block'):
            token.content = rewrite_html_links(token.content, source, output)
        for child in token.children or []:
            if child.type == 'link_open':
                child.attrSet('href', reading_url(child.attrGet('href') or '', source, output))
            elif child.type == 'html_inline':
                child.content = rewrite_html_links(child.content, source, output)
    body = md.renderer.render(tokens, md.options, {})
    body = re.sub(r'<table>(.*?)</table>', r'<div class="table-scroll"><table>\1</table></div>', body, flags=re.S)
    body = re.sub(r'<li>\[([ xX])\] ', lambda m: '<li class="task-item"><input type="checkbox" disabled' + (' checked' if m.group(1).lower() == 'x' else '') + '> ', body)
    title = next((tokens[i+1].content for i, t in enumerate(tokens[:-1]) if t.type == 'heading_open'), source.stem)
    topic = next((parent / 'README.md' for parent in source.parents if (parent / 'README.md').is_file() and parent.is_relative_to(NOTES)), NOTES / 'README.md')
    if topic == source:
        topic = NOTES / 'README.md'
    output.write_text(render_note_page(source, output, title, body, headings, rendered_path(topic)))
    return output



def main():
    outputs = [render_note(source) for source in note_sources()]
    print(f'Rendered {len(outputs)} speculative Markdown reading pages.')


if __name__ == '__main__':
    main()
