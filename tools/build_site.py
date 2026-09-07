#!/usr/bin/env python3
"""Build the GitHub Pages site for the Agent IR specification.

README.md is the single source of truth. This script renders it to a static
page (site/index.html) with a table of contents, syntax highlighting and
Mermaid diagrams left intact for client-side rendering.

    python3 tools/build_site.py [--out site]
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import shutil
import sys

import markdown
from pygments.formatters import HtmlFormatter

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

REPO_URL = "https://github.com/rustnew/Agent-IR"
SITE_URL = "https://rustnew.github.io/Agent-IR/"
DESCRIPTION = (
    "Agent IR is a compilation infrastructure for agentic systems: a typed, "
    "verifiable intermediate representation whose effect and capability system "
    "makes agent optimizations decidable rather than heuristic."
)

# Pygments has no MLIR lexer; LLVM is close enough for the textual syntax.
LANG_ALIASES = {"mlir": "llvm"}

MERMAID_TOKEN = "MERMAIDFIGURE{}ENDMERMAIDFIGURE"


def split_front_matter(text: str) -> tuple[str, list[str], str]:
    """Return (h1 title, intro lines, body) split at the first horizontal rule."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        sys.exit("error: README.md must start with a level-1 heading")
    title = lines[0][2:].strip()
    try:
        cut = next(i for i, ln in enumerate(lines) if i > 0 and ln.strip() == "---")
    except StopIteration:
        sys.exit("error: README.md needs a '---' rule after the introduction")
    return title, lines[1:cut], "\n".join(lines[cut + 1 :]).strip()


def build_hero(title: str, intro: list[str]) -> dict[str, str]:
    """Turn the README preamble into hero fields."""
    tagline = ""
    meta: list[tuple[str, str]] = []
    for line in intro:
        line = line.strip()
        if not line:
            continue
        bold = re.fullmatch(r"\*\*(.+)\*\*", line)
        if bold and not tagline:
            tagline = bold.group(1)
            continue
        field = re.fullmatch(r"([A-Z][A-Za-z ]{2,20}):\s*(.+)", line)
        if field and field.group(1) != "Site":  # the page need not link to itself
            meta.append((field.group(1), field.group(2)))

    heading, _, version = title.partition("—")
    version = version.strip()
    rows = "\n".join(
        "    <div><dt>{}</dt><dd>{}</dd></div>".format(html.escape(k), html.escape(v))
        for k, v in meta
    )
    return {
        "TITLE": title,
        "HEADING": html.escape(heading.strip() or title),
        "EYEBROW": html.escape(version or "Specification"),
        "TAGLINE": html.escape(tagline),
        "META": rows,
        "VERSION": html.escape(
            (re.search(r"v\d[\d.]*", version).group(0) if re.search(r"v\d[\d.]*", version) else "v0.1")
        ),
    }


def extract_mermaid(body: str) -> tuple[str, list[str]]:
    """Replace ```mermaid fences with placeholders Markdown will leave alone."""
    blocks: list[str] = []

    def take(match: re.Match[str]) -> str:
        blocks.append(match.group(1))
        return MERMAID_TOKEN.format(len(blocks) - 1)

    pattern = re.compile(r"^```mermaid[ \t]*\n(.*?)\n```[ \t]*$", re.M | re.S)
    return pattern.sub(take, body), blocks


def restore_mermaid(rendered: str, blocks: list[str]) -> str:
    for i, source in enumerate(blocks):
        token = MERMAID_TOKEN.format(i)
        # Mermaid reads textContent, so the source must be HTML-escaped here:
        # otherwise the <br/> in node labels would be parsed away by the browser.
        figure = (
            '<div class="figure" role="figure">'
            '<pre class="mermaid">{}</pre>'
            "</div>"
        ).format(html.escape(source))
        rendered = rendered.replace("<p>{}</p>".format(token), figure).replace(token, figure)
    return rendered


def number_headings(fragment: str) -> str:
    """Wrap the leading section number of a heading so it can be accented."""
    return re.sub(
        r"(<(h[23])[^>]*>)(\d+(?:\.\d+)*\.?)(\s)",
        r'\1<span class="num">\3</span>\4',
        fragment,
    )


def number_toc(fragment: str) -> str:
    return re.sub(
        r"(<a href=\"#[^\"]+\">)(\d+(?:\.\d+)*\.?)(\s)",
        r'\1<span class="num">\2</span>\3',
        fragment,
    )


def wrap_tables(fragment: str) -> str:
    return re.sub(
        r"<table>(.*?)</table>",
        lambda m: '<div class="table-wrap"><table>{}</table></div>'.format(m.group(1)),
        fragment,
        flags=re.S,
    )


def pygments_css() -> str:
    """Light and dark highlighting, keyed on the resolved data-theme attribute."""
    light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    dark = HtmlFormatter(style="dracula").get_style_defs('html[data-theme="dark"] .codehilite')
    # Drop the styles' own block backgrounds; the page palette owns those.
    strip = re.compile(r"^[^{]*\.codehilite\s*\{[^}]*\}\s*", re.M)
    light = strip.sub("", light, count=1)
    dark = strip.sub("", dark, count=1)
    return (
        "\n/* ---------- syntax highlighting (generated by Pygments) ---------- */\n"
        f"{light}\n{dark}\n"
    )


def render(readme: pathlib.Path, out_dir: pathlib.Path) -> None:
    text = readme.read_text(encoding="utf-8")
    title, intro, body = split_front_matter(text)

    for alias, lexer in LANG_ALIASES.items():
        body = body.replace(f"```{alias}\n", f"```{lexer}\n")

    body, diagrams = extract_mermaid(body)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list", "sane_lists"],
        extension_configs={
            "codehilite": {"guess_lang": False, "linenums": False},
            "toc": {"toc_depth": "2-3", "permalink": "#", "anchorlink": False},
        },
    )
    content = md.convert(body)
    content = restore_mermaid(content, diagrams)
    content = wrap_tables(content)
    content = number_headings(content)
    toc = number_toc(md.toc)

    fields = build_hero(title, intro)
    fields.update(
        {
            "CONTENT": content,
            "TOC": toc,
            "DESCRIPTION": html.escape(DESCRIPTION),
            "REPO_URL": REPO_URL,
            "SITE_URL": SITE_URL,
        }
    )

    page = (TOOLS / "template.html").read_text(encoding="utf-8")
    for key, value in fields.items():
        page = page.replace(f"__{key}__", value)
    left = re.findall(r"__[A-Z_]+__", page)
    if left:
        sys.exit(f"error: unfilled template placeholders: {sorted(set(left))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "style.css").write_text(
        (TOOLS / "site.css").read_text(encoding="utf-8") + pygments_css(), encoding="utf-8"
    )
    shutil.copyfile(TOOLS / "favicon.svg", out_dir / "favicon.svg")

    print(
        f"built {out_dir/'index.html'} "
        f"({len(page):,} bytes, {len(diagrams)} diagrams, {content.count('<table>')} tables)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site", help="output directory (default: site)")
    parser.add_argument("--readme", default="README.md", help="source document")
    args = parser.parse_args()
    render(ROOT / args.readme, ROOT / args.out)


if __name__ == "__main__":
    main()
