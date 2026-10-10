"""
Regenerate the bundled documentation.

Three kinds of page live in ``docs/``:

- ``index.html`` and ``introduction.html`` are written by hand, and are not
  touched here.
- ``usage.html`` is rendered from ``GUIDE.md`` at the top of the repository, so
  the guide the GUI shows and the guide on GitHub cannot drift apart.
- ``getelec.html``, ``search.js`` and ``getelec/`` are generated from the
  docstrings by pdoc, and are overwritten wholesale.

``docs/`` is also the website, published by GitHub Pages at ``SITE``:
``sitemap.xml`` lists every page for search engines and is rewritten here.

Run from the repository root::

    pip install pdoc
    python docs/regenerate.py

The GUI loads these files from disk, so anything not regenerated here will keep
showing the previous release's contents.
"""

from __future__ import annotations

import html as html_module
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

#: Where files of the repository are read online. The rendered guide links there
#: for anything that is not one of these pages -- the notebook, INSTALL.md --
#: since the application ships only docs/ and a relative link would be dead.
REPOSITORY = "https://github.com/sbcarceles13/GETELEC/blob/main/"

#: Where GitHub Pages serves docs/.
SITE = "https://sbcarceles13.github.io/GETELEC/"


def render_markdown(text: str) -> str:
    """
    Minimal Markdown to HTML.

    Deliberately small: it covers headings, paragraphs, lists, tables, fenced
    and indented code and inline formatting, which is all the guide uses. A
    general converter would be a dependency for no benefit.

    As in Markdown, the lines of a paragraph or of a list item are joined before
    anything else is done with them. Taken one at a time, every line of the
    source became a paragraph of its own, with a gap after each, and bold text
    running across a line break was printed with its asterisks.
    """
    lines = [line.rstrip() for line in text.split("\n")]
    out, paragraph, items = [], [], []
    list_tag = None

    def end_paragraph():
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def end_list():
        nonlocal list_tag
        if items:
            out.append(f"<{list_tag}>" + "".join(
                f"<li>{_inline(' '.join(item))}</li>" for item in items)
                + f"</{list_tag}>")
            items.clear()
        list_tag = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):                     # fenced code
            end_paragraph()
            end_list()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(html_module.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            i += 1
            continue

        if stripped.startswith("|"):                       # table
            end_paragraph()
            end_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                i += 1
                if set(row.replace("|", "").strip()) <= set("-: "):
                    continue                               # the rule under the header
                tag = "td" if rows else "th"
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>"
                                             for c in cells) + "</tr>")
            out.append("<table>\n" + "\n".join(rows) + "\n</table>")
            continue

        if not stripped:
            end_paragraph()
            end_list()
            i += 1
            continue

        if re.match(r"^-{3,}$", stripped):
            end_paragraph()
            end_list()
            out.append("<hr>")
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            end_paragraph()
            end_list()
            level = len(heading.group(1))
            out.append(f'<h{level} id="{_anchor(heading.group(2))}">'
                       f'{_inline(heading.group(2))}</h{level}>')
            i += 1
            continue

        item = re.match(r"^(?:([-*])|\d+\.)\s+(.*)$", line)
        if item:
            end_paragraph()
            tag = "ul" if item.group(1) else "ol"
            if list_tag != tag:
                end_list()
                list_tag = tag
            items.append([item.group(2)])
            i += 1
            continue
        if items and line.startswith(" "):                # an item's next line
            items[-1].append(stripped)
            i += 1
            continue

        if line.startswith("    ") and not paragraph:     # indented code
            end_list()
            block = []
            while i < len(lines) and lines[i].startswith("    "):
                block.append(html_module.escape(lines[i][4:]))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        end_list()
        paragraph.append(stripped)
        i += 1

    end_paragraph()
    end_list()
    return "\n".join(out)


def _anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", title.replace("`", "").lower()).strip("-")


def _inline(text: str) -> str:
    # Code spans are set aside first and put back last, so nothing inside them
    # is read as formatting: `m_e*` twice in one sentence used to pair its two
    # asterisks into an emphasis crossing both code tags.
    spans = []

    def keep(match):
        spans.append(f"<code>{html_module.escape(match.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep, text)
    text = html_module.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: '<a href="{}">{}</a>'.format(_link_target(m.group(2)), m.group(1)),
        text)
    return re.sub("\x00(\\d+)\x00", lambda m: spans[int(m.group(1))], text)


def _link_target(target: str) -> str:
    """Where a link in GUIDE.md points once the guide is usage.html."""
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return target
    if target.split("#")[0] == "GUIDE.md":
        return "usage.html" + target[len("GUIDE.md"):]
    return REPOSITORY + target


def regenerate_api_docs():
    """Run pdoc into a temporary tree, then swap it in."""
    try:
        import pdoc  # noqa: F401
    except ImportError:
        raise SystemExit("pdoc is not installed. Run: pip install pdoc")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-m", "pdoc", "getelec", "-o", tmp, "--no-show-source"],
            cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit("pdoc failed.")
        shutil.rmtree(DOCS / "getelec", ignore_errors=True)
        shutil.copytree(Path(tmp) / "getelec", DOCS / "getelec")
        for name in ("getelec.html", "search.js"):
            shutil.copy(Path(tmp) / name, DOCS / name)
    for page in [DOCS / "getelec.html", *sorted((DOCS / "getelec").glob("*.html"))]:
        describe_api_page(page)
    print("  regenerated API reference from docstrings")


def describe_api_page(page: Path):
    """
    Give a pdoc page the description a search engine shows under its title.

    pdoc writes none, and a result without one is shown with whatever text the
    page happens to start with.
    """
    module = "getelec" if page.parent == DOCS else f"getelec.{page.stem}"
    text = page.read_text(encoding="utf-8")
    title = f"<title>{module} API documentation</title>"
    if text.count(title) != 1:
        raise SystemExit(f"{page.name}: pdoc's title line was not found once.")
    description = (f"API reference of {module}, from GETELEC, the Python package for "
                   "thermal-field electron emission calculations.")
    text = text.replace(title, f'{title}\n    <meta name="description" content="{description}">')
    page.write_text(text, encoding="utf-8", newline="\n")


def write_sitemap():
    """List every page of docs/ in sitemap.xml, by its address on the website."""
    pages = sorted(path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.html"))
    pages.remove("index.html")
    urls = [SITE] + [SITE + page for page in pages]
    lines = (['<?xml version="1.0" encoding="UTF-8"?>',
              '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
             + [f"  <url><loc>{url}</loc></url>" for url in urls] + ["</urlset>", ""])
    (DOCS / "sitemap.xml").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"  listed {len(urls)} pages in sitemap.xml")


SIDEBAR_SECTIONS = [
    ("Getting Started", [("The physical problem", "the-physical-problem"),
                         ("Architecture", "architecture")]),
    ("Modules", [("potential_barrier", "potential_barrier"),
                 ("band_structure", "band_structure"),
                 ("electron_supply", "electron_supply"),
                 ("transmission_solver", "transmission_solver"),
                 ("transmission_solutions", "transmission_solutions"),
                 ("electron_emitter", "electron_emitter")]),
    ("Reference", [("Semiconductors: energy distributions",
                    "semiconductors-the-two-energy-distributions"),
                   ("Graphical interface", "the-graphical-interface"),
                   ("Choosing a solver", "choosing-a-solver"),
                   ("Accuracy", "accuracy-and-how-to-check-it"),
                   ("Units", "units")]),
]


def regenerate_usage_page():
    """
    Render usage.html from GUIDE.md, reusing the existing page's stylesheet.

    Rendering rather than maintaining a second copy means the guide shown inside
    the application and the one on GitHub cannot drift apart.
    """
    source = DOCS / "usage.html"
    head = source.read_text(encoding="utf-8")
    head = head[:head.index("</head>") + len("</head>")]

    guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
    guide = re.sub(r"## Contents.*?\n---\n", "", guide, flags=re.S)
    guide = re.sub(r"^# .*?\n", "", guide, count=1)

    def links(items):
        return "\n".join(
            f'                <li><a href="#{anchor}">{title}</a></li>'
            for title, anchor in items)

    sections = "".join(
        f"            <h2>{name}</h2>\n            <ul>\n{links(items)}\n"
        f"            </ul>\n" for name, items in SIDEBAR_SECTIONS)
    sidebar = ("        <aside>\n            <h1>GETELEC Guide</h1>\n" + sections +
               '            <h2>Pages</h2>\n            <ul>\n'
               '                <li><a href="index.html">Overview</a></li>\n'
               '                <li><a href="introduction.html">Introduction</a></li>\n'
               '                <li><a href="getelec.html">API reference</a></li>\n'
               "            </ul>\n        </aside>\n")

    page = (head + '\n<body>\n\n    <div class="container">\n' + sidebar +
            '\n        <main>\n            <h1 class="page-title">'
            'Usage &amp; Reference Guide</h1>\n'
            '            <p class="subtitle">How GETELEC is organised, what each '
            'module does, how to choose a solver, and how to check the answer.</p>\n'
            + render_markdown(guide) +
            "\n        </main>\n    </div>\n\n</body>\n</html>\n")
    source.write_text(page, encoding="utf-8")

    identifiers = set(re.findall(r'id="([^"]+)"', page))
    broken = [ref for ref in re.findall(r'href="#([^"]+)"', page)
              if ref not in identifiers]
    if broken:
        print(f"  warning: usage.html has dead in-page links: {broken}")
    print("  rendered usage.html from GUIDE.md")


def main():
    regenerate_api_docs()
    regenerate_usage_page()
    write_sitemap()
    print("\nDone. The GUI reads these files directly, so no further step is needed.")
    print("Edit GUIDE.md rather than usage.html: the HTML is overwritten here.")


if __name__ == "__main__":
    main()
