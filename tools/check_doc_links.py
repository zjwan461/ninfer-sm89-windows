"""Check that relative links in the repository's Markdown files resolve.

Only local targets are checked: absolute URLs, mail links and `#anchor` links are skipped, and so
is prose that merely looks like a link (a bracket expression followed by a parenthesised
expression, e.g. `[e](SiLU(W_gate[e] h)` in a formula). Run from anywhere:

    python tools/check_doc_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#")


def local_targets(text: str) -> list[str]:
    """Return the local paths a Markdown text links to, in source order."""

    targets: list[str] = []
    for raw in LINK.findall(text):
        target = raw.strip().split(" ")[0]
        if not target or target.startswith(EXTERNAL_PREFIXES):
            continue
        path = target.split("#")[0]
        if not path:
            continue
        # A formula such as `[e](SiLU(W_gate[e] h)` is not a link: a real path has no whitespace
        # and no unbalanced parenthesis.
        if any(character.isspace() for character in path) or path.count("(") != path.count(")"):
            continue
        targets.append(path)
    return targets


def check(root: Path) -> list[tuple[Path, str]]:
    missing: list[tuple[Path, str]] = []
    for document in sorted(root.rglob("*.md")):
        if ".git" in document.parts:
            continue
        for path in local_targets(document.read_text(encoding="utf-8", errors="ignore")):
            if not (document.parent / path).exists():
                missing.append((document.relative_to(root), path))
    return missing


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parent.parent
    missing = check(root)
    if missing:
        for document, path in missing:
            print(f"{document}: missing {path}")
        print(f"{len(missing)} broken relative link(s)")
        return 1
    print("all relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
