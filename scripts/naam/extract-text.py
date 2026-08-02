#!/usr/bin/env python3
"""
Provenance record for src/data/naam/source-text.txt.

DESIGN.md P9 ("proof over claim"): every number on /naam is traceable to a page
of Boy_Name_Candidates.pdf. That only holds if the text dump the parser reads is
itself reproducible, so this file exists to say exactly how it was made.

THIS IS A ONE-TIME STEP. IT HAS ALREADY RUN. DO NOT RUN IT AGAIN.

  - src/data/naam/source-text.txt is committed. It is the input of record for
    scripts/naam/build-dataset.mjs, which asserts its sha256 on every build.
  - Boy_Name_Candidates.pdf is *untracked* (the repo is public; publishing the
    family's 172-page research PDF is a decision that has not been made). So a
    clone cannot re-run this script, and does not need to: the dump is in the
    tree and the hashes below detect any drift.
  - Re-running under a different pypdf version would almost certainly produce a
    byte-different dump, break the sha256 assertion, and invalidate every page
    citation on the page. If the PDF is ever regenerated, run this deliberately,
    re-run `npm run naam:build`, and update the frozen hashes and counts in
    scripts/naam/build-dataset.mjs in the same commit.

Inputs / outputs, both hashed by the build:

  Boy_Name_Candidates.pdf       sha256 295e3bb094b70e36e52a198edf3e08ea7da44cfb0a37ffaa578be951d96c5497
  src/data/naam/source-text.txt sha256 ef684850d536685f9cf341afcc769cc2c0eda186e0888155888b5335e208b701
                                373,031 bytes, 172 pages

Environment it ran in:

  python3 -m pip install pypdf
  python3 scripts/naam/extract-text.py

Page markers. Each page is emitted as, literally:

    \\n<<<PAGE 12>>>\\nBoy Name Candidates - Vedic & Theravada\\np. 12\\n<page text>

The first two lines after the marker are the PDF's own running header, which
pypdf repeats on every page. build-dataset.mjs strips that triplet but keeps the
page number on every row, because the page number is the citation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / 'Boy_Name_Candidates.pdf'
OUT = REPO / 'src' / 'data' / 'naam' / 'source-text.txt'

# Frozen. Also asserted by scripts/naam/build-dataset.mjs.
PDF_SHA256 = '295e3bb094b70e36e52a198edf3e08ea7da44cfb0a37ffaa578be951d96c5497'
TEXT_SHA256 = 'ef684850d536685f9cf341afcc769cc2c0eda186e0888155888b5335e208b701'

RUNNING_HEADER = 'Boy Name Candidates - Vedic & Theravada'


def extract(pdf_path: Path) -> str:
    from pypdf import PdfReader  # noqa: PLC0415 — optional, one-time dependency

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        chunks.append(f'<<<PAGE {index}>>>\n{RUNNING_HEADER}\np. {index}\n{page.extract_text()}')
    return '\n'.join(chunks)


def main() -> int:
    if '--i-really-mean-it' not in sys.argv:
        print(__doc__)
        print('Refusing to run. This is a one-time step and its output is committed.')
        print('If you genuinely need to regenerate, pass --i-really-mean-it and expect')
        print('to update the frozen hashes in scripts/naam/build-dataset.mjs.')
        return 1

    if not PDF.exists():
        print(f'missing {PDF} — the PDF is untracked, so a clone will not have it.')
        return 1

    got = hashlib.sha256(PDF.read_bytes()).hexdigest()
    if got != PDF_SHA256:
        print(f'PDF sha256 drift: expected {PDF_SHA256}, got {got}')
        return 1

    text = extract(PDF)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding='utf-8')
    print(f'wrote {OUT} ({len(text.encode("utf-8")):,} bytes)')
    print(f'sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}  (frozen: {TEXT_SHA256})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
