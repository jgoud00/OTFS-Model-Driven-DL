"""Render each PDF page to PNG for visual inspection."""
import sys
from pathlib import Path

import fitz  # pymupdf

src = Path(sys.argv[1]).resolve()
outdir = src.parent / (src.stem + "_pages")
outdir.mkdir(exist_ok=True)
doc = fitz.open(src)
print(f"{len(doc)} pages -> {outdir}")
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=100)
    pix.save(outdir / f"page{i}.png")
