"""Render every page of a PDF and run automated layout checks; the record is completed by a person's visual inspection.

Writes, into OUT_DIR: page_NNN.png (every page), contact_NN.png (pages in sheets of 6 for the visual pass) and
automated_checks.json (file name, SHA-256, page count, page sizes, per-page text length, text blocks within 5 pt of an
edge, overlapping text spans, header and footer lines, leftover template placeholders). Uses PyMuPDF.

    python -m src.utils.pdf_inspect PDF OUT_DIR [--dpi 100]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz
from PIL import Image


def overlaps(page) -> int:
    spans = [fitz.Rect(s["bbox"]) for b in page.get_text("dict")["blocks"] for line in b.get("lines", [])
             for s in line["spans"] if s["text"].strip()]
    n = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            r = spans[i] & spans[j]
            if not r.is_empty and r.get_area() > 4 and r.get_area() > 0.4 * min(spans[i].get_area(), spans[j].get_area()):
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise SystemExit(f"{a.out_dir} exists; refusing to overwrite")
    a.out_dir.mkdir(parents=True)
    doc = fitz.open(a.pdf)
    pages = []
    for i, pg in enumerate(doc):
        W, H = pg.rect.width, pg.rect.height
        text = pg.get_text()
        lines = [s for s in text.splitlines() if s.strip()]
        edge = [[round(x, 1) for x in b[:4]] for b in pg.get_text("blocks")
                if b[4].strip() and (b[0] < 5 or b[1] < 5 or b[2] > W - 5 or b[3] > H - 5)]
        pg.get_pixmap(dpi=a.dpi).save(a.out_dir / f"page_{i + 1:03d}.png")
        pages.append({"page": i + 1, "size_pt": [round(W), round(H)], "chars": len(text), "edge_blocks": edge,
                      "overlapping_spans": overlaps(pg), "first_line": lines[0][:100] if lines else "",
                      "last_line": lines[-1][:100] if lines else "", "placeholder": "{{" in text})
    ims = [Image.open(a.out_dir / f"page_{i + 1:03d}.png") for i in range(doc.page_count)]
    w, h = max(im.size[0] for im in ims), max(im.size[1] for im in ims)
    for s in range(0, len(ims), 6):
        sheet = Image.new("RGB", (w * 3, h * 2), "white")
        for k, im in enumerate(ims[s:s + 6]):
            sheet.paste(im, ((k % 3) * w, (k // 3) * h))
        sheet.save(a.out_dir / f"contact_{s // 6 + 1:02d}.png")
    rec = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "file": a.pdf.name, "path": str(a.pdf),
           "sha256": hashlib.sha256(a.pdf.read_bytes()).hexdigest(), "pages": doc.page_count, "dpi": a.dpi,
           "command": " ".join([Path(sys.executable).name, "-m", "src.utils.pdf_inspect"] + sys.argv[1:]),
           "pages_with_edge_blocks": [p["page"] for p in pages if p["edge_blocks"]],
           "pages_with_overlapping_spans": [p["page"] for p in pages if p["overlapping_spans"]],
           "pages_with_placeholders": [p["page"] for p in pages if p["placeholder"]],
           "pages_with_little_text": [p["page"] for p in pages if p["chars"] < 200],
           "page_sizes": sorted({tuple(p["size_pt"]) for p in pages}), "per_page": pages}
    (a.out_dir / "automated_checks.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print({k: rec[k] for k in ("file", "sha256", "pages", "pages_with_edge_blocks", "pages_with_overlapping_spans",
                                "pages_with_placeholders", "pages_with_little_text", "page_sizes")})


if __name__ == "__main__":
    main()
