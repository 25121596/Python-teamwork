"""Render each slide to PNG via PowerPoint COM and report overflow/overlap risks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "deliverables" / "项目演示.pptx"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "ppt_render"

prs = Presentation(PPTX)
SW, SH = prs.slide_width, prs.slide_height
EMU_IN = 914400
problems = []

for i, slide in enumerate(prs.slides, 1):
    boxes = []
    for shape in slide.shapes:
        if shape.left is None or shape.top is None:
            continue
        l, t = shape.left, shape.top
        r, b = l + (shape.width or 0), t + (shape.height or 0)
        if r > SW + 1000 or b > SH + 1000 or l < -1000 or t < -1000:
            problems.append({
                "slide": i, "kind": "out_of_canvas",
                "name": shape.name,
                "box_in": [round(l / EMU_IN, 2), round(t / EMU_IN, 2),
                           round(r / EMU_IN, 2), round(b / EMU_IN, 2)],
            })
        # estimate text overflow for text frames
        if shape.has_text_frame and shape.text_frame.text.strip():
            text = shape.text_frame.text
            size = None
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    if run.font.size:
                        size = run.font.size.pt
                        break
                if size:
                    break
            size = size or 12
            width_in = (shape.width or 0) / EMU_IN
            height_in = (shape.height or 0) / EMU_IN
            cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            ascii_n = len(text) - cjk
            # approximate: CJK glyph ~1.0em wide, ascii ~0.55em
            per_line = max(1, int(width_in * 72 / (size * (cjk / max(1, cjk + ascii_n) + 0.55 * ascii_n / max(1, cjk + ascii_n)))))
            lines_needed = 0
            for raw in text.split("\n"):
                c = sum(1 for ch in raw if "\u4e00" <= ch <= "\u9fff")
                a = len(raw) - c
                eff = c + 0.55 * a
                lines_needed += max(1, int(eff / per_line) + (1 if eff % per_line else 0))
            needed_h = lines_needed * size * 1.25 / 72
            if needed_h > height_in * 1.45 and needed_h - height_in > 0.18:
                problems.append({
                    "slide": i, "kind": "text_overflow_risk",
                    "name": shape.name,
                    "text": text[:70].replace("\n", " / "),
                    "size": size, "box_h_in": round(height_in, 2),
                    "needed_h_in": round(needed_h, 2),
                })
        boxes.append((shape.name, l, t, r, b))

print(json.dumps({
    "slides": len(prs.slides.__iter__.__self__._sldIdLst),
    "canvas_in": [round(SW / EMU_IN, 2), round(SH / EMU_IN, 2)],
    "problems": problems,
}, ensure_ascii=False, indent=2))
