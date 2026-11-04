"""Layout check of the slide deck without rendering it (no LibreOffice on this machine).

For every shape: it must lie inside the slide. For every text box: estimate how many lines its text wraps to
(average glyph width ~0.52 x font size for a sans font) and fail if the estimated height overflows the box by
more than the space to the next shape below / the footer. Tables: estimated row heights must fit above the footer.
Exit 0 only if no problems are found.  Usage: python check_pptx.py [deck.pptx]
"""
import math
import os
import sys

from pptx import Presentation
from pptx.util import Emu, Pt

DECK = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "deliverables", "DualNet-AL_Midterm.pptx"))
EMU_PER_PT = 12700
FOOTER_TOP = Emu(int(7.0 * 914400))


def text_height_pt(tf, width_emu):
    width_pt = width_emu / EMU_PER_PT - 14  # default left+right insets ~0.1in each
    total = 0.0
    for para in tf.paragraphs:
        size = max([r.font.size.pt for r in para.runs if r.font.size] or [18])
        chars = sum(len(r.text) for r in para.runs) or 1
        per_line = max(int(width_pt / (0.52 * size)), 1)
        lines = math.ceil(chars / per_line)
        space_after = para.space_after.pt if para.space_after is not None else 0
        total += lines * size * 1.2 + space_after
    return total + 7  # top+bottom insets


def main():
    p = Presentation(DECK)
    W, H = p.slide_width, p.slide_height
    problems = []
    for i, s in enumerate(p.slides, 1):
        shapes = list(s.shapes)
        for sh in shapes:
            if sh.left < 0 or sh.top < 0 or sh.left + sh.width > W + 1 or sh.top + sh.height > H + 1:
                problems.append(f"slide {i}: '{sh.name}' extends outside the slide")
            if sh.has_text_frame and sh.text_frame.text.strip():
                need = text_height_pt(sh.text_frame, sh.width)
                top_pt = sh.top / EMU_PER_PT
                # room = until the next shape that starts below this one, or the footer line
                below = [o.top for o in shapes if o is not sh and o.top > sh.top + Emu(int(Pt(4)))
                         and o.left < sh.left + sh.width and o.left + o.width > sh.left]
                limit = min(below + [FOOTER_TOP if sh.top < FOOTER_TOP else H]) / EMU_PER_PT
                if top_pt + need > limit + 2:
                    problems.append(f"slide {i}: text '{sh.text_frame.text[:40]}…' needs ~{need:.0f}pt but only "
                                    f"{limit - top_pt:.0f}pt available")
            if sh.has_table:
                tbl = sh.table
                est = 0.0
                for r in tbl.rows:
                    row_h = 0.0
                    for c_i, cell in enumerate(r.cells):
                        row_h = max(row_h, text_height_pt(cell.text_frame, tbl.columns[c_i].width))
                    est += row_h
                if sh.top / EMU_PER_PT + est > FOOTER_TOP / EMU_PER_PT + 2:
                    problems.append(f"slide {i}: table needs ~{est:.0f}pt, runs into the footer")
        pics = [sh for sh in shapes if sh.shape_type == 13]
        print(f"slide {i:2d}: {len(shapes):2d} shapes, {len(pics)} picture(s)")
    print(f"{len(p.slides)} slides checked, {len(problems)} problem(s)")
    for pr in problems:
        print("  PROBLEM:", pr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
