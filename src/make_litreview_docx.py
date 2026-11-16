"""Generate Literature_Review_and_Proposed_Solution.docx.

Concise Word report combining the audited literature review (round-4
evidence standard) with the proposed solution (spec v2.3). Content is
verbatim from the frozen Review-1 deck + spec — verified numbers only.
Figures are the three verified Stage-1 I/O demonstration PNGs.
Design tokens match the deck: #00684A headings, #EAF7F0 cards, Segoe UI.
References are parsed straight from literature/references.bib (no manual
transcription). Claim discipline: no "first/novel/state of the art".
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(".")
BIB = ROOT / "literature/references.bib"
FIG1 = ROOT / "results/fig1_input.png"
FIG2 = ROOT / "results/fig2_channel.png"
FIG3 = ROOT / "results/fig3_output.png"
OUT = ROOT / "Literature_Review_and_Proposed_Solution.docx"

GREEN = RGBColor(0x00, 0x68, 0x4A)   # dark green (headings)
INK = RGBColor(0x00, 0x1E, 0x2B)     # near-black (body)
GREY = RGBColor(0x66, 0x77, 0x77)    # captions
PALE = "EAF7F0"                      # pale green fill (cards/headers)

# ---------- bib parsing (same helpers as the downloader scripts) ----------
bib_text = BIB.read_text(encoding="utf-8")
raw = re.findall(r"@(\w+)\s*\{([^,]+),((?:[^@])*)\}", bib_text)
BIBS = {}
for etype, key, body in raw:
    def fld(name):
        m = re.search(name + r"\s*=\s*\{([^}]*)\}", body)
        return m.group(1).strip() if m else ""

    # titles contain nested braces ({OTFS}) — extract brace-aware so the
    # text after an inner brace is not dropped
    tm = re.search(r"title\s*=\s*\{", body)
    title = ""
    if tm:
        depth, i, out = 1, tm.end(), []
        while i < len(body) and depth:
            c = body[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth:
                out.append(c)
            i += 1
        title = "".join(out)
    title = title.replace("{", "").replace("}", "")
    author = fld("author").replace("{", "").replace("}", "")
    author = author.replace(" and others", " et al.")
    if author and not author.endswith("."):
        author += "."
    BIBS[key.strip()] = {
        "venue": fld("journal") or fld("booktitle") or fld("howpublished"),
        "year": fld("year"), "doi": fld("doi"), "author": author,
        "title": title,
    }

SELECTED = [
    "hadani2017otfs", "raviteja2018icid", "raviteja2019pilot",
    "wei2021promising", "samuel2019learning", "he2020modeldriven",
    "gao2020comnet", "zhuang2024ampgnn", "xu2024rc2d", "chong2024crossdomain",
    "yue2024modeldriven", "pnp2025", "leia2026sip", "duan2026sparse",
    "duan2026fractional", "gao2025joint", "ispir2026refine",
    "aslandogan2025survey",
]

# ---------- docx helpers ----------
doc = Document()
for sec in doc.sections:  # A4, comfortable margins
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
normal = doc.styles["Normal"]
normal.font.name = "Segoe UI"
normal.font.size = Pt(10.5)
normal.font.color.rgb = INK


def _run(p, text, size, bold=False, italic=False, color=INK):
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(size)
    r.bold, r.italic = bold, italic
    r.font.color.rgb = color
    return r


def p(text, size=10.5, italic=False, after=6, color=INK):
    par = doc.add_paragraph()
    par.paragraph_format.space_after = Pt(after)
    _run(par, text, size, italic=italic, color=color)
    return par


def bullet(text, size=10):
    par = doc.add_paragraph(style="List Bullet")
    par.paragraph_format.space_after = Pt(2)
    _run(par, text, size)
    return par


def h1(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(14), Pt(5)
    _run(par, text, 15, bold=True, color=GREEN)
    return par


def h2(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(10), Pt(3)
    _run(par, text, 12, bold=True, color=GREEN)
    return par


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(headers, rows, widths_cm, size=9.5):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for row in t.rows:  # never split a row across a page break
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
    for j, (h, w) in enumerate(zip(headers, widths_cm)):
        c = t.cell(0, j)
        c.width = Cm(w)
        shade(c, PALE)
        cp = c.paragraphs[0]
        cp.paragraph_format.space_after = Pt(0)
        _run(cp, h, size, bold=True, color=GREEN)
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths_cm[j])
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(0)
            _run(cp, val, size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width))
    pic_par = doc.paragraphs[-1]
    pic_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_par.paragraph_format.keep_with_next = True  # keep caption on same page
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    _run(cap, caption, 8.5, italic=True, color=GREY)


def quotebox(text):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.cell(0, 0)
    c.width = Cm(17)
    shade(c, PALE)
    cp = c.paragraphs[0]
    cp.paragraph_format.space_after = Pt(0)
    _run(cp, text, 10.5, italic=True)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ---------- title block ----------
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_after = Pt(2)
_run(tp, "Model-Driven Deep Learning for OTFS Modulation", 20, bold=True, color=GREEN)
sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
_run(sp, "Literature Review and Proposed Solution — Review 1", 12, italic=True)
ap = doc.add_paragraph()
ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
ap.paragraph_format.space_after = Pt(4)
_run(ap, "[Student Name]  ·  [Roll No.]  ·  Guide: [Guide Name]", 10)
bp = doc.add_paragraph()
bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
bp.paragraph_format.space_after = Pt(10)
_run(bp, "KLH University · MCS · November 2026", 9.5, color=GREY)

# ---------- 1. Introduction ----------
h1("1.  Introduction and Problem Statement")
p("High-mobility wireless channels are doubly dispersive: multipath creates delay "
  "spread, and motion creates Doppler spread. OFDM handles multipath with a cyclic "
  "prefix, but Doppler shifts subcarriers off their frequencies and destroys "
  "orthogonality (inter-carrier interference). The 6G regimes that make this acute "
  "include 500 km/h high-speed rail, vehicle-to-everything, UAVs and LEO satellites "
  "(e.g., 500 km/h at 3.5 GHz gives a Doppler shift of about 1.6 kHz).")
p("OTFS modulation addresses this by placing data on a two-dimensional delay–Doppler "
  "grid, which is then transformed to a time–frequency waveform. Because a physical "
  "channel has few paths, the channel is sparse in the delay–Doppler domain, and "
  "every data symbol experiences approximately the whole channel rather than one "
  "fading snapshot. Declared caveats: rectangular pulses add interference, and "
  "fractional Doppler smears paths across neighbouring Doppler bins.")
figure(FIG1, "Figure 1 — Transmitter side at a glance: QPSK data constellation, delay–Doppler "
             "resource map (pilot, guards, 179 data cells) and the resulting time waveform "
             "(verified Stage-1 demonstration).")

# ---------- 2. Methodology ----------
h1("2.  Survey Methodology")
for b in [
    "About 160 arXiv and 300 Crossref records screened → 73 references, "
    "reconciled one-to-one with the BibTeX file.",
    "Four independent audit rounds; all findings accepted; conventions verified "
    "against the primary source (Raviteja et al., IEEE TWC 2018, full text).",
    "Evidence policy: “verified”, “unverified” and “not reported” are kept as "
    "distinct labels; no novelty language; negative results are reportable.",
]:
    bullet(b)

# ---------- 3. Timeline ----------
h1("3.  Literature Timeline (2017–2026)")
table(
    ["Period", "What happened", "Representative works"],
    [
        ["2017", "OTFS introduced", "Hadani et al. (WCNC)"],
        ["2018–19", "Discrete delay–Doppler model, iterative detection, "
                    "embedded-pilot channel estimation", "Raviteja et al. (TWC/TVT)"],
        ["2021–23", "Data-driven wave: DNN, GNN and 2D-convolutional detectors; "
                    "CNN estimators", "several conference/journal works"],
        ["2024", "Model-driven deep unfolding reaches OTFS: unfolded detector "
                 "under channel-estimation error", "Yue et al. (Comm. Letters)"],
        ["2025–26", "Hybrid joint receivers (plug-and-play), unfolding for "
                    "fractional-Doppler estimation, model-driven detection under "
                    "phase noise", "PnP 2025; Duan et al.; CL/TVT 2026"],
    ],
    [2.2, 9.0, 5.8],
)

# ---------- 4. Landscape ----------
h1("4.  The Model-Driven Landscape")
p("Under our stated counting rule, about ten works (2024–2026) qualify as explicit "
  "deep unfolding for OTFS — a young niche. Almost all unroll one module only: the "
  "estimator or the detector, with the other module given. Joint estimation and "
  "detection exists in hybrid (optimization plus learned priors) and data-driven "
  "forms, but not as jointly trained unfolding. Fractional Doppler is handled on the "
  "estimation side; an unfolded detector operating under fractional Doppler with "
  "estimated CSI is not documented in the checked sources.")

# ---------- 5. Gaps ----------
h1("5.  Headline Gaps in the Literature")
for i, g in enumerate([
    "Joint training of two unfolded algorithms through a differentiable channel "
    "model — unverified in the audited literature.",
    "Detector-side fractional Doppler for the unfolding line — open.",
    "Training-time versus evaluation-time robustness to CSI error — conflated "
    "in the field.",
    "Systematic generalization splits (SNR transfer, few-shot, channel transfer) "
    "— unclaimed.",
    "Complexity reporting uneven across the unfolding line.",
    "No shared benchmark identified (public toolboxes exist; no common protocol).",
], 1):
    bullet(f"Gap {i}: {g}")

# ---------- 6. Verified demonstration ----------
h1("6.  Verified Baseline Demonstration")
p("The simulation chain is built and verified before any learning is attempted "
  "(stage gates are measured, not asserted):")
for b in [
    "Stage-1 simulator: transform identity < 10⁻¹²; noiseless end-to-end chain "
    "error 2.22×10⁻¹⁵ over 5 seeds; unitary transforms (mean sample energy 1.0).",
    "Full input–output path on the 32×8 delay–Doppler grid: 358 bits → 179 QPSK "
    "data cells → 264 time samples → detected bits; BER = 0.0000 at 10 and 20 dB "
    "(true-CSI smoke test); about 1.36 bits/sample spectral efficiency.",
]:
    bullet(b)
figure(FIG2, "Figure 2 — True sparse channel taps (left) and the noiseless received "
             "delay–Doppler grid (right): sparsity is what the estimator exploits.")
figure(FIG3, "Figure 3 — Received grid at 10 dB SNR (left) and detector output "
             "constellation with data-position BER (right).")

# ---------- 7. Proposed solution ----------
h1("7.  Proposed Solution: UOR-Net")
p("UOR-Net is not claimed to be unique; the contribution is a controlled experiment "
  "that tests an open, testable question.", italic=True)
h2("7.1  Research question")
quotebox("Does allowing the detection loss to update the unfolded channel estimator — "
         "through a differentiable delay–Doppler channel operator — improve data-bit "
         "BER relative to separate training of the same architecture, under matched "
         "fractional-Doppler channels and pilot-estimated CSI?")
p("A negative result is a reportable result.", italic=True, after=8)
h2("7.2  Architecture (20 unfolded layers, ~40 learnable parameters)")
for b in [
    "Block A — unfolded ISTA channel estimator: 10 layers, learnable step size and "
    "threshold per layer.",
    "Block B — differentiable delay–Doppler channel operator: Ĥ = Σ ĥ_j H_j "
    "(weighted atoms); the only gradient path from the detection loss to the "
    "estimator.",
    "Block C — unfolded OAMP detector: 10 layers, learnable variance and damping "
    "per layer.",
    "Output: logsumexp LLRs → soft bits → hard bits; detection loss = binary "
    "cross-entropy on data bits only.",
]:
    bullet(b)
h2("7.3  Design rules that make it trustworthy")
for b in [
    "Parity at initialization: every learnable scalar starts at its classical "
    "value; a unit test proves the network equals the classical algorithm before "
    "any training (difference < 10⁻⁶).",
    "Operator-space channel supervision with probe vectors — honest when off-grid "
    "truth has no unique label.",
    "Honest OAMP caveat: state evolution is approximate for structured channels, "
    "so variances are learnable and sensitivity is reported.",
]:
    bullet(b)
h2("7.4  Pre-registered experiment design")
for b in [
    "One decisive grid: 32×8 delay–Doppler, QPSK (16×4 debug grid).",
    "Paired separate-vs-joint training with identical architecture, initialization "
    "and data budget; at least 5 seeds.",
    "50,000 procedurally generated frames — no external dataset exists for OTFS "
    "DL (itself gap 6); exact ground truth comes from the generator.",
    "Statistical protocol: at least 10⁶ bits and at least 100 error events per "
    "BER point, with 95% Clopper–Pearson confidence intervals.",
    "Controls: held-out Doppler-offset band, CSI-error sweep, and an "
    "inverse-crime check with an independent generator.",
    "Fine-tune route: no public OTFS deep-learning checkpoint exists (verified at "
    "repository file level), so a module is pretrained on source simulations and "
    "fine-tuned on a held-out fractional-Doppler distribution (frozen vs "
    "fine-tuned vs from-scratch) — which simultaneously delivers the "
    "generalization study (gap 4).",
]:
    bullet(b)

# ---------- 8. Conclusion ----------
h1("8.  Conclusions and Plan")
for b in [
    "The survey is audited (73 references, four rounds) and the gaps are concrete; "
    "the open research question stands: no counter-example was found in a "
    "cross-check against an independent AI-assisted survey (34 references, 7 "
    "load-bearing citations verified).",
    "Ten-week plan trimmed to a trimester-compatible core: simulator → baselines "
    "→ unfolded estimator → unfolded detector → joint experiment; the report is "
    "written as stages complete, and every stage ends with a measured gate.",
    "Deliverables so far: audited survey, governing spec v2.3, verified simulator "
    "and I/O demonstration, research-paper draft (pre-registered), study guide.",
    "Repository: github.com/jgoud00/OTFS-Model-Driven-DL (44/73 references also "
    "downloaded as open-access PDFs; the rest listed for campus access).",
]:
    bullet(b)

# ---------- References ----------
h1("References  (selected, 18 of 73)")
p("The full 73-entry bibliography is maintained in literature/references.bib.",
  size=9, italic=True, after=4, color=GREY)
for i, key in enumerate(SELECTED, 1):
    e = BIBS[key]
    bits = []
    if e["author"]:
        bits.append(e["author"].replace("{", "").replace("}", ""))
    bits.append(f"“{e['title']},”" if not e["title"].endswith(",") else f"“{e['title']}")
    if e["venue"]:
        bits.append(f"{e['venue'].replace('{','').replace('}','')}, {e['year']}.")
    else:
        bits.append(f"{e['year']}.")
    line = " ".join(bits)
    if e["doi"]:
        line += f" doi: {e['doi']}."
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_after = Pt(3)
    pf.left_indent, pf.first_line_indent = Cm(0.55), Cm(-0.55)
    _run(par, f"[{i}]  {line}", 9)

doc.save(OUT)

# ---------- self-check ----------
chk = Document(OUT)
pics = len(chk.inline_shapes)
print(f"saved {OUT.name}: {OUT.stat().st_size//1024} KB, "
      f"{len(chk.paragraphs)} paragraphs, {len(chk.tables)} tables, {pics} figures")
missing = [k for k in SELECTED if k not in BIBS]
print("missing bib keys:", missing if missing else "none — all 18 resolved")
