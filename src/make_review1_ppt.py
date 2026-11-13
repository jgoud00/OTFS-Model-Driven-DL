# Review 1 deck: literature survey + proposed solution only.
# Palette: MongoDB greens — #00ED64 accent, #00684A deep, #EAF7F0 cards, #001E2B ink.
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

INK   = RGBColor(0x00, 0x1E, 0x2B)
GREEN = RGBColor(0x00, 0xED, 0x64)
DEEP  = RGBColor(0x00, 0x68, 0x4A)
CARD  = RGBColor(0xEA, 0xF7, 0xF0)
CARD2 = RGBColor(0xF3, 0xFB, 0xF7)
MUTED = RGBColor(0x43, 0x55, 0x4D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

F  = "Segoe UI"
FB = "Segoe UI Semibold"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide_new():
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background(); bg.shadow.inherit = False
    return s

def rect(s, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sp = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.fill.background(); sp.shadow.inherit = False
    if radius is not None:
        try: sp.adjustments[0] = radius
        except Exception: pass
    return sp

def card(s, x, y, w, h, fill=CARD):
    return rect(s, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE, 0.10)

def box(s, x, y, w, h):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf

def _para(tf):
    p0 = tf.paragraphs[0]
    return p0 if (len(tf.paragraphs) == 1 and not p0.runs) else tf.add_paragraph()

def para(tf, text, size=14, color=INK, bold=False, font=F, align=PP_ALIGN.LEFT, after=4, line=None):
    p = _para(tf); r = p.add_run(); r.text = text
    r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(size), color, bold, font
    p.alignment, p.space_after = align, Pt(after)
    if line: p.line_spacing = line
    return p

def runs_para(tf, specs, align=PP_ALIGN.LEFT, after=4, line=None):
    p = _para(tf)
    for text, size, color, bold, font in specs:
        r = p.add_run(); r.text = text
        r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(size), color, bold, font
    p.alignment, p.space_after = align, Pt(after)
    if line: p.line_spacing = line
    return p

def chip(s, x, y, w, h, text, fill=WHITE, color=DEEP, size=12, bold=True):
    c = rect(s, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE, 0.5)
    tf = c.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12); tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, text, size, color, bold, FB if bold else F, PP_ALIGN.CENTER, after=0)
    return c

def header(s, kicker, title, n):
    rect(s, 0.9, 0.62, 0.5, 0.09, GREEN)
    tf = box(s, 0.9, 0.84, 11.6, 1.05)
    para(tf, kicker.upper(), 11.5, DEEP, True, F, after=5)
    para(tf, title, 27, INK, True, FB, after=0)
    pn = box(s, 12.35, 7.08, 0.6, 0.3)
    para(pn, str(n), 10, MUTED, align=PP_ALIGN.RIGHT)

def cardtext(c, margin=0.25):
    tf = c.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.14)
    return tf

# ---------------------------------------------------------------- 1 · TITLE
s = slide_new()
panel = rect(s, 8.6, 0, 4.733, 7.5, CARD)
tf = box(s, 9.1, 1.15, 3.8, 0.4)
para(tf, "UOR-NET", 13, DEEP, True, F)
for i, t in enumerate(["A — Unfolded ISTA estimator", "B — DD channel operator", "C — Unfolded OAMP detector"]):
    chip(s, 9.1, 1.75 + i * 0.85, 3.75, 0.6, t)
tf = box(s, 9.1, 5.0, 3.8, 1.6)
para(tf, "Spread. Unfold. Detect.", 15, DEEP, True, FB, after=2)
para(tf, "One receiver, three blocks, ~hundreds of trainable scalars.", 11, MUTED)

rect(s, 0.9, 1.35, 0.6, 0.1, GREEN)
tf = box(s, 0.9, 1.62, 7.3, 0.4)
para(tf, "REVIEW 1  ·  MCS PROJECT", 13, DEEP, True, F)
tf = box(s, 0.9, 2.15, 7.4, 2.4)
para(tf, "Model-Driven Deep Learning", 40, INK, True, FB, after=2, line=1.02)
para(tf, "for OTFS Modulation", 40, INK, True, FB, after=0, line=1.02)
tf = box(s, 0.9, 4.45, 7.2, 0.8)
para(tf, "Literature survey & the proposed solution", 17, MUTED, after=0)
tf = box(s, 0.9, 5.9, 7.2, 1.1)
para(tf, "[Student Name]  ·  [Roll No.]", 13, INK, True, F, after=2)
para(tf, "Guide: [Guide Name]", 12, MUTED, after=2)
para(tf, "KLH University  ·  12 November 2026", 12, MUTED, after=0)

# ---------------------------------------------------------------- 2 · OUTLINE
s = slide_new(); header(s, "Agenda", "What Review 1 covers", 2)
items = [
    ("01", "Motivation", "why OTFS for high mobility"),
    ("02", "Survey method", "how the literature was collected & verified"),
    ("03", "Literature landscape", "paradigms · timeline · key works · comparison"),
    ("04", "Gap analysis", "six gaps the survey exposes"),
    ("05", "Proposed solution", "UOR-Net — three unfolded blocks"),
]
for i, (n, t, sub) in enumerate(items):
    c = card(s, 0.9, 1.95 + i * 0.98, 11.5, 0.82, CARD2 if i % 2 else CARD)
    tf = cardtext(c); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    runs_para(tf, [(n + "   ", 16, GREEN, True, FB), (t, 16, INK, True, FB),
                   ("   —   " + sub, 12.5, MUTED, False, F)], after=0)

# ---------------------------------------------------------------- 3 · MOTIVATION
s = slide_new(); header(s, "01 · Motivation", "Why OTFS for high mobility", 3)
tf = box(s, 0.9, 2.0, 7.0, 4.6)
for b in [
    "High mobility everywhere — 500 km/h rail, V2X, UAV swarms, LEO links",
    "Channels become doubly dispersive: delay and Doppler spreads both matter",
    "OFDM: Doppler destroys subcarrier orthogonality → inter-carrier interference",
    "OTFS: symbols spread over the whole time–frequency plane via the delay–Doppler transform",
    "Each path = one (delay, Doppler) point → compact channel view, every symbol sees the whole channel",
    "But receivers stay hard: message passing / AMP degrade on dense or off-grid channels",
]:
    runs_para(tf, [("— ", 14, GREEN, True, FB), (b, 14.5, INK, False, F)], after=10, line=1.08)
c = card(s, 8.35, 2.0, 4.05, 4.0)
tf = cardtext(c)
para(tf, "WHERE OTFS IS HEADING", 11.5, DEEP, True, F, after=8)
for b in ["6G NTN waveform candidate", "Integrated sensing & communication (ISAC)", "LEO satellite downlinks", "mmWave links"]:
    runs_para(tf, [("· ", 13, GREEN, True, FB), (b, 13.5, INK, False, F)], after=8)

# ---------------------------------------------------------------- 4 · SURVEY METHOD
s = slide_new(); header(s, "02 · Survey method", "How the literature was collected", 4)
stats = [("≈160", "arXiv records screened"), ("≈300", "Crossref records screened"),
         ("73", "references retained"), ("1", "evidence standard for all tables")]
for i, (big, sub) in enumerate(stats):
    c = card(s, 0.9 + i * 2.95, 2.0, 2.7, 1.7)
    tf = cardtext(c)
    para(tf, big, 30, DEEP, True, FB, after=2)
    para(tf, sub, 11.5, MUTED, after=0, line=1.05)
c = card(s, 0.9, 4.05, 5.6, 2.5)
tf = cardtext(c)
para(tf, "SEARCHES & VERIFICATION", 11.5, DEEP, True, F, after=6)
for b in ["Run 12 Nov 2026 — arXiv API, Crossref, HF Papers", "Core entries DOI-verified via Crossref",
          "Key claims checked against primary full texts (e.g., Raviteja's delay/Doppler axis convention verified from the paper itself)"]:
    runs_para(tf, [("· ", 12.5, GREEN, True, FB), (b, 12.5, INK, False, F)], after=7, line=1.1)
c = card(s, 6.8, 4.05, 5.6, 2.5)
tf = cardtext(c)
para(tf, "COUNTING RULE (WHAT COUNTS AS MODEL-DRIVEN)", 11.5, DEEP, True, F, after=6)
for b in ["Explicit unfolding: iterative algorithm → layers with trainable scalars",
          "Hybrid: model pieces inside a learned net (PnP, AMP+GNN) — not unfolding",
          "Every table cell labelled: verified · unverified · not reported"]:
    runs_para(tf, [("· ", 12.5, GREEN, True, FB), (b, 12.5, INK, False, F)], after=7, line=1.1)

# ---------------------------------------------------------------- 5 · PARADIGMS
s = slide_new(); header(s, "03 · Literature landscape", "Three receiver-design paradigms", 5)
cards = [
    ("MODEL-BASED", "MP · AMP · VB estimators", "Interpretable, provable, little data — but hand-tuned and degrade when their approximations break."),
    ("DATA-DRIVEN", "DNN · CNN · GNN", "Learns end to end — but opaque, data-hungry, and retrained per configuration."),
    ("MODEL-DRIVEN", "Deep unfolding", "Iterations become layers with a few learnable scalars — small networks, less data, physics kept."),
]
for i, (tag, sub, body) in enumerate(cards):
    c = card(s, 0.9 + i * 4.15, 2.05, 3.9, 3.55, CARD if i == 2 else CARD2)
    tf = cardtext(c)
    para(tf, tag, 12.5, DEEP, True, F, after=4)
    para(tf, sub, 17, INK, True, FB, after=8)
    para(tf, body, 12.5, MUTED, after=0, line=1.15)
    if i == 2:
        chip(s, 0.9 + i * 4.15 + 1.05, 5.05, 1.8, 0.4, "THIS PROJECT")
c = card(s, 0.9, 6.0, 11.5, 0.8, CARD2)
tf = cardtext(c); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
runs_para(tf, [("Takeaway:  ", 13, DEEP, True, FB),
               ("model-driven design keeps the physics and learns only the handful of scalars that matter.", 13, INK, False, F)], after=0)

# ---------------------------------------------------------------- 6 · TIMELINE
s = slide_new(); header(s, "03 · Literature landscape", "Six years in six steps", 6)
rect(s, 1.0, 3.42, 11.4, 0.035, GREEN)
nodes = [
    ("2017", "OTFS proposed — Hadani et al. (WCNC)"),
    ("2018", "Discrete DD model + iterative detection — Raviteja et al."),
    ("2021", "OAMP & TMP receivers formalised — Wei et al."),
    ("2021–23", "First data-driven OTFS receivers (DNN / GNN / CNN)"),
    ("2024", "First model-driven OTFS detector — Yue et al."),
    ("2025–26", "Joint CE+DD frameworks · fractional-Doppler unfolding (Duan et al.)"),
]
xs = [1.4, 3.5, 5.6, 7.7, 9.8, 11.9]
for (yr, txt), x in zip(nodes, xs):
    dot = rect(s, x - 0.09, 3.34, 0.18, 0.18, DEEP, MSO_SHAPE.OVAL)
    tf = box(s, x - 1.0, 2.75, 2.0, 0.45)
    para(tf, yr, 14, DEEP, True, FB, PP_ALIGN.CENTER, after=0)
    tf = box(s, x - 1.02, 3.75, 2.05, 1.7)
    para(tf, txt, 10.5, INK, False, F, PP_ALIGN.CENTER, after=0, line=1.1)
tf = box(s, 0.9, 5.9, 11.5, 0.5)
para(tf, "The model-driven line is the newest — barely two years old.", 13, MUTED, False, F, PP_ALIGN.CENTER)

# ---------------------------------------------------------------- 7 · KEY WORKS
s = slide_new(); header(s, "03 · Literature landscape", "Who does what (2021–2026)", 7)
c = card(s, 0.9, 2.0, 5.6, 3.75, CARD2)
tf = cardtext(c)
para(tf, "DATA-DRIVEN RECEIVERS", 12, DEEP, True, F, after=8)
for b in ["DNN / GNN / reservoir-computing detectors", "CNN & RNN fractional-Doppler estimators",
          "Plug-and-Play joint CE + detection — TCOM 2025", "Lightweight joint framework — IoT-J 2025",
          "Environment-robust one-sample adaptation — GLOBECOM 2022"]:
    runs_para(tf, [("· ", 13, GREEN, True, FB), (b, 13, INK, False, F)], after=8, line=1.1)
c = card(s, 6.8, 2.0, 5.6, 3.75, CARD)
tf = cardtext(c)
para(tf, "MODEL-DRIVEN / UNFOLDING", 12, DEEP, True, F, after=8)
for b in ["Unfolded detector, robust to CE error — IEEE CL 2024 [36]",
          "Fractional-Doppler CE unfolding — TVT 2026 [39]",
          "FISTA sparse CE unfolding — WCL 2026 [40]",
          "Phase-noise model-driven detector — CL 2026 [38]",
          "LAMP for MIMO-OTFS CE [42] · ISAC target estimation [41]"]:
    runs_para(tf, [("· ", 13, GREEN, True, FB), (b, 13, INK, False, F)], after=8, line=1.1)
c = card(s, 0.9, 6.05, 11.5, 0.8, CARD2)
tf = cardtext(c); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
runs_para(tf, [("Pattern:  ", 13, DEEP, True, FB),
               ("almost every work unfolds ONE module — the other is treated as given.", 13, INK, False, F)], after=0)

# ---------------------------------------------------------------- 8 · COMPARISON TABLE
s = slide_new(); header(s, "03 · Literature landscape", "Comparative snapshot", 8)
rows, cols = 6, 5
tbl = s.shapes.add_table(rows, cols, Inches(0.9), Inches(2.0), Inches(11.5), Inches(3.9)).table
tbl.first_row = False; tbl.horz_banding = False
for i, w in enumerate([2.1, 1.6, 3.0, 2.3, 2.5]):
    tbl.columns[i].width = Inches(w)
data = [
    ["Work", "Venue", "Unfolded module", "Fractional Doppler", "CSI-error robustness"],
    ["Yue et al. '24", "IEEE CL", "Detection", "On-grid", "Trained against CE error"],
    ["Zhang et al. '25", "TCOM", "CE + det (hybrid PnP)", "Yes", "Sweep @ 12 dB SNR"],
    ["Zhuang et al. '24", "arXiv", "Detection (AMP + GNN)", "Yes", "Perturbed path gains"],
    ["Duan et al. '26", "TVT", "CE", "Yes", "Unverified"],
    ["Gao et al. '25", "IoT-J", "CE + det (data-driven)", "Unverified", "Unverified"],
]
for r in range(rows):
    tbl.rows[r].height = Inches(0.62)
    for c_i in range(cols):
        cell = tbl.cell(r, c_i)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.1)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DEEP if r == 0 else (CARD2 if r % 2 else WHITE)
        tf = cell.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; r_ = p.add_run(); r_.text = data[r][c_i]
        r_.font.name = FB if (r == 0 or c_i == 0) else F
        r_.font.size = Pt(12 if r == 0 else 11.5)
        r_.font.bold = (r == 0 or c_i == 0)
        r_.font.color.rgb = WHITE if r == 0 else (DEEP if c_i == 0 else INK)
tf = box(s, 0.9, 6.15, 11.5, 0.6)
para(tf, "One evidence standard: bold = verified against the source; “unverified” = not checkable at the audited level — never treated as absence.",
     11.5, MUTED, False, F, after=0)

# ---------------------------------------------------------------- 9 · GAPS
s = slide_new(); header(s, "04 · Gap analysis", "Six gaps the survey exposes", 9)
gaps = [
    ("Young, fragmented field", "Only ~10 explicit-unfolding works (2024–26); nearly all unfold a single module."),
    ("Joint training — unverified", "Hybrid & data-driven joint CE+DD exist; jointly unfolded CE + detector through a differentiable operator not found in checked sources."),
    ("Fractional Doppler — detector side", "Estimation side and hybrid detectors handle it; an unfolded detector under fractional Doppler + estimated CSI is not documented."),
    ("Robustness & generalization", "Training-time robustness vs evaluation-time sweeps rarely separated; no declared generalization splits."),
    ("No shared benchmark", "Public repos exist (OTFS toolbox, LSTM-CE), but no common waveform / pilot / protocol across papers; complexity reporting uneven."),
    ("No guarantees for learned receivers", "Algorithm-level proofs exist; convergence guarantees under off-grid Doppler + estimated CSI do not."),
]
for i, (t, b) in enumerate(gaps):
    x = 0.9 + (i % 2) * 6.0
    y = 2.0 + (i // 2) * 1.62
    c = card(s, x, y, 5.5, 1.44, CARD if i % 2 == 0 else CARD2)
    tf = cardtext(c, margin=0.2)
    runs_para(tf, [(t, 13, DEEP, True, FB)], after=3)
    para(tf, b, 11, INK, False, F, after=0, line=1.08)

# ---------------------------------------------------------------- 10 · RESEARCH QUESTION
s = slide_new(); header(s, "04 → 05", "From gaps to one testable question", 10)
c = card(s, 1.2, 2.1, 10.93, 3.3)
tf = cardtext(c, margin=0.5)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
para(tf, "“Does allowing detection loss to update the unfolded channel estimator improve data-bit BER — relative to separate training of the same architecture — under matched fractional-Doppler channels and pilot-estimated CSI?”",
     24, DEEP, True, FB, PP_ALIGN.CENTER, after=0, line=1.18)
tf = box(s, 1.2, 5.75, 10.93, 0.8)
para(tf, "Answered by a paired experiment: the identical architecture trained separately vs jointly — same blocks, same initialization, same data budget.",
     13.5, MUTED, False, F, PP_ALIGN.CENTER, after=0)

# ---------------------------------------------------------------- 11 · ARCHITECTURE
s = slide_new(); header(s, "05 · Proposed solution", "UOR-Net — three unfolded blocks", 11)
chip(s, 0.9, 1.95, 6.3, 0.5, "Received CP-OTFS frame  y", fill=CARD2, color=INK, size=12)
blocks = [
    ("BLOCK A — Unfolded sparse channel estimator (ISTA)",
     "learnable step γk & threshold λk per layer → coefficients over a fixed DD dictionary"),
    ("BLOCK B — Differentiable DD channel operator",
     "Ĥ = Σ ĥj · Hj  (weighted atoms, batched)  ·  pilot removed:  y′ = y − Ĥ x p"),
    ("BLOCK C — Unfolded OAMP detector",
     "G = Ĥ S d  ·  LMMSE filter with learnable variance  ·  QPSK posterior → LLRs (logsumexp)"),
]
ys = [2.6, 3.85, 5.1]
for (t, sub), y in zip(blocks, ys):
    c = card(s, 0.9, y, 6.3, 1.0, CARD)
    tf = cardtext(c, margin=0.22)
    para(tf, t, 12.5, DEEP, True, FB, after=2)
    para(tf, sub, 10.5, INK, False, F, after=0, line=1.05)
for y in [2.28, 3.53, 4.78]:
    rect(s, 3.9, y, 0.3, 0.26, GREEN, MSO_SHAPE.DOWN_ARROW)
chip(s, 0.9, 6.3, 6.3, 0.5, "Uncoded data-bit BER", fill=CARD2, color=INK, size=12)
c = card(s, 7.55, 2.6, 4.85, 4.2, CARD)
tf = cardtext(c)
para(tf, "THE EXPERIMENT — JOINT TRAINING", 12, DEEP, True, F, after=10)
para(tf, "L = α · L_CE + β · L_det", 20, INK, True, FB, PP_ALIGN.CENTER, after=10)
for b in ["Detection loss flows through Ĥ back into Block A",
          "Classical values at initialization → parity guaranteed by unit test",
          "Modest expected gains — the deliverable is the mechanism comparison, not a SOTA claim"]:
    runs_para(tf, [("· ", 12.5, GREEN, True, FB), (b, 12.5, INK, False, F)], after=9, line=1.12)

# ---------------------------------------------------------------- 12 · BLOCKS DETAIL
s = slide_new(); header(s, "05 · Proposed solution", "Inside the blocks", 12)
cols = [
    ("A · UNFOLDED ISTA", ["Fixed dictionary: integer delays × oversampled Dopplers",
                           "Soft-threshold with S(0) = 0; γk, λk learnable per layer",
                           "Supervised in operator space on probe vectors — no coefficient labels"]),
    ("B · CHANNEL OPERATOR", ["Ĥ = Σ ĥj Hj — applied batched, never materialized",
                              "Gradients reach every atom (no top-K selection)",
                              "Oracle dictionary-approximation error reported as the bound"]),
    ("C · UNFOLDED OAMP", ["LMMSE filter — dense reference first (D ≈ 240 @ 32×8 grid)",
                           "Per-layer learnable variance (state evolution treated as approximation)",
                           "BCE on data bits only; LLRs via logsumexp"]),
]
for i, (t, items) in enumerate(cols):
    c = card(s, 0.9 + i * 4.15, 2.05, 3.9, 4.0, CARD if i == 1 else CARD2)
    tf = cardtext(c)
    para(tf, t, 12.5, DEEP, True, F, after=10)
    for b in items:
        runs_para(tf, [("· ", 12.5, GREEN, True, FB), (b, 12.5, INK, False, F)], after=10, line=1.12)
c = card(s, 0.9, 6.35, 11.5, 0.7, CARD2)
tf = cardtext(c); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
runs_para(tf, [("Design rule:  ", 13, DEEP, True, FB),
               ("every non-trivial formula comes with a measured acceptance check — nothing is asserted without a unit test.", 12.5, INK, False, F)], after=0)

# ---------------------------------------------------------------- 13 · CONTRIBUTIONS → GAPS
s = slide_new(); header(s, "05 · Proposed solution", "Contributions → gaps addressed", 13)
rows13 = [
    ("Operator-space supervised unfolded channel estimator", "G3 · fractional Doppler"),
    ("Unfolded OAMP detector on the estimated operator", "G3 · detector side"),
    ("Paired separate-vs-joint training experiment", "G2 · joint training"),
    ("Reproducible benchmark: BER · NMSE · latency · memory", "G8/G9 · complexity & benchmark"),
]
for i, (c1, c2) in enumerate(rows13):
    y = 2.0 + i * 1.05
    a = card(s, 0.9, y, 7.2, 0.88, CARD)
    tf = cardtext(a); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, c1, 13.5, INK, True, F, after=0, line=1.0)
    rect(s, 8.35, y + 0.31, 0.3, 0.26, GREEN, MSO_SHAPE.RIGHT_ARROW)
    b = card(s, 8.9, y, 3.5, 0.88, CARD2)
    tf = cardtext(b); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, c2, 12, DEEP, True, F, PP_ALIGN.CENTER, after=0)
chip(s, 0.9, 6.4, 11.5, 0.6, "Status — Stage 1 complete: the OTFS simulation chain is verified to 2.2 × 10⁻¹⁵ error (unitary transforms).",
     fill=CARD, color=DEEP, size=12)

# ---------------------------------------------------------------- 14 · REFERENCES
s = slide_new(); header(s, "References", "Key references — 10 of 73", 14)
refs_l = [
    "Hadani et al., “Orthogonal Time Frequency Space Modulation,” IEEE WCNC, 2017.",
    "Raviteja et al., “Interference Cancellation and Iterative Detection for OTFS,” IEEE TWC, 2018.",
    "Raviteja et al., “Embedded Pilot-Aided Channel Estimation for OTFS,” IEEE TWC, 2019.",
    "Wei et al., “OTFS: A Promising Next-Generation Waveform,” IEEE Wireless Commun., 2021.",
    "He et al., “Model-Driven Deep Learning for MIMO Detection,” IEEE TSP, 2020.",
]
refs_r = [
    "Yue et al., “Model-Driven DL Assisted Detector for OTFS With Channel-Estimation Error,” IEEE CL, 2024.",
    "Zhang et al., “OTFS CE & Symbol Detection With Plug-and-Play Framework,” IEEE TCOM, 2025.",
    "Gao et al., “Joint CE & Data Detection for OTFS: Lightweight DL Framework,” IEEE IoT-J, 2025.",
    "Duan et al., “Physics-Aware Deep Unfolding for OTFS CE Under Fractional Doppler,” IEEE TVT, 2026.",
    "Zhuang et al., “AMP-Enhanced GNN for OTFS Data Detection,” arXiv:2402.10071, 2024.",
]
for x, lst in [(0.9, refs_l), (6.8, refs_r)]:
    tf = box(s, x, 2.1, 5.6, 4.3)
    for r in lst:
        runs_para(tf, [("· ", 12, GREEN, True, FB), (r, 12, INK, False, F)], after=10, line=1.12)
tf = box(s, 0.9, 6.6, 11.5, 0.4)
para(tf, "Full list: 73 references with DOIs / arXiv IDs in the report (literature_review.md).", 11.5, MUTED, False, F)

# ---------------------------------------------------------------- 15 · THANK YOU
s = slide_new()
rect(s, 6.17, 2.35, 1.0, 0.12, GREEN)
tf = box(s, 1.5, 2.85, 10.33, 1.2)
para(tf, "Thank you", 46, INK, True, FB, PP_ALIGN.CENTER, after=0)
tf = box(s, 1.5, 4.15, 10.33, 0.5)
para(tf, "Review 1  ·  Model-Driven Deep Learning for OTFS Modulation", 15, MUTED, False, F, PP_ALIGN.CENTER, after=0)
tf = box(s, 1.5, 4.75, 10.33, 0.5)
para(tf, "Questions welcome", 13, DEEP, True, F, PP_ALIGN.CENTER, after=0)

prs.core_properties.title = "Review 1 — Model-Driven Deep Learning for OTFS Modulation"
prs.core_properties.author = "Jaswant"
OUT = r"D:\College\KLH\2nd Year\EVEN SEM\MCS\PROJECT\Review1_LitReview_Solution.pptx"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides._sldIdLst))
