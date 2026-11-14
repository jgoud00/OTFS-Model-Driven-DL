# Project Report — Model-Driven Deep Learning for OTFS Modulation

**Course:** MCS · KLH (2nd Year, Even Semester) · **Date:** 14 Nov 2026
**Repository:** `jgoud00/OTFS-Model-Driven-DL` (private) · 2 commits
**Companion documents:** `literature/literature_review.md` (full survey), `literature/references.bib` (73 entries), `docs/proposed_solution.md` v2.3 (full design), `docs/study_guide.md`, `docs/acronyms.md`

---

## Abstract

Orthogonal Time Frequency Space (OTFS) modulation targets high-mobility scenarios (high-speed rail, V2X, UAVs, LEO satellites) where Doppler spread destroys OFDM orthogonality. Deep-learning OTFS receivers split into data-driven networks (opaque, retrain-per-config) and model-driven *deep unfolding* (iterative algorithms become layers with few learnable physics-anchored scalars). Our survey of 73 references (4 audit rounds) finds that nearly all unfolding works unroll **one** module (estimator *or* detector) while the other is given, and that a detector-side unfolding validated under **fractional Doppler with estimated CSI** is not documented in the checked sources. This project builds **UOR-Net** — an unfolded ISTA channel estimator joined to an unfolded OAMP detector through a differentiable delay–Doppler channel operator — and runs a **paired separate-vs-joint training experiment** to answer, as an open testable hypothesis: *does allowing detection loss to update the estimator improve data-bit BER under matched fractional-Doppler channels?* Stage 1 (transform algebra) is complete and verified to 2.22×10⁻¹⁵ symbol error; the remaining stages proceed on a 10-week calendar.

---

## 1. Introduction

### 1.1 Problem

Wireless channels are **doubly dispersive**: multipath causes *delay* spread (inter-symbol interference) and motion causes *Doppler* spread (time-variation within a frame). OFDM fixes multipath with a cyclic prefix but loses subcarrier orthogonality under Doppler (inter-carrier interference). OTFS instead places data on a 2D **delay–Doppler (DD) grid** and transforms it into a time–frequency waveform; with ideal pulses each path stays a point on the grid and every symbol experiences approximately the whole channel rather than one fading snapshot. *(Qualified: under rectangular pulses extra inter-symbol/inter-carrier interference appears, and fractional Doppler smears one path across neighboring bins — the impact is configuration-dependent: pulse/CP/grid/receiver.)*

### 1.2 Why deep learning, and why model-driven

Data-driven networks (DNN/CNN/GNN detectors) reach good BER but need 10⁴–10⁶+ parameters, retrain per configuration, and offer little interpretability. **Deep unfolding** unrolls an iterative algorithm (ISTA, OAMP) into layers, keeps the physics fixed inside each layer, and makes only a few scalars learnable — the step sizes, thresholds, and variances that classical theory must guess. Famous precedents: LISTA (learned ISTA) and He et al.'s OAMP-Net2 for MIMO. The result: O(10²) parameters, interpretable, trains on tens of thousands of synthetic frames, and generalizes via physics.

### 1.3 Application anchor (high-speed travel)

Speed maps to our Doppler parameter via `f_D = v·f_c/c`, and our grid Doppler is `ν·N·T = k + κ` (Raviteja Eq. 6, κ ∈ (−½, ½]):

| Scenario | Speed | Carrier | f_D | Regime in our experiments |
|---|---|---|---|---|
| City car | 120 km/h | 2 GHz | ≈ 222 Hz | sub-bin — easy |
| High-speed train | 500 km/h | 3.5 GHz | ≈ 1.6 kHz | our fractional-Doppler regime (≈1–2 bins) |
| LEO satellite | 7.5 km/s | 2 GHz | ≈ 50 kHz | many bins — future work |

*(Exact bin counts depend on the final frame duration T; computed, not eyeballed, when the Stage-2/3 Doppler sweep lands. BER is reported as a function of Doppler spread, which maps directly to speed.)*

---

## 2. Literature Review (summary — full version in `literature/literature_review.md`)

### 2.1 Methodology and evidence standard

- **Sources:** ≈160 arXiv + ≈300 Crossref candidates screened (12 Nov 2026); web search unavailable (exhausted credits) — stated, not hidden.
- **Result:** **73 references** reconciled 1:1 with BibTeX; journal/conference entries DOI-checked via Crossref; status = *PARTIALLY VERIFIED* (core entries primary-checked, rest metadata-level).
- **Survived 4 independent audit rounds** (all findings accepted): conventions verified against the primary source (Raviteja et al., arXiv:1802.05242, read in full via ar5iv); competitor works verified via Crossref/DOI; false claims caught and corrected (e.g., "no public code exists" → two verified repos exist; "model-driven never handles impairment X" → falsified by [38]).
- **Table cell policy:** "—" / "unverified" / "verified" are distinct labels; absence of evidence is never stated as evidence of absence.

### 2.2 Thematic findings

1. **OTFS foundations (2017–):** Hadani et al. introduced OTFS (WCNC 2017); Raviteja et al. (TWC 2018) gave the discrete model, iterative detection, and the fractional-Doppler parameterization we adopt (all conventions verified from the paper itself: k = Doppler axis, l = delay axis, ν = (k+κ)/(NT)).
2. **Data-driven wave (2021–23):** DNN/GNN detectors, CNN estimators, 2D-RC — effective but opaque, retrain-per-config, parameter-heavy.
3. **Model-driven unfolding line (2024–26):** born with Yue et al.'s unfolded OTFS detector (robustness to channel-estimation error); by the explicit-unfolding counting rule ~10 works qualify (2024–2026) — a young niche.
4. **Joint CE+detection exists** in *hybrid* (Zhang PnP, TCOM 2025 — optimization skeleton + learned priors, sequential pipeline) and *data-driven* (Gao et al., IoT-J 2025) forms; **jointly trained explicit unfolding through a differentiable channel operator remains unverified in the checked sources**.
5. **Fractional Doppler:** handled on the *estimation* side (CE-side unfolding, Duan TVT 2026) and by at least one hybrid detector (AMP-GNN); an **explicitly unfolded detector trained/validated under fractional Doppler with estimated CSI** is not documented in the checked sources.
6. **Complexity reporting is uneven:** the hybrid line reports FLOPs and measured runtime; the unfolding line reports parameter counts at best.
7. **No shared benchmark** (same waveform + pilot pattern + protocol reported on by multiple papers) was identified in this audit — though public code exists: `textremo/Phy_Mod_OTFS` (toolbox, 68★) and `anafreis/OTFS_CE` (LSTM-CE reproduction, 57★), both verified on GitHub.

### 2.3 Gap analysis (G1–G14, condensed)

Load-bearing for this project: **G2** (what exactly is *jointly trained* — the paired comparison is open), **G3** (detector-side fractional Doppler for the unfolding line), **G4** (training-time vs evaluation-time robustness to CSI error — Yue et al. is the only unfolded detector claiming training-time robustness), **G5** (systematic generalization splits — SNR transfer / few-shot / channel-transfer are conflated in the field), **G8** (complexity transparency in the unfolding line), **G9** (no shared benchmark identified).
Not load-bearing (future work): **G6** (MIMO-OTFS detection unfolding), **G7** (unified ISAC receiver), **G10** (combined-impairment matrix), **G11** (no measured-channel/OTA validation identified), **G12** (no convergence guarantees for *learned* receivers — algorithm-level proofs exist), **G13** (pilot *value* learning verified; pilot *location* / guard-pattern learning unverified), **G14** (LLM-era PHY — demoted).

**Six headline gaps:** young fragmented field · joint training unverified · detector-side fractional Doppler open · robustness/generalization splits missing · no shared benchmark · no guarantees for learned receivers.

### 2.4 Conclusion of the review (empirical form)

The review supports **questions, not closure**. The decisive question this project tests: **does allowing detection loss to update the unfolded channel estimator improve data-bit BER relative to separate training of the same architecture, under matched fractional-Doppler channels and pilot-estimated CSI?** — reported with error/bit counts, 95% confidence intervals, and negative results included.

---

## 3. Proposed Solution — UOR-Net (v2.3)

**UOR-Net** (Unified Unfolded OTFS Receiver): two unrolled iterative algorithms joined by a differentiable physics operator.

```
received DD frame y (32×8) + pilot observation y_p
        │
BLOCK A — Unfolded ISTA channel estimator, K = 10 layers
   h ← S_λk( h + γk·Aᴴ(y_p − A h) )        (soft-threshold S_λ(0)=0)
   learnable per layer: γk (step, σ(s_k)/L), λk (threshold, softplus ≥ 0)
   pilot model: y_p = S_p H x_p + S_p H S_d x_d + w_p  (data leakage modeled, measured)
        │  sparse atom gains ĥ_j
        ▼
BLOCK B — Differentiable DD channel operator (NO learning)
   Ĥ = Σ_j ĥ_j H_j   (weighted atoms, batched mat-vecs — dense sum never materialized)
   y′ = y − Ĥ x_p    (pilot contribution removed)
   G = Ĥ S_d
        │  ← the ONLY gradient path from detection loss into the estimator
        ▼
BLOCK C — Unfolded OAMP detector, T = 10 layers
   linear filter  W̃_t = v_t² Gᴴ(v_t² GGᴴ + C_w)⁻¹,  W_t = D·W̃_t / tr(W̃_t G)
   denoiser η_t = QPSK posterior mean; per-layer learnable variance v_t², damping
   soft bits: logsumexp LLRs → BCE on DATA BITS ONLY (logits = −LLR)
        │
L = α·L_CE + β·L_det
   L_CE = operator-space probe loss: Σ_i ‖(Ĥ−H)u_i‖² / (Σ_i ‖Hu_i‖² + ε)  (8 fresh probes/batch)
```

**Key properties**

| Property | Why it matters |
|---|---|
| **~40 learnable parameters total** (20 in A + 20 in C) | Small data budget, interpretable, trains on a laptop |
| **Parity-at-init** | Every scalar starts at its classical value; a unit test asserts the net ≡ classical ISTA+OAMP (<1e-6) before training — a guaranteed-sane start |
| **Weighted-atom operator** (v2.3) | Replaces v2.2's top-K + straight-through estimator: gradients reach every atom with no selection discontinuity |
| **Operator-space CE supervision** | Off-grid (fractional-Doppler) truth has no unique coefficient label; supervising the operator's *action* on probe vectors is representation-free and honest |
| **Honest OAMP caveat** | State evolution assumes i.i.d./right-orthogonal structure that our channel violates approximately — hence learnable variances (the OAMP-Net2 remedy), sensitivity reported, not assumed away |
| **Transmitter is classical by design** | Learned Tx (pilot-value learning, pre-equalization) is future work; a learned transmitter would confound the receiver experiment |

**Scope (decisive experiment):** ONE grid — **32×8** (16×4 debug), **QPSK**, rectangular pulses, P = 4 paths, SNR −5…15 dB training (20 dB = extrapolation), 50k procedurally generated training frames, batch 16, Adam lr 1e-3, float32 training / float64 reference. No external dataset: data is generated on-the-fly by the contract-tested simulator with exact channel and bit ground truth; splits are by channel realization, test draws excluded from model selection.

---

## 4. Methodology

### 4.1 Staged build plan (gates are measured, not adjectives)

| Stage | Deliverable | Gate | Status |
|---|---|---|---|
| 1 | LTI smoke test (frame-level CP, delay-only) — proves transform algebra only | T1 < 1e-12; T2 < 1e-8 over 5 seeds; unitary energy 1.0 | **DONE 12 Nov** (T2 = 2.22e-15) |
| 2 | Per-slot-CP LTV time-domain simulator + operator contract | dense-reference comparison (complex phase + leakage), shift-direction check, adjoint identity < 1e-10, noise-variance reproduction | pending |
| 3 | Embedded pilot + full guards + leakage measurement + classical baselines | full-guard leakage ≤ −25 dB (measured); CE+LMMSE and true-CSI LMMSE curves on identical frames | pending |
| 4 | Unfolded ISTA vs classical ISTA (same depth) | parity at init < 1e-6; effective-NMSE ≤ classical; detection-loss-only gradient into CE verified (finite differences) | pending |
| 5 | Dense reference OAMP detector + trainable variant + solver cost | tr(W_tG)/D = 1 ± 1e-6; parity < 1e-6; noisy perfect-CSI BER ≤ LMMSE before estimated-CSI runs | pending |
| 6 | Joint-training wiring + unit acceptance | losses finite, O(1) at init; LLR unit checks; paired-run configs locked | pending |
| 7 | **Paired separate-vs-joint decisive experiment** | ≥5 training seeds; held-out Doppler-offset band + CSI-error sweep; inverse-crime side-by-side; full-receiver table | pending |
| 8 | Deferred menu | bigger grids (after solver-cost case), 16-QAM, Yue reproduction, FISTA, refinement head | — |

### 4.2 Statistical protocol

- ≥ **10⁶ bits AND ≥ 100 error events** per BER point (≥10⁷ bits when BER < 10⁻⁴); **95% Clopper–Pearson CIs**; zero-error points reported with an exact binomial upper bound.
- Eval seeds 1–20; train seed 1, validation seed 2; hyperparameters/stopping logged in advance.
- **Inverse-crime control:** final models *and* baselines re-evaluated on channels from an independent generator (8× oversampled continuous-time waveform, true fractional delay, independent RNG).
- **Full-receiver reporting:** data-position BER + effective-channel NMSE + error/bit counts + CIs + pilot overhead (spectral efficiency) + per-frame latency (hardware/precision declared, includes solves and trace terms) + peak memory + FLOP conventions.

### 4.3 Baselines

- **Primary (implemented here, same frames):** classical embedded-pilot CE + LMMSE; classical CE + same detector; true-CSI LMMSE reference; classical ISTA (same depth); classical OAMP.
- **Secondary (reported numbers):** Yue [36], Zhang PnP [27], Gao [71], Duan [39]/[40], DD-DNN [15], GNN [19], 2D-RC [21].
- **Pretrained models:** permitted, not required (faculty, 12 Nov 2026) → **train from scratch is the default**; the public LSTM-CE repo weights are CE-transfer only and do not initialize unfolded architectures.

---

## 5. Results So Far

### 5.1 Stage-1 verification (12 Nov 2026)

| Check | Result |
|---|---|
| T1 — transform identity (ISFFT/Heisenberg/Wigner/SFFT round trip) | **PASS** < 1×10⁻¹² |
| T2 — noiseless chain, 4-tap channel, 5 seeds | **PASS** — worst symbol error **2.22×10⁻¹⁵** (< 1e-8) |
| Unitary normalization | mean time-sample energy **1.0**, net DD gain 1 |
| Sanity BER (LTI, true CSI) | 0.0156 @ 10 dB, 0 @ 20 dB |
| Bibliography reconciliation | 73 MD refs = 73 BibTeX entries |

### 5.2 I/O representation demo (13 Nov 2026 — `src/io_demo.py`, 32×8 decisive grid)

| Quantity | Value |
|---|---|
| Input | **358 bits/frame → 179 QPSK data cells** + 1 pilot + guards → **264 time samples** (incl. CP=8) |
| Channel | 4 random taps, unit energy + AWGN at declared SNR |
| Output | estimated DD grid → LLRs → hard bits on 179 data cells |
| Spectral efficiency (schematic guards) | ≈ **1.36 bits/sample** |
| Data-position BER @10 / 20 dB (true-CSI LMMSE) | **0.0000 / 0.0000** |

**Figures (all visually verified, in `results/`):**
- `fig1_input.png` — QPSK constellation · DD resource map (pilot/guards/179 data cells) · transmit waveform with CP
- `fig2_channel.png` — true channel = 4 sparse taps · noiseless received DD grid (every symbol sees the whole channel)
- `fig3_output.png` — noisy received grid @10 dB · detector output constellation with error marking

### 5.3 Deliverables produced

| Artifact | Path |
|---|---|
| Literature review (73 refs, G1–G14 evidence tables, 4 audit rounds) | `literature/literature_review.md` |
| BibTeX (73 entries, DOI-checked where applicable) | `literature/references.bib` |
| Governing spec v2.3 (architecture, losses, protocol, calendar) | `docs/proposed_solution.md` |
| Verified Stage-1 simulator (runs tests at import) | `src/otfs_demo.py` |
| I/O representation demo + 3 figures | `src/io_demo.py`, `results/fig1–3.png` |
| Review-1 deck (15 slides, MongoDB-green minimalist) | `Review1_LitReview_Solution.pptx` + `src/make_review1_ppt.py` |
| Self-study guide (8 sessions + 24-question viva bank) | `docs/study_guide.md` |
| Glossary (70+ terms, symbols, venues) | `docs/acronyms.md` |

---

## 6. Tech Stack

| Layer | Tool | Status |
|---|---|---|
| Language | Python 3.12 (venv) | installed |
| Simulation / DSP | NumPy 2.5.4 + SciPy 1.18.1 (FFT, LMMSE, linalg) | installed |
| Plots | Matplotlib 3.11.2 → `results/*.png` | installed |
| Neural training (Stage 4+) | PyTorch (autodiff, Adam, float32) | deferred to S4 |
| Demo dashboard | Streamlit (reads `results/*.csv` only) | optional, near review |
| Slides | python-pptx | installed |
| Version control | Git + GitHub (`jgoud00/OTFS-Model-Driven-DL`, private) | live |

Deliberately absent: frontend frameworks, databases, Rust/C++, transformer libraries. **Why PyTorch over TensorFlow:** the model-driven receiver literature we build on is PyTorch-based and all our layers are custom (learnable scalars, complex tensors, gradients through Ĥ) — framework conveniences don't apply; either could implement the design. **Why not CNN/ResNet:** images have no physics model to exploit; DD channels have an exact sparse structure — unfolding *starts from* the algorithm instead of *searching* for structure.

---

## 7. Timeline

- **Original:** 10-week calendar from 17 Nov 2026 (W1 Stage 2 → W6–W7 paired experiment → W9 analysis).
- **Trimester trim (competing projects):** irreducible core = S2 (1.5 wk) · S3 baselines (1 wk) · S4 unfolded ISTA (1.5 wk) · S5 OAMP (1.5 wk) · S6+S7 merged joint experiment, 3 seeds (2 wk) · report written as-you-go (1 wk buffer).
- **Pre-approved cut list:** inverse-crime probe, FISTA, refinement head, 5→3 seeds, deep latency study, secondary baselines → reported numbers only.
- **Survival rules:** every stage ends with a commit + one report paragraph; hard time-boxes (cut, never extend); one fixed OTFS block per week; Review-1 deck frozen.

---

## 8. Discussion — Claims Discipline

1. **No "first"/novelty claims.** The neighborhood is crowded (LISTA, OAMP-Net2, PnP, Gao, Lei, Duan). The claim is an **open, testable question** plus a clean mechanism study; a formal novelty sweep is a declared open item.
2. **One evidence standard.** Every gap table distinguishes verified / unverified / not reported; absence of evidence is never evidence of absence.
3. **Negative results are results.** If joint training shows no gain, that is the deliverable answer.
4. **Grounded conventions.** All waveform conventions verified against the primary source (Raviteja et al. full text), not secondary summaries.
5. **Known limitations (declared):** synthetic data only (no measured channels — G11); state evolution approximate for our structured channel (G12 — mitigated by learnable variances, sensitivity reported); rectangular pulses (extra interference, configuration-dependent impact); counting-precision caveat on the ~10-work corpus estimate (G1).

---

## 9. Conclusion

The project has completed the expensive discovery phase: a 4-round-audit-hardened literature review (73 references, evidence-labeled gap analysis), a concrete governing spec (UOR-Net v2.3), a verified Stage-1 simulator (2.22×10⁻¹⁵ noiseless error, unitary transforms), and I/O representation figures demonstrating the full input→output path. The decisive scientific question — **does joint training of an unfolded estimator and detector through a differentiable DD channel operator improve BER under fractional Doppler and estimated CSI?** — is stated as an open hypothesis with a controlled paired experiment, defined statistical protocol, and pre-registered cut list. Remaining work proceeds stage-by-stage with measured gates: LTV simulator (S2) → baselines (S3) → unfolded estimator (S4) → unfolded detector (S5) → joint wiring (S6) → paired experiment (S7), on a trimester-compatible calendar with the report written as results land.

**Future work (documented, deferred):** MIMO-OTFS unfolding, learned transmitter (pilot value/location, pre-equalization), learned-denoiser / zero-init refinement-head ablations, measured-channel or OTA probe, bigger grids + 16-QAM, unified ISAC receiver.

---

## References

73 numbered references with DOIs/arXiv IDs are maintained in `literature/literature_review.md` §References and `literature/references.bib` (reconciled 1:1). Anchor works: [1] Hadani et al., WCNC 2017 (OTFS) · [2] Raviteja et al., TWC 2018 (discrete model, arXiv:1802.05242, primary-source verified) · [8] He et al. (OAMP-Net2, MIMO) · [36] Yue et al. 2024 (unfolded OTFS detector) · [27] Zhang et al., TCOM 2025 (hybrid PnP) · [71] Gao et al., IoT-J 2025 (data-driven joint CE+DD) · [39] Duan et al., TVT 2026 (fractional-Doppler CE unfolding).
