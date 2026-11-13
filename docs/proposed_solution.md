# Proposed Solution v2.3 — UOR-Net (Unified Unfolded OTFS Receiver)

**Status:** UNDER REVISION after audit round 4 (12 Nov 2026). FINAL is declared only when §8 open items are closed. Supersedes v2.2.

**Working title:** *"UOR-Net: Physics-Consistent Deep Unfolding for Fractional-Doppler-Robust OTFS Channel Estimation and Detection."* ("Joint" stays out of the title — joint CE+DD alone is not a novelty axis [27], [71]; joint *training* is the mechanism under test in §1/§6.)

### Changelog vs v2.2 (audit round 4)
1. **Research question restated as the empirical question it is** (§1): paired separate-vs-joint comparison; no novelty claim until the systematic full-text sweep (G2) closes.
2. **Top-K + STE removed from the training graph:** the detector's channel is now the **weighted-atom operator** `Ĥ = Σ_j ĥ_j H_j` (batched atom application; never materialized). Hard peak-picking / parabolic interpolation = **evaluation-only** (§3.3). Reason: top-K selection is discontinuous at rank changes and blocks gradients to unselected atoms.
3. **CE supervision replaced:** probe-vector **operator-space loss** `L_CE` (§4) supersedes snapped-support coefficient labels.
4. **Pilot model made explicit** (§5.1): `y_p = S_p H(h)x_p + S_p H(h)S_d x_d + w_p`; full-guard isolation verified first; detection input `y′ = y − Ĥx_p`; BCE/BER on **data positions only**.
5. **OAMP filter concretized** (§3.5): `G = ĤS_d`; `W̃_t = v_t²Gᴴ(v_t²GGᴴ+C_w)⁻¹`, `W_t = D·W̃_t/tr(W̃_t G)`; dense reference first (16×4/32×8); structured iterative solve only if needed (tolerance + iteration cap + trace method). Stated explicitly: **forward/adjoint scaling ≠ solve scaling** — unitary transform normalization does not remove the detector's linear solves.
6. **ISTA numerics pinned** (§3.3, §4): exact spectral norm at small grids or certified bound `‖A‖²_F` (finite power iteration can underestimate); `S_λ(0) = 0` explicit; interior initialization `γ_init = 0.5/L`.
7. **Soft-bit computation fixed** (§3.1): logsumexp LLRs; **bit-1 BCE logits = −LLR**; variance **per complex symbol**; positive floor; one mean over data bits; confident-symbol unit check.
8. **Doppler parameterized exactly as Raviteja Eq. (6):** `ν·N·T = k + κ`, κ ∈ (−½, ½]; signed offsets incl. 0; a held-out offset band; 20 dB labeled **SNR extrapolation** (§5.3).
9. **Stage plan rebuilt around ONE decisive grid (32×8)** until reproducible: S4 = unfolded ISTA vs classical at same depth; S5 = dense reference detector + measured solver cost; S7 = paired separate-vs-joint. Extra grids, 16-QAM and paper reproductions **deferred** (S8).
10. **Simulator split:** Stage-1 demo = frame-level-CP **LTI smoke test only** (Raviteja's ICID itself assumes no CP); Stage-2+ = **per-slot-CP LTV** time-domain model with the Doppler phase advanced over all samples incl. CP (§3.2).
11. **Yue et al. reframed** (§5.2): the dense reference detector is our own implementation; a Yue reproduction is deferred with recorded acceptance criteria; exact-parity claims reserved for identical updates/initialization/depth/variance rules.
12. **Pretrained-model routes documented** (§5.6): permitted but not required (faculty, 12 Nov 2026) → training from scratch is the default; external-checkpoint route fully specified; LSTM repo weights = CE-transfer experiment only.
13. **Demo upgraded to unitary normalization** (`src/otfs_demo.py`): mean sample energy 1.0, net DD gain 1; T1 < 1e-12, T2 = 2.22e-15 (5 seeds). Bibliography reconciled 73 ↔ 73 (`references.bib`).

---

## 1. Research framing

**Research question (verbatim; tested by Stage 7):**
> Does allowing detection loss to update the unfolded channel estimator improve data-bit BER relative to separate training of the same architecture, under matched fractional-Doppler channels and pilot-estimated CSI?

**Proposed contributions** (each an open hypothesis until §8 closes):
1. An unfolded sparse channel estimator (ISTA lineage) supervised in **operator space** on a fixed fractional-Doppler dictionary.
2. An unfolded OAMP detector operating on the **estimated** channel operator, with a concretized LMMSE filter and measured solver cost.
3. The **paired decisive experiment** — detection-loss gradients into CE vs frozen CE, same blocks/initialization/data budget — plus the reported mechanism analysis.
4. A reproducible benchmark: contract-tested simulator, BER/NMSE/latency/memory/FLOP-convention reporting, negative results included.

**Claim discipline:** initialization to classical values gives performance **parity at init** only if every operation matches exactly — asserted nowhere without the unit test. Expected gains over *well-tuned* classical baselines are **modest** (Blocks A + C carry O(10²) scalars); small or negative results are reported as-is. No "first" claims; no "state of the art" language without a direct supporting comparison (literature review §5 cell policy).

## 2. Scope

| In scope (decisive experiment) | Out of scope (deferred, S8) |
|---|---|
| QPSK · SISO · rectangular pulses · **ONE grid: 32×8** (16×4 debug) | 64×8 / 128×16 grids (only with a solver-cost case for why) |
| Fractional Doppler `νNT = k + κ`, κ ∈ (−½, ½], signed offsets | 16-QAM |
| Embedded impulse pilot + **full guards** first [2b] | Reduced-guard *training* (leakage-as-noise / reconstructed-subtraction: measured in S3, not gated) |
| Pilot-estimated CSI as the operating condition; controlled CSI-error sweep as a separate axis | MIMO, multi-user, phase noise / IQI / sync |
| Integer delays (quantization declared) | Fractional **delay** training (inverse-crime probe only, §5.5) |
| Per-slot-CP LTV time-domain simulator | Paper reproductions (Yue/PnP/Gao/Duan) until S7 is reproducible |

## 3. Architecture v2.3

```
received CP-OTFS frame
        │
┌───────▼─────────────────────────────────────────────────────┐
│ BLOCK A — Unfolded sparse CE (plain ISTA; FISTA deferred)    │ K layers
│  y_p = S_p H(h) x_p + S_p H(h) S_d x_d + w_p                 │
│   ≈ A h + B x_d + w   (A: pilot+guard measurement matrix;    │
│    B x_d: data leakage — in the simulator, measured in S3;   │
│    full guard ⇒ ≈0 in the residual; reduced guard ⇒ §5.1)    │
│  h^(k+1) = S_{λ_k}( h^(k) + γ_k Aᴴ(y_p − A h^(k)) )          │
│  γ_k = σ(s_k)·(1/L);  s_k init 0 ⇒ γ_init = 0.5/L (interior) │
│  L = exact ‖AᴴA‖₂ at 32×8 (or certified bound ‖A‖²_F;        │
│      finite power iteration may UNDERestimate)               │
│  λ_k = softplus(ρ_k) ≥ 0;  S_λ(0) = 0 (explicit)             │
│  S_λ(z) = max(1 − λ/|z|, 0)·z                                │
│  output: ĥ over ALL dictionary coefficients (no top-K)       │
└───────┬─────────────────────────────────────────────────────┘
│  weighted-atom operator (§3.3): Ĥ = Σ_j ĥ_j H_j              │
│  (batched: apply each atom H_j to the input, sum by ĥ_j)     │
┌───────▼─────────────────────────────────────────────────────┐
│ BLOCK B — Differentiable channel operator (no learning)      │
│  v = Ĥx, u = Ĥᴴy as structured matvecs (atom-batched)        │
│  y′ = y − Ĥ x_p   (pilot contribution removed)               │
└───────┬─────────────────────────────────────────────────────┘
┌───────▼─────────────────────────────────────────────────────┐
│ BLOCK C — Unfolded OAMP detector,  G = Ĥ S_d      │ T layers
│  r_t = x_t + W_t (y′ − G x_t)                                │
│  W̃_t = v_t² Gᴴ (v_t² G Gᴴ + C_w)⁻¹;  W_t = D·W̃_t/tr(W̃_t G)   │
│  D = #unknown data symbols;  C_w declared per §5.1           │
│  dense factorization first (D ≈ 240 @ 32×8 — trivial);       │
│  structured iterative solve only if needed: stated tolerance │
│  + iteration cap + trace method (§3.5)                       │
│  x_{t+1} = η_t(r_t, σ_t²)   η_t = QPSK posterior mean;       │
│  σ_t² per complex symbol, learnable, positive floor          │
└───────┬─────────────────────────────────────────────────────┘
        │ logsumexp LLRs (§3.1) → BCE on DATA BITS ONLY
        ▼
  L = α·L_CE + β·L_det   (normalized; gradient norms logged)
```

### 3.1 Soft-bit computation (round-4 corrected)
- Symbol posteriors (equal priors), computed with **logsumexp**: `p_a(z) ∝ exp(−|z−a|²/σ_t²)`.
- Bit LLRs over the declared QPSK bit labeling: `LLR_i = log( Σ_{a∈C_0(i)} p_a / Σ_{a∈C_1(i)} p_a )` — evaluated as a **difference of logsumexps** (never plain sums of small exponentials).
- **Loss:** BCE targeting bit 1 with **logits = −LLR** (PyTorch `BCEWithLogits` convention), **one mean reduction over data bits only**.
- **Variance:** σ_t² is per **complex symbol** (matches η_t's posterior); positive floor 1e-9.
- **Unit check (S6):** a confident symbol produces the expected hard bits and loss < 1e-6; the logsumexp path matches direct computation < 1e-10 at moderate σ.

### 3.2 Block B channel-model contract (enforced by unit tests)
1. **Grid and indexing (primary-source verified 12 Nov 2026):** M×N grid, **delay axis l (M points), Doppler axis k (N points)** — per the footnote of Raviteja et al. [2]: *"the first and second indexes, k and l, represent the Doppler and delay axis, respectively."* ISFFT = Eq. (2) of [2] (demo T1 verifies the algebraic identity).
2. **Fractional Doppler = Raviteja Eq. (6):** `ν = (k+κ)/(NT)`, κ ∈ (−½, ½]. All sweeps/splits in §5.3 are expressed in (k, κ).
3. **CP:** Stage-1 demo = ONE frame-level CP — a declared LTI smoke-test choice (ICID [2] itself assumes no CP). **Stage-2+ simulator = per-slot CP:** Ts = sample period, Tslot = (M+Ncp)Ts, Ncp ≥ max integer delay; the Doppler phase advances over ALL samples including CP; `r[n] = Σ_p g_p e^{j2πν_p nTs} s[n−l_p] + w[n]` (a delayed phase reference would require documenting the per-path phase — not used in v1).
4. **Unitary normalization (demo-verified):** unitary ISFFT/Heisenberg/Wigner/SFFT ⇒ mean sample energy 1.0 before the channel and net DD gain 1; unitary transforms preserve AWGN variance (unit test). Single-path norm scaling is checked separately from general multipath behavior.
5. Rectangular pulses v1.
6. Operator built from **estimated** atoms (§3.3); the true-parameter operator is kept for oracle curves.
7. **Stage-2 unit tests:** one-path waveform vs an independent dense reference at each test Doppler (± integer, off-grid near 0 and ½ bin) with **complex phase and leakage compared, not only magnitudes**; explicit shift-direction check; adjoint identity `⟨Hx, y⟩ = ⟨x, Hᴴy⟩` < 1e-10; structured-vs-dense operator < 1e-10 (float64) / < 1e-4 (float32), several seeds; measured input noise variance reproduces the declared σ²; delay near CP length; multipath. Recorded per run: library versions, grid order, CP, sample period, RNG seeds.

### 3.3 Weighted-atom operator (replaces v2.2 top-K + STE)
- **Training graph:** the detector's channel is `Ĥ = Σ_j ĥ_j H_j` over ALL dictionary atoms (M delay × η-oversampled Doppler), applied by batching: each atom `H_j` is applied to the input and the results are summed with weights `ĥ_j` — the dense sum is never materialized. Gradients reach every atom (no selection discontinuity); this is the exact graph the S4/S6 gradient tests validate.
- **Evaluation-only:** hard peak-picking + parabolic interpolation produce continuous (τ̂, ν̂, ĥ_p) for reports/plots. Interpolation is tested separately on isolated off-grid paths first, then on close/overlapping paths; reported Doppler error is absolute, in bins (and Hz via νNT·Δf), with and without noise.
- **Dictionary properties:** fixed over integer delays and candidate Dopplers; oversampling raises inter-atom coherence — resolution comparisons run at matched compute and include the **oracle dictionary approximation error** (best achievable Ĥ from the dictionary vs the true H).

### 3.4 OAMP assumptions — honest caveat (unchanged)
State evolution assumes (approximately) i.i.d. / right-orthogonally-invariant matrices; Ĥ is structured and worse under estimated CSI. Per-layer variance (and scalar) parameters are therefore **learnable** — the standard OAMP-Net2 remedy for SE mismatch. SE is treated as a modeling approximation for our structured channel; sensitivity is reported in S7, not assumed away (G12).

### 3.5 Detector complexity and solve scaling (round-4 concretized)
- **Solve scaling is its own step.** Unitary transform normalization fixes the *operator's* gain (§3.2) but does NOT implement the detector's linear solves: `W̃_t` needs `(v_t² GGᴴ + C_w)⁻¹` and the normalization needs `tr(W̃_t G)` — both computed, never assumed.
- **Dense reference first:** at 32×8, D ≈ 240 unknown data symbols → GGᴴ dense complex64 ≈ 0.5 MB; a direct Cholesky/eigendecomposition is trivial. The 16×4 debug grid is smaller still.
- **Structured iterative solve only if measurement demands it:** stated relative tolerance (1e-6 vs dense), iteration cap, and trace method (Hutchinson probes or exact Gram trace — declared). Filter outputs and BER are compared against the dense reference **before any grid scale-up**.
- **Cost accounting includes** setup, all solves, and trace estimation — in the latency, memory, and FLOP tables (§5.7). Peak memory is measured **before** moving to 128×16.

## 4. Training and losses

| Decision | Choice |
|---|---|
| CE supervision | **Probe-vector operator-space loss:** `L_CE = Σ_i ‖(Ĥ−H)u_i‖² / (Σ_i ‖Hu_i‖² + ε)` over K_probe = 8 fresh seeded unit-norm complex Gaussian probes per batch; ε = 1e-12. Supervises the operator's *action* — representation-free and off-grid-honest (replaces coefficient/snapped-support labels). The dictionary approximation error (§3.3) bounds what Ĥ can reach and is reported as the oracle curve. |
| Headline CE metric | **Effective-channel NMSE:** ‖Ĥ−H‖_F²/‖H‖_F² — the same object L_CE supervises. |
| Detection loss | BCE over data bits via logits −LLR (§3.1); one mean; data positions only. |
| Joint loss | `L = α·L_CE + β·L_det`, both O(1) at init, α = β = 1; per-loss gradient norms logged; re-normalize if one dominates > 10×. |
| Gradients into CE | Detection loss reaches CE parameters **through Ĥ = Σ ĥ_j H_j into G** — the S6 finite-difference test verifies nonzero, correct gradients from detection loss alone. |
| Optimizer / schedule | Adam, lr 1e-3, halved on 3-epoch validation plateau, early stop patience 5. |
| Batch / data budget | Identical for the paired runs (S7): 16 frames; 50k training frames (SNR ∈ [−5, 15] dB training; (k, κ) sampled per batch); 2k validation frames (seed 2). **20 dB = reported SNR extrapolation, not training.** |
| Classical-ISTA baseline | Same depth K, same A, fixed γ = 0.5/L, fixed λ at Block-A init values — **parity at init is a unit test** (outputs equal < 1e-6 relative). |
| ISTA numerics | L from the exact spectral norm at 32×8 (fallback: certified bound ‖A‖²_F); S_λ(0) = 0; λ_k = softplus(ρ_k) ≥ 0; γ_k = σ(s_k)/L with s_k init 0 (interior 0.5/L). |
| Numerics | float64 reference / float32 training, declared; noiseless checks < 1e-8 (f64) / < 1e-4 (f32); dtype recorded everywhere. |

## 5. Experimental design

### 5.1 Pilot model, guards, and CSI
- **Forward model (explicit):** `y_p = S_p H(h) x_p + S_p H(h) S_d x_d + w_p`, where S_d inserts unknown data into their resource elements and S_p selects pilot observations. Unknown-data leakage is channel-dependent — it is in the simulator, measured in S3, and never waved away.
- **Full-guard pilot isolation verified first** (residual leakage ≤ −25 dB of pilot observation power — measured, not assumed). **Reduced guards** are studied two ways: leakage-as-effective-noise (C_w inflated accordingly) or reconstructed-subtraction (detected data + current Ĥ). Both are reported side-by-side with the full-guard case; pilot-only vs data-only simulations quantify leakage under both guard sizes.
- **Detection input:** `y′ = y − Ĥx_p`, detection on `G = ĤS_d` (known pilot/guard symbols clamped); the linear filter is normalized with **D = dimension of the unknown data vector**.
- **C_w declared:** v1 default = estimated effective scalar from pilot-only residuals (the same information goes to every practical baseline); oracle-AWGN and modeled-covariance variants are reported separately. With estimated CSI the residual contains **channel mismatch × transmitted symbols in addition to AWGN** — the effective-noise reading is approximate; empirical residual variance, positivity floors, and BER under mismatch are tested (S7).
- **Metrics on data positions only** (BCE and BER).

### 5.2 Baselines (same frames wherever possible)
- **Primary — implemented here:** classical embedded-pilot CE + LMMSE; classical CE + the same detector (if affordable); **true-CSI LMMSE reference curve** (specific to that detector); classical ISTA (same depth) for Block A; classical OAMP for Block C. All share waveform, receiver information, frame/pilot budget, channel draws, noise, and test bits; if a published method requires different pilots/CP, its setting is reported separately.
- **Secondary — reported numbers:** Yue [36], Zhang PnP [27], Gao [71], Duan [39]/[40], DD-DNN [15], GNN [19], 2D-RC [21].
- **Yue et al. reframed:** our dense reference detector is our own implementation; a Yue **reproduction is deferred** (S8), with acceptance criteria recorded from the paper first (waveform, CSI-perturbation model, grid, layers, training config, BER acceptance). Adapting their update to our chain is labeled an adaptation; exact-parity claims only for identical updates, initialization, depth, and variance rules.

### 5.3 Doppler parameterization, sweeps, and splits
- All Dopplers are written `ν·N·T = k + κ`, κ ∈ (−½, ½] (Raviteja Eq. (6)). Sweeps sample **signed offsets including 0** (e.g., κ ∈ {−0.45, −0.2, 0, +0.2, +0.45} at a few integer k). An offset band (e.g., κ ∈ (0.3, 0.45)) is **held out** from training and hyperparameter selection.
- **Channel profile declared:** P paths (default 4), integer delays uniform in {0,…,15}, complex Gaussian gains with exponential power decay (15 dB dynamic range), Doppler bins k ∈ {−1, 0, 1} + signed κ; the physical anchor (carrier, speed, sample rate → max |νNT|) is recorded. Integer-delay quantization and any merged paths are declared in the results.
- **CSI axes separated:** the operating condition is pilot-estimated CSI; controlled perturbations of true/estimated CSI form a separate sweep with a declared perturbation domain; achieved effective-channel NMSE is reported **alongside** BER at every point.
- **Splits by channel realization:** test draws are excluded from early stopping and hyperparameter selection.

### 5.4 Statistical protocol (retained; no-error rule added)
- Eval seeds 1–20 (independent channels/frames); train seed 1, val seed 2.
- Per BER point: ≥ 10⁶ bits AND ≥ 100 error events (≥ 10⁷ bits when BER < 10⁻⁴); 95% Clopper–Pearson CIs. **If zero errors occur: report the bit count and a justified upper bound** (exact binomial / rule of three).
- All hyperparameters, learning curves, seeds, iteration budgets, and stopping rules are logged in advance (CSV/TensorBoard).

### 5.5 Inverse-crime control (retained)
Final models **and all classical baselines** are re-evaluated on channels from an independent generator: 8× oversampled continuous-time waveform with true fractional delay and an independent RNG stream. Fractional-delay training remains out of scope; results are reported side-by-side in S7.

### 5.6 Pretrained-model routes (faculty, 12 Nov 2026: permitted, NOT required)
- **Default: training from scratch** — the decisive experiment needs no external weights.
- **External-checkpoint route (optional, documented):** verify public download + license → pin version/hash → check framework and input dimensions → run inference in the source configuration → fine-tune on a distinct target distribution → compare **frozen / fine-tuned / randomly initialized** on the same held-out target set.
- **LSTM repo weights** (`anafreis/OTFS_CE`, existence verified): a **CE-transfer experiment only** — they do not initialize unfolded ISTA or OAMP. No compatible checkpoint verified for Yue et al. or UOR-Net (searched, audit round 4).
- **Self-pretraining route (needs instructor sign-off):** pretrain a module on a source simulation → save the checkpoint → fine-tune on independently generated off-grid channels; source/target configs recorded; evaluated on unseen target draws.

### 5.7 Full-receiver reporting (every headline table)
Uncoded BER on data positions + effective-channel NMSE + error/bit counts + CIs; pilot/guard overhead (spectral efficiency); **per-frame latency** (batch size, warmup, hardware, precision declared); **peak memory**; FLOP conventions documented (per linear op); GPU timings synchronized and **include channel construction, filter solves, and trace estimates**; where runtimes differ, BER is compared at matched latency explicitly.

## 6. Staged build plan (decisive experiment at ONE grid: 32×8)

| Stage | Deliverable | Gate (measured, not adjectives) | Status |
|---|---|---|---|
| 1 | LTI smoke test (frame-level CP, delay-only, ZF) — **proves transform algebra only** | T1 < 1e-12; T2 < 1e-8 (f64) over 5 seeds; unitary energy 1.0 | **DONE 12 Nov** (T2 = 2.22e-15) |
| 2 | Per-slot-CP LTV simulator + operator contract | All §3.2 item-7 tests pass (incl. shift direction, noise-variance reproduction, complex phase/leakage comparison) | pending |
| 3 | Pilot + full guards + leakage + classical baselines | Full-guard leakage ≤ −25 dB (measured); data-selector unit test; classical CE+LMMSE and true-CSI LMMSE BER curves on identical frames (≥ 10⁶ bits + CIs); pilot-only vs data-only leakage, both guard sizes | pending |
| 4 | **Unfolded ISTA vs classical ISTA (same depth)** | Parity at init < 1e-6; effective-NMSE ≤ classical CE at equal overhead (parity = pass); detection-loss-only gradient into CE nonzero + finite-difference check | pending |
| 5 | **Dense reference detector + trainable variant + solver cost** | tr(W_tG)/D = 1 ± 1e-6; init parity vs classical OAMP < 1e-6; finite soft bits; finite-difference gradients < 1e-5; noisy perfect-CSI BER ≤ LMMSE (or within 0.2 dB) **before** estimated-CSI runs; solver cost + peak memory measured | pending |
| 6 | Joint-training wiring + unit acceptance | Losses finite, O(1) at init; LLR/loss unit checks (§3.1); paired-run configs locked (same init/data/validation budget) | pending |
| 7 | **Paired separate-vs-joint decisive experiment** | ≥ 5 independent training seeds; BER/NMSE/timing with counts + CIs; held-out offset band + CSI-error sweep; inverse-crime side-by-side; full-receiver table (§5.7) | pending |
| 8 | Deferred menu (not gates) | 64×8/128×16 (post solver-cost case), 16-QAM, Yue reproduction (recorded criteria), η-oversampling at matched compute, FISTA, generator-variant hold-out promoted | — |

**Memory rules:** Block A matvec-only (Gram ≤ 2048² complex64); Block C dense factorization allowed at 16×4/32×8 (§3.5); structured solves only with the §3.5 contract; 128×16 is deferred behind a measured solver-cost + peak-memory case.

### Calendar (10 weeks, from Mon 17 Nov 2026)
| Week | Milestones |
|---|---|
| W1 | Stage 2 (LTV simulator + contract tests) + topics 1–4 |
| W2 | Stage 3 (pilots/leakage/classical curves) + topic 5 |
| W3 | Stage 4 (unfolded ISTA + parity + gradient checks) + topics 6–7 |
| W4 | Stage 5 (dense reference detector + solver cost) + topic 8 (QAM/MMSE/OAMP) + **buffer** |
| W5 | Stage 6 (joint-training wiring + unit acceptance) |
| W6–W7 | Stage 7 paired experiment (≥ 5 seeds) |
| W8 | Hold-out band + CSI sweep + inverse-crime + full-receiver tables |
| W9 | Analysis, plots, negative-results write-up |
| W10 | Report, viva prep (buffer) |

## 7. Priority reading order (unchanged)
1. Yue et al. 2024 (DOI 10.1109/LCOMM.2024.3362970) — detector foundation; adaptation criteria recorded before any reproduction.
2. Zhang et al. 2025 PnP (arXiv:2503.11102) — strongest unified competitor.
3. Gao et al. 2025 (DOI 10.1109/JIOT.2025.3586012) — second joint CE+DD competitor; identify exactly what it does and does not unfold.
4. Duan et al. 2026 (TVT, DOI 10.1109/TVT.2026.3736276) — fractional-Doppler CE unfolding.
5. Duan et al. 2026 (WCL, DOI 10.1109/LWC.2025.3626987) — FISTA-based DD sparse CE.
6. Raviteja et al. 2018 (TWC) — discrete I/O model + Eq. (6) fractional Doppler (S1 demo implements it; the k/l footnote is verified against the full text).
7. He et al. 2020 (arXiv:1907.09439) — unfolding theory; Eqs. (24)–(27) are Block C's ancestor.

### Companion learning sequence (unchanged)
multipath channels → delay & Doppler → OTFS mod/demod (Stage 1 ✓) → fractional Doppler (Stage 2) → channel estimation & pilots (Stage 3) → sparse recovery/ISTA (Stage 4) → matrix channel model (Block B) → QAM/MMSE/OAMP (Stages 4–5) → deep unfolding & joint training (Stages 6–7).

## 8. Open verification items (FINAL declaration blocked on these)
- [ ] Systematic novelty sweep (full text): joint/unfolded training mechanisms vs [27], [29], [36], [71] — using the G2 comparison axes (which algorithm is unfolded; which losses reach which module; whether detection feeds back at inference).
- [ ] IEEE Xplore verification of the 2026 early-access DOIs (e.g., LWC.2025.3626987) — cite the volume year as given, or correct.
- [ ] Complete author lists for the remaining author-less bib entries (73 entries now reconciled 1:1 with the Markdown list; `hadani2017mmwave` currently "and others").
- [ ] Exact network names/claims of Duan et al. TVT vs WCL from the IEEE pages.
- [ ] Yue et al. public-code availability (S8; acceptance criteria recorded first).
- [ ] S6 unit-acceptance evidence pack (LLR checks, trace residual, gradient tests) attached to the report.
