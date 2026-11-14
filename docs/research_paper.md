# UOR-Net: A Unified Unfolded OTFS Receiver — Design, Verification, and a Pre-Registered Study of Joint Estimator–Detector Training under Fractional Doppler

**Authors:** [Student Name], [Roll No.] — Guide: [Guide Name]
**Course:** MCS, KLH · 2nd Year, Even Semester · **Version:** 1.0 (14 Nov 2026)
**Status:** Design + verification complete; decisive experiment pending (Stage 7). This paper follows a pre-registration format: experimental design is fixed before results exist, and negative results will be reported.

---

## Abstract

Orthogonal Time Frequency Space (OTFS) modulation addresses doubly dispersive channels — simultaneous delay and Doppler spread — that arise in high-mobility scenarios such as high-speed rail, vehicle-to-everything links, unmanned aerial vehicles, and low-Earth-orbit satellites, where Doppler destroys the subcarrier orthogonality that OFDM relies on. Recent deep-learning OTFS receivers divide into data-driven networks (effective but parameter-heavy, opaque, retrained per configuration) and model-driven *deep unfolding*, in which an iterative algorithm is unrolled into layers with only a few physics-anchored scalars made learnable. Our survey of 73 references, hardened across four audit rounds, indicates that (i) the explicit-unfolding line for OTFS is young (≈10 works, 2024–2026, under a stated counting rule), (ii) almost all unfolding works unroll exactly one module — estimator *or* detector — while the other is given, and (iii) an explicitly unfolded detector trained and validated under fractional Doppler with pilot-estimated CSI is not documented in the sources we checked. We present the design of **UOR-Net**, a unified unfolded receiver that connects an unfolded ISTA channel estimator to an unfolded OAMP detector through a differentiable delay–Doppler channel operator, with ≈40 learnable parameters and a parity-at-init guarantee. We specify a pre-registered paired separate-vs-joint training experiment with a fixed statistical protocol, and report completed verification results: the transform algebra is validated to a worst-case symbol error of 2.22×10⁻¹⁵ (noiseless, five seeds) under unitary normalization, and the full input–output path is demonstrated on the 32×8 decisive grid. The decisive comparison will be reported either way, negative results included.

**Keywords** — OTFS, delay–Doppler, deep unfolding, model-driven deep learning, channel estimation, OAMP detection, fractional Doppler, joint training

---

## I. Introduction

### A. Motivation

A wireless channel that is both delay-dispersive (multipath) and time-varying (motion) is **doubly dispersive**. OFDM copes with multipath via the cyclic prefix but loses subcarrier orthogonality under Doppler, producing inter-carrier interference that grows with mobility [3]. OTFS instead places data on a two-dimensional **delay–Doppler (DD) grid** and modulates it into a time–frequency waveform; with ideal pulses, each physical path maps to a point on the DD grid and every data symbol experiences approximately the full channel rather than a fading snapshot [1], [3]. This benefit is qualified: under rectangular pulses extra inter-symbol/inter-carrier interference appears, and **fractional Doppler** — path Doppler offsets κ between grid points — smears each path across neighboring bins, reducing effective sparsity [2], [46], [47]. The impact is configuration-dependent (pulse shape, CP, grid size, receiver), which our experiments declare explicitly.

### B. Deep learning for the physical layer

Data-driven receivers (DNN, CNN, GNN, 2D residual networks [14], [15], [19], [21]) learn the receive map from data. They achieve competitive BER but require 10⁴–10⁶+ parameters, retraining per configuration, and offer limited interpretability. **Model-driven deep unfolding** [7], [8], [9] takes the opposite route: unroll an iterative algorithm with known structure (ISTA for sparse recovery; OAMP for detection) into layers, keep the physics fixed inside each layer, and make only the scalars that classical theory must guess — step sizes, thresholds, variances — learnable. This yields O(10²) parameters, trainability on tens of thousands of synthetic frames, and interpretability.

### C. The open question

The literature establishes unfolded OTFS *estimators* (e.g., [39], [40]), an unfolded OTFS *detector* (Yue et al. [36], robust to channel-estimation error), and joint estimation–detection in *hybrid* (optimization-skeleton, [27]) and *data-driven* ([71]) forms. What is not documented in our checked sources is an **explicitly unfolded detector operated under fractional Doppler with estimated CSI**, and no work reports the controlled comparison that our question demands. We therefore pose an open, testable hypothesis:

> **RQ:** Does allowing the detection loss to update the unfolded channel estimator — through a differentiable DD channel operator — improve data-bit BER relative to separate training of the *same* architecture, under matched fractional-Doppler channels and pilot-estimated CSI?

We make no novelty claim; a formal novelty sweep is a declared open item. The contribution is the design, the controlled experimental protocol, and the mechanism answer — whichever direction it points.

### D. Contributions

1. **Architecture:** UOR-Net — unfolded ISTA estimator + differentiable weighted-atom DD operator + unfolded OAMP detector (Sec. IV), designed for parity at initialization.
2. **Honest supervision:** operator-space (probe-vector) channel loss rather than coefficient labels, which is representation-free and honest under fractional Doppler (Sec. IV-D).
3. **Pre-registered evaluation:** paired separate-vs-joint experiment, ≥5 training seeds, held-out Doppler-offset band, controlled CSI-error sweep, inverse-crime control, and a fixed statistical protocol (Sec. V).
4. **Verification results:** contract-tested Stage-1 simulator (unitary transforms, 2.22×10⁻¹⁵ worst-case noiseless error) and a full input–output demonstration on the decisive grid (Sec. VI).

### E. Paper organization

Sec. II summarizes related work; Sec. III fixes the system model and conventions; Sec. IV details UOR-Net; Sec. V pre-registers the experimental design; Sec. VI reports verification results; Sec. VII defines the decisive experiment as a results template; Sec. VIII discusses limitations; Sec. IX concludes.

---

## II. Related Work

### A. OTFS foundations

Hadani et al. introduced OTFS [1]; Raviteja et al. gave the discrete DD model, an iterative interference-cancellation detector, and the fractional-Doppler parameterization adopted here [2], plus the embedded-pilot estimation scheme [2b]. Message-passing receivers exploit DD sparsity [12], [13]. Classical studies establish that fractional Doppler degrades on-grid detection in the studied regimes [46], [47] — an effect our receiver must confront by design.

### B. Data-driven OTFS receivers (2021–2023)

The data-driven wave produced DD-domain DNN detectors [15], GNN detectors [19], 2D-RC networks [21], environment-robust detectors with one-sample adaptation [17], and AMP-enhanced GNNs covering fractional Doppler with reported FLOPs and runtime [20]. These works motivate model-driven alternatives on interpretability and parameter-budget grounds.

### C. Model-driven unfolding for OTFS (2024–2026)

Yue et al. [36] introduced a model-driven (unfolded) OTFS detector trained against channel-estimation error — the anchor for our detector block. The estimator side saw unfolding under fractional Doppler [39], [40]. Hybrid optimization-plus-learned-prior receivers unify estimation and detection sequentially [27], and lightweight data-driven joint estimation–detection exists [71]; modular end-to-end learning over superimposed pilots optimizes the pilot value at a fixed location [29]. By our stated counting rule (explicit unrolling of an iterative algorithm), ≈10 OTFS works qualify as model-driven unfolding — a young, fragmented niche.

### D. Gap summary

Six headline gaps survive four audit rounds (evidence tables in the companion survey):

| # | Gap | Load-bearing here |
|---|---|---|
| G2 | Jointly trained *unfolding* (two unrolled algorithms co-optimized through a differentiable channel model) unverified | **Yes** |
| G3 | Fractional Doppler on the *detector* side for the unfolding line | **Yes** |
| G4 | Training-time vs evaluation-time robustness to CSI error conflated | **Yes** |
| G5 | Systematic generalization splits (SNR transfer / few-shot / channel transfer) unclaimed | **Yes** |
| G8 | Complexity reporting uneven across the unfolding line | **Yes** |
| G9 | No shared benchmark identified (public code/toolboxes exist) | **Yes** |

G6 (MIMO detection unfolding), G7 (unified ISAC receiver), G10 (combined impairments), G11 (no measured-channel/OTA validation identified), G12 (no guarantees for *learned* receivers), G13 (pilot-location / guard-pattern learning unverified), G14 (LLM-era PHY) are recorded as future work and do not motivate this SISO study.

---

## III. System Model

### A. Grid, transforms, and normalization

Data occupy an M×N DD grid; per the primary source [2] (verified in full text), the first grid index **k is Doppler (N points)** and the second **l is delay (M points)**. The transform chain (rectangular pulses):

```
X[l,k] --ISFFT--> S[m,n] --Heisenberg (unitary IFFT per slot)--> s(t) --CP--> channel
r(t) --remove CP-- Wigner (unitary FFT per slot) --SFFT--> Y[l,k]
```

ISFFT: `S[m,n] = (1/√(MN)) Σ_{l,k} X[l,k] exp(+j2π(kn/N − lm/M))`, algebraically equal to `√(N/M)·fft_delay(ifft_doppler(X))` under NumPy conventions. All transforms use **unitary normalization**: mean time-sample energy 1.0 before the channel and net DD gain 1 (unit-tested); unitary transforms preserve AWGN variance.

### B. Fractional Doppler and channel

Path Doppler is `ν = (k+κ)/(NT)` with **signed offset κ ∈ (−½, ½]** (Raviteja Eq. (6) [2]). The declared channel profile: P = 4 paths; integer delays uniform in {0,…,15}; complex Gaussian gains with exponential power decay (15 dB dynamic range); Doppler bins k ∈ {−1, 0, 1} with signed κ. The physical anchor (carrier, speed, sample rate → max |νNT|) is recorded per run; e.g., a 500 km/h train at 3.5 GHz gives f_D ≈ 1.6 kHz, which lands in our 1–2-bin regime — the high-speed scenario is exactly the parameter regime our sweeps cover.

### C. CP conventions (declared, two regimes)

- **Stage 1 (this paper's verification):** ONE frame-level CP, delay-only LTI channel — a declared smoke test that proves transform algebra only; the interference-cancellation literature [2] itself assumes no CP.
- **Stage 2 onward:** per-slot CP with the time-varying model, Doppler phase advancing over **all** samples including CP:

  `r[n] = Σ_p g_p · exp(j2π ν_p n T_s) · s[n − l_p] + w[n]`

### D. Pilot model

Embedded pilot with full guards first [2b]; the forward model makes data leakage explicit and measured, never waved away:

`y_p = S_p H(h) x_p + S_p H(h) S_d x_d + w_p`

where S_p selects pilot observations and S_d inserts unknown data. Full-guard isolation is verified before any reduced-guard study (leakage ≤ −25 dB of pilot-observation power, measured).

---

## IV. Proposed Method: UOR-Net

### A. Block A — unfolded ISTA channel estimator (K = 10 layers)

The DD channel is sparse in a dictionary of candidate atoms (integer delays × candidate Dopplers). With measurement matrix A built from the pilot/guard geometry:

```
h^(k+1) = S_{λ_k}( h^(k) + γ_k Aᴴ (y_p − A h^(k)) )
γ_k = σ(s_k)·(1/L)          (s_k init 0 ⇒ γ_init = 0.5/L, interior of the feasible interval)
λ_k = softplus(ρ_k) ≥ 0
S_λ(z) = max(1 − λ/|z|, 0)·z    (S_λ(0) = 0, explicit)
```

L is the exact spectral norm of AᴴA at 32×8 (fallback: certified Frobenius bound). The output is ĥ over **all** dictionary coefficients — no top-K selection, which would be discontinuous and block gradients.

### B. Block B — differentiable DD channel operator (no learning)

The estimated channel enters the detector only through a **weighted-atom operator**, applied batched so the dense sum is never materialized:

```
Ĥ = Σ_j ĥ_j H_j        (H_j = candidate atom operators)
y′ = y − Ĥ x_p          (pilot contribution removed)
G = Ĥ S_d               (data-only operator seen by the detector)
```

This operator is the **sole gradient path** from the detection loss back into Block A: because it is a weighted sum of fixed atoms, gradients reach every atom exactly — the mechanism that makes joint training well-defined (this replaced an earlier top-K + straight-through-estimator design, which blocked gradients).

### C. Block C — unfolded OAMP detector (T = 10 layers)

With G = ĤS_d and D = number of unknown data symbols:

```
r_t = x_t + W_t (y′ − G x_t)
W̃_t = v_t² Gᴴ (v_t² G Gᴴ + C_w)⁻¹ ;   W_t = D·W̃_t / tr(W̃_t G)   (divergence-free)
x_{t+1} = η_t (r_t, σ_t²)     η_t = QPSK posterior mean (denoiser)
```

At 32×8, D ≈ 240 ⇒ G Gᴴ dense complex64 ≈ 0.5 MB: a direct factorization is trivial, so dense reference first; structured iterative solves only with a stated tolerance, iteration cap, and declared trace method. **Honest caveat:** OAMP state evolution assumes approximately i.i.d./right-orthogonally-invariant operators; Ĥ is structured and worse under estimated CSI. We therefore make the per-layer variances (and a scalar) **learnable** — the OAMP-Net2 [8] remedy — treat SE as a modeling approximation, and report empirical sensitivity rather than assuming it away.

### D. Soft bits and losses

Symbol posteriors with equal priors: `p_a(z) ∝ exp(−|z−a|²/σ_t²)`, computed via **logsumexp** (never plain sums of small exponentials). Bit LLRs over the declared QPSK labeling:

`LLR_i = log( Σ_{a∈C_0(i)} p_a / Σ_{a∈C_1(i)} p_a )`

Detection loss: BCE on **data bits only** with logits = −LLR (`BCEWithLogits` convention, one mean reduction).

Channel-estimation supervision is **operator-space**: with K_probe = 8 fresh unit-norm complex Gaussian probe vectors u_i per batch,

`L_CE = Σ_i ‖(Ĥ−H)u_i‖² / (Σ_i ‖Hu_i‖² + ε)`

This supervises the operator's *action* — representation-free and honest under fractional Doppler, where no unique coefficient-level label exists. The headline CE metric is the effective-channel NMSE ‖Ĥ−H‖²_F/‖H‖²_F — the same object L_CE supervises. Joint loss: `L = α·L_CE + β·L_det` (α = β = 1, both O(1) at init; per-loss gradient norms logged).

### E. Parity at initialization

Every learnable scalar (γ_k, λ_k, v_t², filter scalar) is initialized to its classical value; a unit test asserts UOR-Net output ≡ classical ISTA + OAMP (< 1e-6 relative) **before any training**. Training can only improve from a guaranteed-sane start — and any measured gain is attributable to learning, not to architecture luck.

### F. Parameter and complexity budget

| Block | Layers | Fixed physics inside | Learnable | Params |
|---|---|---|---|---|
| A — estimator | 10 | dictionary A, mat-vecs, soft-threshold | γ_k, λ_k | 20 |
| B — operator | — | Ĥ = Σ ĥ_j H_j (batched) | none | 0 |
| C — detector | 10 | LMMSE-form filter, QPSK posterior | v_t², scalar | 20 |
| **Total** | **20** | | | **≈40** |

Complexity accounting (G8) includes filter setup, all solves, and trace estimation; per-frame latency, peak memory, and FLOPs (documented conventions) are reported for every headline table.

---

## V. Experimental Design (pre-registered)

### A. Scope

Decisive experiment at **ONE grid: 32×8** (16×4 debug), **QPSK**, rectangular pulses. Training SNR ∈ [−5, 15] dB (20 dB = reported extrapolation, not training). 50k procedurally generated training frames, 2k validation frames (independent seed), batch 16, Adam lr 1e-3 (halved on 3-epoch validation plateau; early stop patience 5); float32 training / float64 reference with declared dtype tolerances. **No external dataset exists for this problem — data are generated on the fly by the contract-tested simulator with exact channel and bit ground truth; splits are by channel realization.**

### B. The paired experiment (RQ)

Two training runs, **identical in architecture, initialization, and data budget**: (1) **separate** — CE frozen (α = 0) while the detector trains; (2) **joint** — detection gradients reach Block A through Ĥ. ≥5 independent training seeds. Metrics: data-position BER + effective-channel NMSE + error/bit counts + 95% Clopper–Pearson CIs.

### C. Generalization and controls

- **Held-out Doppler-offset band** (e.g., κ ∈ (0.3, 0.45)) excluded from training and hyperparameter selection.
- **CSI-error sweep:** operating condition is pilot-estimated CSI; controlled perturbations of true/estimated CSI form a separate sweep with a declared domain (G4).
- **Inverse-crime control:** final models *and* all classical baselines re-evaluated on channels from an independent generator (8× oversampled continuous-time waveform, true fractional delay, independent RNG stream).
- Test draws excluded from early stopping and model selection.

### D. Baselines

- **Primary (implemented here, same frames):** classical embedded-pilot CE + LMMSE; classical CE + same detector; true-CSI LMMSE reference curve; classical ISTA (same depth) for Block A; classical OAMP for Block C.
- **Secondary (reported numbers):** Yue [36], Zhang PnP [27], Gao [71], Duan [39]/[40], DD-DNN [15], GNN [19], 2D-RC [21].

### E. Statistical protocol

Per BER point: ≥ **10⁶ bits AND ≥ 100 error events** (≥ 10⁷ bits when BER < 10⁻⁴); 95% Clopper–Pearson CIs; zero-error points reported with an exact binomial upper bound. Eval seeds 1–20; all hyperparameters, learning curves, seeds, iteration budgets, and stopping rules logged in advance (CSV).

### F. Full-receiver reporting

Every headline table reports: data-position BER + effective-channel NMSE + counts/CIs + pilot/guard overhead (spectral efficiency) + per-frame latency (batch, warmup, hardware, precision declared; includes solves and trace terms) + peak memory + FLOPs/frame with documented conventions.

---

## VI. Verification Results (completed)

### A. Stage-1 transform verification (`src/otfs_demo.py`)

| Check | Result |
|---|---|
| T1 — transform round-trip identity (ISFFT/Heisenberg/Wigner/SFFT) | **PASS** < 1×10⁻¹² |
| T2 — noiseless chain, 4-tap LTI channel, 5 seeds | **PASS** — worst symbol error **2.22×10⁻¹⁵** (gate < 1e-8, float64) |
| Unitary normalization | mean time-sample energy **1.0**; net DD gain 1 |
| Sanity BER (true-CSI LMMSE, LTI) | 0.0156 @ 10 dB · 0.0000 @ 20 dB |
| Bibliography reconciliation | 73 numbered refs = 73 BibTeX entries |

*Interpretation (declared):* Stage 1 proves the **transform algebra** under unitary normalization on the delay-only special case. It does not validate fractional Doppler, per-slot CP, or any learned component — those are Stage-2+ gates with their own contract tests (dense-reference complex-phase/leakage comparison, adjoint identity ⟨Hx,y⟩ = ⟨x,Hᴴy⟩ < 1e-10, shift-direction check, noise-variance reproduction).

### B. Input–output representation demonstration (`src/io_demo.py`, 32×8 decisive grid)

| Quantity | Value |
|---|---|
| Input per frame | 358 bits → 179 QPSK data cells (+1 pilot + guards) → 264 time samples incl. CP |
| Channel | 4 random taps (unit energy) + AWGN at declared SNR |
| Output per frame | estimated DD grid → LLRs → hard bits on 179 data cells |
| Spectral efficiency (schematic guards) | ≈ 1.36 bits/sample |
| Data-position BER @10/20 dB (true-CSI LMMSE) | 0.0000 / 0.0000 |

Figures (produced and visually verified; see `results/`): **fig1** — QPSK constellation, DD resource map (pilot/guards/data), transmit waveform with CP; **fig2** — true sparse channel (4 taps) and noiseless received DD grid; **fig3** — noisy received grid at 10 dB and detector-output constellation with error marking.

### C. Reproducibility

Deterministic RNG seeds per experiment; library versions, grid order, CP convention, sample period, and seeds recorded per run; code versioned on GitHub (private during coursework).

---

## VII. Decisive Experiment — Pre-Registered Results Template

*The following is fixed before running (Stage 7). Cells marked "—" are pending and will be filled with measured values, CIs, and counts; a negative outcome (no joint-training gain) is a reportable result, not a failure.*

| Configuration | BER @ SNR points (CI) | Eff.-channel NMSE | Params | Latency/frame | Peak memory |
|---|---|---|---|---|---|
| True-CSI LMMSE (reference) | — | n/a | 0 | — | — |
| Classical CE + LMMSE | — | — | 0 | — | — |
| Classical ISTA (K) + classical OAMP (T) | — | — | 0 | — | — |
| UOR-Net, **separate** training (α = 0) | — | — | ≈40 | — | — |
| UOR-Net, **joint** training | — | — | ≈40 | — | — |
| Δ (joint − separate), per seed | — | — | — | — | — |

Pre-registered analyses: per-seed paired differences with CIs; sensitivity to held-out κ band; CSI-error sweep; inverse-crime side-by-side; NMSE-vs-BER consistency check (estimator quality must co-move with detection quality or the anomaly is investigated before any claim).

---

## VIII. Discussion and Limitations

1. **Claims discipline.** No "first" or uniqueness claims are made; the neighborhood is crowded (LISTA [7], OAMP-Net2 [8], PnP [27], Gao [71], Lei [29], Duan [39]). The stated contribution is an open, testable question plus a controlled mechanism study.
2. **Synthetic-only evidence.** All training and evaluation use simulated channels (G11: no measured-channel or OTA validation identified in the checked sources). The inverse-crime control partially mitigates model-fitting-to-generator; a measured-channel probe is future work.
3. **State evolution is approximate.** OAMP SE assumptions do not hold exactly for structured, estimated DD operators (G12); learnable variances absorb part of the mismatch and sensitivity is reported empirically.
4. **Rectangular pulses.** Extra inter-symbol/inter-carrier interference is present and configuration-dependent; results are not generalized to optimal pulses without reruns.
5. **Corpus counting.** The "≈10 unfolding works" figure bounds what our searches *found* under a stated rule, not what exists (G1); a full-text novelty sweep remains open.
6. **Scale.** All gates are defined at 32×8; larger grids and 16-QAM are deferred behind a measured solver-cost and peak-memory case.

---

## IX. Conclusion

We designed UOR-Net, a unified unfolded OTFS receiver that joins a sparse channel estimator to an OAMP detector through a differentiable delay–Doppler operator, with ≈40 learnable physics-anchored parameters and parity at initialization. The system model and all waveform conventions are verified against the primary source; the Stage-1 simulator passes its contract tests at 2.22×10⁻¹⁵ worst-case noiseless error under unitary normalization, and the complete input–output path is demonstrated on the decisive 32×8 grid. The decisive scientific question — whether joint training through the differentiable operator improves BER under fractional Doppler and estimated CSI — is pre-registered with a fixed statistical protocol and will be answered with the same evidence standard either way, negative results included. Future work (documented, deferred): MIMO-OTFS unfolding, learned transmitter-side parameters (pilot value/location, pre-equalization), learned-denoiser and zero-init refinement-head ablations, measured-channel/OTA probes, larger grids and 16-QAM.

---

## Acknowledgment of Sources

The full 73-entry reference list below is identical to the companion survey (`literature/literature_review.md` §References) and is reconciled 1:1 with `literature/references.bib`. Status: **PARTIALLY VERIFIED** — core entries DOI-checked via Crossref; the rest abstract/metadata level; author lists for [17], [29], [45], [71] completed from Crossref after audit round 4.

## References

[1] R. Hadani, S. Rakib, M. Tsatsanis, A. Monk, A. J. Goldsmith, A. F. Molisch, and R. Calderbank, "Orthogonal Time Frequency Space Modulation," *IEEE WCNC*, 2017. DOI: 10.1109/WCNC.2017.7925924.
[1b] R. Hadani et al., "Orthogonal Time Frequency Space (OTFS) modulation for millimeter-wave communications systems," *IEEE IMS*, 2017. DOI: 10.1109/MWSYM.2017.8058662.
[2] P. Raviteja, K. T. Phan, Y. Hong, and E. Viterbo, "Interference Cancellation and Iterative Detection for Orthogonal Time Frequency Space Modulation," *IEEE Trans. Wireless Commun.*, 2018. DOI: 10.1109/TWC.2018.2860011. arXiv:1802.05242.
[2b] P. Raviteja, K. T. Phan, Y. Hong, and E. Viterbo, "Embedded Pilot-Aided Channel Estimation for OTFS in Delay–Doppler Channels," *IEEE Trans. Wireless Commun.*, 2019. arXiv:1808.08360.
[3] Z. Wei, W. Yuan, S. Li, J. Yuan, G. Bharatula, R. Hadani, et al., "Orthogonal Time-Frequency Space Modulation: A Promising Next-Generation Waveform," *IEEE Wireless Commun.*, 2021. DOI: 10.1109/MWC.001.2000408.
[4] *Orthogonal Time Frequency Space Modulation* (monograph), CRC Press, 2022. DOI: 10.1201/9781003339021.
[5] E. Aslandogan, H. Ilhan, B. A. Ozden, and E. Aydin, "A Comprehensive Survey of Channel Estimation Techniques for OTFS in 6G and Beyond Wireless Networks," arXiv:2512.13032, 2025.
[6] T. O'Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," *IEEE Trans. Cognitive Commun. Networking*, 2017. DOI: 10.1109/TCCN.2017.2758370.
[7] N. Samuel, T. Diskin, and A. Wiesel, "Learning to Detect," *IEEE Trans. Signal Process.*, 2019. DOI: 10.1109/TSP.2019.2899805.
[8] H. He, C.-K. Wen, S. Jin, and G. Y. Li, "Model-Driven Deep Learning for MIMO Detection," *IEEE Trans. Signal Process.*, 2020. arXiv:1907.09439.
[9] X. Gao, S. Jin, C.-K. Wen, and G. Y. Li, "ComNet: Combination of Deep Learning and Expert Knowledge in OFDM Receivers," arXiv:1810.09082 (IEEE Trans. Wireless Commun., 2020).
[10] Y. Wei, M.-M. Zhao, M. Hong, and M.-J. Zhao, "Learned Conjugate Gradient Descent Network for Massive MIMO Detection," arXiv:1906.03814.
[11] J. Zhang, H. He, C.-K. Wen, S. Jin, and G. Y. Li, "Deep Learning Based on Orthogonal Approximate Message Passing for CP-Free OFDM," *IEEE ICASSP*, 2019. DOI: 10.1109/ICASSP.2019.8682639.
[12] X. Li, W. Yuan, Z. Li, et al., "Hybrid Message Passing Detection for OTFS Modulation," *IEEE WCNC*, 2023. DOI: 10.1109/WCNC55385.2023.10118784.
[13] R. Chong, S. Li, Z. Wei, M. Matthaiou, and D. W. K. Ng, "Analysis of Cross-Domain Message Passing for OTFS Transmissions," *IEEE ITW*, 2024. DOI: 10.1109/ITW61385.2024.10806925.
[14] "Data-Driven Receiver for OTFS System with Deep Learning," *IEEE IC-NIDC*, 2021. DOI: 10.1109/IC-NIDC54101.2021.9660432.
[15] "Low-complexity Delay-Doppler Symbol DNN for OTFS Signal Detection," *IEEE VTC2021-Spring*, 2021. DOI: 10.1109/VTC2021-Spring51267.2021.9448630.
[16] A. Naikoti and A. Chockalingam, "A DNN-based OTFS Transceiver with Delay-Doppler Channel Training and IQI Compensation," arXiv:2107.09376.
[17] Q. Cheng, Z. Shi, J. Yuan, and M. Zhou, "Environment-robust Signal Detection for OTFS Systems Using Deep Learning," *IEEE GLOBECOM*, 2022. DOI: 10.1109/GLOBECOM48099.2022.10000940.
[18] Y. Gong, Q. Li, F. Meng, X. Li, Z. Xu, et al., "Data-driven deep learning for OTFS detection," *China Communications*, 2023. DOI: 10.23919/JCC.2023.01.008.
[19] X. Zhang, S. Zhang, L. Xiao, S. Li, and T. Jiang, "Graph Neural Network Assisted Efficient Signal Detection for OTFS Systems," *IEEE Commun. Letters*, 2023. DOI: 10.1109/LCOMM.2023.3286800.
[20] W. Zhuang, Y. Mao, H. He, L. Xie, et al., "Approximate Message Passing-Enhanced Graph Neural Network for OTFS Data Detection," arXiv:2402.10071.
[21] J. Xu, K. Said, L. Zheng, and L. Liu, "2D-RC: Two-Dimensional Neural Network Approach for OTFS Symbol Detection," arXiv:2311.08543.
[22] "Deep Learning-Assisted Signal Detection for OTFS-NOMA Systems," *IEEE Access*, 2024. DOI: 10.1109/ACCESS.2024.3449812.
[23] "Delay-Doppler Frequency Domain-Aided Superimposing Pilot OTFS Channel Estimation Based on Deep Learning," *IEEE VTC2022-Fall*, 2022. DOI: 10.1109/VTC2022-Fall57202.2022.10012759.
[24] "Learning based Delay-Doppler Channel Estimation with Interleaved Pilots in OTFS," *IEEE VTC2022-Fall*, 2022. DOI: 10.1109/VTC2022-Fall57202.2022.10012974.
[25] "Deep Learning-based Channel Estimation for Massive MIMO-OTFS Communication Systems," *IEEE WTS*, 2024. DOI: 10.1109/WTS60164.2024.10536672.
[26] "Deep Learning Supported Path Prediction and Channel Estimation for MIMO-OTFS System With High Delay Resolution," *IEEE Trans. Veh. Technol.*, 2025. DOI: 10.1109/TVT.2024.3493921.
[27] X. Zhang, Z. Ni, W. Yuan, J. A. Zhang, and T. Q. S. Quek, "Deep Learning-based OTFS Channel Estimation and Symbol Detection with Plug-and-Play Framework," *IEEE Trans. Commun.*, 2025. DOI: 10.1109/TCOMM.2025.3636093; arXiv:2503.11102.
[28] "Channel Estimation for MIMO-OTFS Satellite Communication: a Deep Learning-Based Approach," *IEEE ICCCN*, 2024. DOI: 10.1109/ICCCN61486.2024.10637519.
[29] Y. Lei, Y. Liu, G. Liu, L. Wan, and K. Yang, "Superimposed-Pilot OTFS Under Fractional Doppler: Modular End-to-End Learning," arXiv:2601.22523, posted 30 Jan 2026.
[30] S. Gehlot, S. Shinde, S. Srivastava, and S. K. Yadav, "Multi-Snapshot Deep Denoising for Channel Estimation in OTFS Modulated Systems," arXiv:2605.29777, 2026.
[31] E. Ispir and I. P. Roberts, "A Deep Iterative Refinement Receiver for OTFS Symbol Detection in Doubly-Dispersive Channels," arXiv:2609.00465, 2026.
[32] "GAN-Based Fractional Doppler Channel Estimation for OTFS Modulation," *IEEE WCCCT*, 2025. DOI: 10.1109/WCCCT65447.2025.11027948.
[33] "Fractional Delay-Doppler Channel Estimation in OTFS with Sparse Superimposed Pilots using RNNs," *IEEE VTC2023-Spring*, 2023. DOI: 10.1109/VTC2023-Spring57618.2023.10199741.
[34] "OTFS Fractional Doppler Channel Estimation Method Based on Super-Resolution Networks," *IEEE WCNC*, 2024. DOI: 10.1109/WCNC57260.2024.10570822.
[35] "Physics-Guided Lightweight Dynamic Graph Liquid Transformer for OTFS Channel Estimation in High-Mobility Vehicle–UAV mmWave Communications," *IEEE Internet of Things J.*, 2026. DOI: 10.1109/JIOT.2026.3752236.
[36] Y. Yue, J. Shi, Z. Li, J. Hu, and Z. Tie, "Model-Driven Deep Learning Assisted Detector for OTFS With Channel Estimation Error," *IEEE Commun. Letters*, 2024. DOI: 10.1109/LCOMM.2024.3362970.
[37] "Model-driven learning for OTFS detection based on an enhanced GNN with prior information," *ICT Express*, 2026. DOI: 10.1016/j.icte.2026.02.009.
[38] "Model-Driven Deep Learning for OTFS Detection With Phase Noise in V2X Communications," *IEEE Commun. Letters*, 2026. DOI: 10.1109/LCOMM.2026.3651608.
[39] F. Duan, X. Liu, J. Chang, and G. Gui, "Physics-Aware Guided Deep Unfolding Framework for OTFS Channel Estimation Under Fractional Doppler Effects," *IEEE Trans. Veh. Technol.*, 2026. DOI: 10.1109/TVT.2026.3736276.
[40] F. Duan, J. Chang, X. Liu, and G. Gui, "Deep Unfolding Network for Delay-Doppler Sparse Channel Estimation in OTFS Systems," *IEEE Wireless Commun. Letters*, 2026. DOI: 10.1109/LWC.2025.3626987.
[41] "Deep Unfolding Network for Target Parameter Estimation in OTFS-based ISAC Systems," *IEEE WCNC*, 2024. DOI: 10.1109/WCNC57260.2024.10571155.
[42] "Deep Unfolded LAMP Network with Bernoulli-Gaussian Mixture Prior for Massive MIMO-OTFS Channel Estimation," *IEEE ICSIPC*, 2026. DOI: 10.1109/ICSIPC69751.2026.11584250.
[43] "A Deep Model-Based Channel Interference Mitigation for OTFS Signals in ISAC Systems," preprint, 2024. DOI: 10.22541/au.171118385.56320709/v1.
[44] "Sea Surface Small Target Detection Integrating OTFS and Deep Unfolding," *J. Marine Sci. Eng.*, 2025. DOI: 10.3390/jmse13101946.
[45] M. Marchese, H. Wymeersch, P. Spallaccini, S. Chinnici, and P. Savazzi, "Reduced-latency DL-based Fractional Channel Estimation in OTFS Receivers," *IEEE ICMLCN*, 2025. DOI: 10.1109/ICMLCN64995.2025.11140575; arXiv:2503.08234.
[46] "Channel Estimation and Equalization for CP-OFDM-based OTFS in Fractional Doppler Channels," *IEEE ICC Workshops*, 2021. DOI: 10.1109/ICCWorkshops50388.2021.9473532.
[47] "Efficient Channel Estimation for OTFS Systems in the Presence of Fractional Doppler," *IEEE WCNC*, 2023. DOI: 10.1109/WCNC55385.2023.10118609.
[48] S. R. Mattu, N. Mehrotra, V. Khammammetti, R. Calderbank, et al., "Zak-OTFS: A Predictable Physical Layer for Communications and Sensing," arXiv:2604.16781, 2026.
[49] F. Barati, R. K. Jaiswal, S. K. Mohammed, and R. Hadani, "MIMO Zak-OTFS: Channel Estimation, Detection, and Throughput Analysis," arXiv:2606.26420, 2026.
[50] J. Zheng, C.-H. Tung, Y. Yao, N. Mehrotra, et al., "Real-Time and Scalable Zak-OTFS Receiver Processing on GPUs," arXiv:2604.02266, 2026.
[51] M. A. Sheikh and R. Calderbank, "Delay-Doppler Duality: Simplifying Exact Equalization of Zak-OTFS in Time or in Frequency," arXiv:2610.16033, 2026.
[52] W. Wang, J. Guo, S. Wang, and X. Wang, "Pre-equalization Design for ISAC-OTFS Air-Ground Transmission: A Deep Learning Approach," arXiv:2412.04751.
[53] B. Shen, Y. Wu, W. Zhang, and G. Y. Li, "LEO Satellite-Enabled Grant-Free Random Access with MIMO-OTFS," arXiv:2208.01828 (IEEE TWC, 2023).
[54] X. Feng, C. Zhang, M. EL-Hajjar, and C. Xu, "OTFS-IM-Assisted Non-Terrestrial Networks Relying on Autoencoder-Aided Soft-Decision Detection," arXiv:2605.08912, 2026.
[55] M. Men, T. Zhou, K. Bao, and Z. Guo, "A Novel Deep Learning-Based Coarse-to-Fine Frame Synchronization Method for OTFS Systems," arXiv:2601.05920, 2026.
[56] M. Kazemian and J. Jasperneite, "Enhancing 6G Wireless Intelligence: Do LLMs Work for CSI Prediction?," arXiv:2604.04028, 2026.
[57] M. Kazemian and J. Jasperneite, "A Conditional Variational Framework for Channel Prediction in High-Mobility 6G OTFS Networks," arXiv:2601.03084, 2026.
[58] E. Mittag, L. Brechtel, M. Stark, and H. D. Schotten, "Toward Agentic AI for the 6G Physical Layer: From Algorithm Discovery to Verified Baseband Silicon," arXiv:2610.27797, 2026.
[59] F. Aït Aoudia, J. Hoydis, S. Cammerer, and G. Marti, "Autonomous Discovery of Wireless Communications Algorithms," arXiv:2607.17762, 2026.
[60] B. Mandal, A. Chandra, R. Roka, and J. Wojtun, "6G NTN Waveforms: A Comparison of OTFS, AFDM and OCDM in LEO Satellite Channels," arXiv:2602.09834, 2026.
[61] R. J. Gachaba, M. Sarker, and A. Bhowal, "Deep Reinforcement Learning for Optimization of STAR-RIS Phase and Energy Splitting Coefficients in OTFS-NOMA Framework," arXiv:2609.04536, 2026.
[62] Y. Zhang, J. Yan, Y. Liu, and T. Ke, "MIMO-OTFS-Based Semantic Communication for High-Mobility Scenarios," arXiv:2605.29332, 2026.
[63] Y.-Y. Huang, Y.-M. Yeh, and J.-Y. Pan, "GPU-Accelerated Real-Time Software Defined Radio-Based OTFS Network-Coded Cooperation System: Hardware Implementation," arXiv:2606.30565, 2026.
[64] X. Zhang, Y. Zeng, and F. C. P. Shin, "OTFS for Joint Radar and Communication: Algorithms, Prototypes, and Experiments," arXiv:2510.00668, 2025.
[65] C. Cao, Z. Liu, Z. Dong, and Y. Zhou, "Nonparametric Variational Bayesian Learning for Channel Estimation with OTFS Modulation," arXiv:2602.10438, 2026.
[66] S. Gehlot, S. Srivastava, S. K. Yadav, and L. Hanzo, "Gaussian Mixture Model Based Bayesian Learning for Sparse Channel Estimation in OTFS," arXiv:2603.26902, 2026.
[67] S. Saxena, S. Srivastava, A. K. Jagannatham, and L. Hanzo, "Data-Aided Bayesian Learning for CSI Estimation over Doubly-Selective DCO-OTFS MIMO VLC Channels with Affine-Precoded Superimposed Training Sequences," arXiv:2609.22312, 2026.
[68] L. Anseaume, B. Böck, F. Weißer, and W. Utschick, "OTFS Channel Estimation Utilizing Sparse Bayesian Generative Modelling," arXiv:2609.01074, 2026.
[69] H. Rou, G. T. F. de Abreu, E. Björnson, and S. Kim, "The Resurrection of Spectrum Spreading for 6G and Beyond: From Sinusoids to Chirps," arXiv:2605.00249, 2026.
[70] B. E. Ouakouak, S. E. Zegrar, and H. Arslan, "Timing Misalignment and Its Impact on OTFS and Other Delay-Based Waveforms," arXiv:2610.17554, 2026.
[71] Y. Gao, X. Xu, Y. Jin, W. Yuan, J. Zhang, and S. Xu, "Joint Channel Estimation and Data Detection for OTFS Systems: A Lightweight Deep Learning Framework With a Novel Data Augmentation Method," *IEEE Internet of Things Journal*, 2025. DOI: 10.1109/JIOT.2025.3586012.

[S11] anafreis/OTFS_CE — LSTM-based OTFS channel-estimation reproduction repository (MATLAB), github.com/anafreis/OTFS_CE. *(Existence verified via GitHub API, audit round 4; checkpoint weights availability NOT verified.)*
[S12] textremo/Phy_Mod_OTFS — open OTFS toolbox (MATLAB), github.com/textremo/Phy_Mod_OTFS. *(Existence verified via GitHub API, audit round 4.)*
