# Study Guide — Everything Before Review 1

**Project:** Model-Driven Deep Learning for OTFS Modulation · 13 Nov 2026
**How to use:** 8 sessions, ~60–90 min each. Read → answer the self-checks → move on. Finish with the Q&A bank (§9) out loud. Pair with `docs/acronyms.md` open in a second window.

---

## Session 1 — Wireless channel basics (the physics underneath everything)

**Multipath.** A transmitted signal reaches the receiver over many reflected paths (buildings, ground, vehicles). Each path has its own:
- **delay** τ_p = extra path length ÷ speed of light. Typical spreads: 0.1–10 μs. More delay spread → more symbols overlap in time (**inter-symbol interference**).
- **Doppler shift** ν_p = relative radial velocity × carrier frequency ÷ c. Formula: `f_D = v·f_c/c`. Example: 500 km/h at 3.5 GHz → `f_D = (138.9 × 3.5e9)/3e8 ≈ 1.62 kHz`. Two effects: a moving receiver sees different paths arrive with different frequency shifts; more Doppler spread → the channel **changes within a frame**.

A channel with *both* significant delay and Doppler spread is **doubly dispersive**. This is the 6G regime: high-speed rail, V2X, UAVs, LEO satellites.

**Key intuition to keep:** delay = "when it arrives," Doppler = "how the arrival is frequency-smeared by motion." A channel that is *sparse in delay-Doppler* (few paths) is the structure every OTFS receiver exploits.

**Check yourself:** 1) What are the two spreads of a doubly dispersive channel? 2) Compute f_D for 120 km/h at 2 GHz. (Answer: 33.3 m/s → f_D ≈ 222 Hz.) 3) Why does motion cause *frequency* shift? (Path length changes over time → phase changes over time → instantaneous frequency shift.)

## Session 2 — OFDM and why Doppler breaks it

OFDM sends data on many narrow **subcarriers** that are mathematically **orthogonal** — they don't interfere *if the channel stays constant during a symbol*. The **cyclic prefix (CP)** copies the tail of each symbol to its front so multipath overlaps harmlessly.

**The failure:** under Doppler, subcarriers shift off their exact frequencies → orthogonality lost → **inter-carrier interference (ICI)**. The faster the motion, the worse. OFDM also treats the channel as *per-subcarrier fading*: some subcarriers fall into deep fades and their bits are lost.

**Check yourself:** 1) What does the CP fix, and what does it NOT fix? (Fixes multipath overlap; does NOT fix time-variation.) 2) Why does orthogonality matter? (Without it, demodulating one subcarrier picks up energy from all others.)

## Session 3 — OTFS mechanics (our Stage-1 territory)

**The idea:** place data on a 2D **delay–Doppler (DD) grid** (M delay bins × N Doppler bins), then transform it into a normal time–frequency signal and transmit. Transform chain (all classical, no learning):

```
bits → QPSK → DD grid (with pilot+guards) → ISFFT → Heisenberg (per-slot IFFT) → CP → channel
receive:  remove CP → Wigner (per-slot FFT) → SFFT → DD grid → estimate channel → detect
```

- **ISFFT** (Raviteja Eq. 2): `X[n,m] = (1/√NM) Σ x[k,l] e^{j2π(nk/N − ml/M)}` — mixes Doppler index k and delay index l into time–frequency.
- **Conventions (verified from the paper itself):** k = **Doppler** axis (N points), l = **delay** axis (M points). Fractional Doppler: `ν = (k+κ)/(NT)`, offset κ ∈ (−½, ½].
- **Why it helps:** with ideal pulses, each path stays a point and every symbol sees approximately the *whole* channel (same average gain) instead of one fading snapshot. **Qualified:** rectangular pulses (our case) add extra inter-symbol/inter-carrier interference; fractional Doppler smears one path across neighboring Doppler bins — a **Dirichlet-type leakage** — which reduces effective sparsity. How much this hurts depends on pulse/CP/grid/receiver.
- **Interference types:** IDI (inter-Doppler), ICI (inter-carrier), ISI (inter-symbol) — rectangular pulses cause all three; Raviteja's ICID derives exact equations and an iterative detector.
- **Our demo choices (declared):** unitary normalization (transform energy preserved: mean sample energy 1.0), one frame-level CP for the Stage-1 smoke test only; Stage 2+ moves to per-slot CP with the time-varying model `r[n] = Σ_p g_p e^{j2πν_p nTs} s[n−l_p] + w[n]` — Doppler phase advancing over all samples including CP.

**Check yourself:** 1) What domain is data in, and why? 2) What does κ represent? 3) Why can't fractional Doppler be ignored? (Off-grid paths leak into neighboring bins → channel looks denser → sparsity-based detectors degrade.)

## Session 4 — Channel estimation: pilots and LMMSE

Before detecting, the receiver must know the channel (**CSI** — channel state information). Standard scheme (**embedded pilot**, Raviteja 2019): put one strong known symbol (impulse) on the DD grid surrounded by **guard** symbols (off/zero) so the pilot's response spreads without corrupting data:
- **Full guards** = pilot protected completely → clean observation, but overhead costs bandwidth.
- **Reduced guards** = cheaper, but data leaks into the pilot observation — must be modeled as extra noise or subtracted using detected data. (Our plan: verify full-guard isolation first; measure leakage.)
- **LMMSE** = the best linear filter under Gaussian assumptions — our main classical baseline. Uses channel statistics to beat plain zero-forcing in noise.
- **How we grade an estimator:** **effective-channel NMSE** = error energy ÷ true channel energy (in dB), measured on the *actual operator the detector uses* — not on internal coefficients.

**Check yourself:** 1) Why guards around the pilot? (Pilot's channel response spills into neighbors.) 2) Why is NMSE measured on the effective channel? (Representation-independent; comparable across methods.)

## Session 5 — Sparse recovery and ISTA (Block A's engine)

The DD channel has few paths → it is **sparse** in a suitable **dictionary**: a set of candidate "channel shapes" (atoms H_j), one per integer delay × candidate Doppler. The observation is a mixture: `y ≈ A·h + noise`, where h has few nonzero entries.

**ISTA** (Iterative Shrinkage-Thresholding Algorithm) — the classic sparse solver:

```
repeat:   h ← S_λ( h + γ·Aᴴ(y − A h) )        S_λ(z) = max(1 − λ/|z|, 0)·z
```

Each iteration: (1) compare prediction vs measurement (**residual**), (2) nudge h along the residual, (3) **soft-threshold** — shrink small values to exactly zero, keeping the solution sparse. Rules: step size γ ≤ 1/L where **L = spectral norm of AᴴA** (the "speed limit"); threshold λ controls sparsity. **FISTA** = ISTA + momentum (faster).

**The off-grid problem:** a path at Doppler k+κ doesn't match any single atom; it lights up several (leakage). Fixes in our design: oversampled dictionary (several candidate Dopplers per bin — but neighboring atoms become correlated, "coherence"), and supervising the estimator on the **operator's action** (probe-vector loss, §8) rather than pretending a unique coefficient label exists.

**Check yourself:** 1) What does the soft-threshold do? 2) Why is there a maximum step size? (Too big → divergence.) 3) Why does fractional Doppler break a plain on-grid dictionary?

## Session 6 — Detection: from brute force to OAMP (Block C's engine)

Detection = recover the QPSK symbols from the received DD grid given the (estimated) channel.
- **Optimal ML** = try every symbol combination — exponentially complex, impossible for real frames.
- **MMSE** = linear filter, good baseline.
- **Message passing / AMP** = iterative statistical detectors exploiting the sparse DD structure; degrade when the channel is dense or off-grid.
- **OAMP (Orthogonal AMP)** = AMP refined: a **decorrelated linear filter** `W_t = D·W̃_t/tr(W̃_t G)` with `W̃_t = v_t²Gᴴ(v_t²GGᴴ + C_w)⁻¹`, paired with a **divergence-free denoiser** (the QPSK posterior mean). Its variance bookkeeping (**state evolution**) is only approximate for our structured channel — which is exactly why our unfolded version makes variances **learnable**.

**Soft output:** for each symbol the detector produces probabilities per constellation point `p_a`, converted to bit **LLRs**: `LLR_i = log(Σ_{a:bit i =0} p_a / Σ_{a:bit i =1} p_a)` — computed via **logsumexp** for numerical safety. Loss = **binary cross-entropy** on data bits only, with logits = −LLR.

**Check yourself:** 1) Why can't we do ML detection? 2) What two parts alternate in OAMP? 3) What is an LLR, qualitatively? (Confidence that bit = 0 vs 1.)

## Session 7 — Deep learning + unfolding (the paradigm)

**DL minimum:** a network = layers of simple weighted sums + nonlinearities; a **loss** scores predictions; **gradients** tell each weight how to change (backpropagation); **Adam** = the standard optimizer; train/validate/test splits prevent memorization (**overfitting**).

**Deep unfolding (model-driven):** take an iterative algorithm (ISTA, OAMP), unroll its loop into K layers, keep the physics fixed inside each layer, and make only a few scalars learnable — our γ_k, λ_k (estimator) and per-layer variances (detector). Famous precedents: LISTA (learned ISTA), He et al.'s OAMP-Net2 for MIMO (our Block C's direct ancestor, Eqs. 24–27).

**Why this beats both extremes:** few parameters (~hundreds) → little data, fast, interpretable; physics inside → doesn't have to rediscover Maxwell; learnable scalars → absorbs exactly the assumptions that break (SE mismatch, off-grid effects).

**Parity-at-init discipline:** if we initialize every learnable scalar to its classical value, the network must behave *identically* to the classical algorithm — we unit-test that (<1e-6). Training then only improves from a guaranteed-sane start.

**Check yourself:** 1) What becomes a layer? 2) What becomes learnable? 3) Why is parity-at-init a test and not an assumption?

## Session 8 — Our system end-to-end (know this cold)

**Blocks:**
- **A — Unfolded ISTA estimator** on the pilot observation `y_p = S_p H x_p + S_p H S_d x_d + w_p` (the second term is data leaking into the pilot — modeled, measured; full guards ⇒ ≈0).
- **B — Differentiable DD operator** `Ĥ = Σ ĥ_j H_j` — weighted atoms, applied batched (never materialized); pilot removed: `y′ = y − Ĥx_p`.
- **C — Unfolded OAMP detector** on `G = ĤS_d`; logsumexp LLRs; BCE on **data bits only**.

**Losses:** `L_CE` = operator-space loss on probe vectors `L_CE = Σ_i ‖(Ĥ−H)u_i‖² / (Σ_i ‖Hu_i‖² + ε)` (supervises what the estimate *does*, not internal coefficients — honest under fractional Doppler); `L_det` = BCE on data bits. Joint: `L = α·L_CE + β·L_det`.

**The decisive experiment:** paired **separate-vs-joint** training — identical blocks, identical initialization, identical data budget; run 1 freezes CE while the detector trains, run 2 lets detection gradients reach CE **through Ĥ**. ≥5 seeds, held-out Doppler-offset band, CSI-error sweep, classical baselines (CE+LMMSE, true-CSI LMMSE) on the same frames.

**Controls:** one grid (32×8) until reproducible; 16×4 debug; bigger grids/16-QAM deferred; SNR training range −5…15 dB (20 dB = extrapolation); κ signed offsets incl. 0; test draws excluded from model selection; ≥10⁶ bits AND ≥100 error events per BER point with 95% CIs; report latency/memory/FLOPs with documented conventions.

**Pretrained models:** permitted, not required → **train from scratch** (default); external-checkpoint route documented as optional (LSTM repo weights = CE-transfer only).

**Check yourself:** 1) Recite the research question. 2) Where do detection gradients flow in joint training? (Through Ĥ into Block A's ĥ_j.) 3) Why is CE supervision operator-space and not coefficient-space? 4) Name the three classical baselines on identical frames.

## Session 9 — Literature in 10 lines (what to say about the survey)

1. Surveyed arXiv (~160) + Crossref (~300) on 12 Nov 2026 → **73 references**, evidence-labeled.
2. OTFS born 2017 (Hadani); discrete model + iterative detection 2018 (Raviteja — our conventions verified from the paper itself).
3. Data-driven wave 2021–23: DNN/GNN detectors, CNN estimators — opaque, retrain-per-config.
4. **Model-driven line born 2024** — Yue et al.'s unfolded detector (robust to CE error) is the anchor.
5. 2025–26: joint CE+DD appears in hybrid (Zhang PnP, TCOM) and data-driven (Gao, IoT-J) forms; fractional-Doppler CE unfolding (Duan, TVT 2026); phase-noise model-driven detector (CL 2026).
6. Pattern: **almost everyone unfolds ONE module**; the other is given.
7. Fractional Doppler: handled on estimation side + one hybrid detector; **unfolded detector under fractional Doppler + estimated CSI is not documented**.
8. No shared benchmark (though public code exists — toolbox + LSTM-CE repo, verified on GitHub).
9. Complexity reporting uneven; no guarantees for learned receivers (only algorithm-level proofs).
10. Hence our research question — an **open, testable hypothesis**, not a novelty claim.

**Six headline gaps:** young fragmented field · joint training unverified · detector-side fractional Doppler open · robustness/generalization splits missing · no shared benchmark · no guarantees for learned receivers.

## Session 10 — Viva Q&A bank (practice out loud)

**Concept:**
1. *Why OTFS over OFDM?* — Delay–Doppler sparsity + full-channel-gain experience per symbol under high mobility; OFDM loses orthogonality under Doppler. (Qualified: depends on pulse/CP/grid/receiver.)
2. *What is deep unfolding?* — Iterative algorithm → layers; physics fixed; few scalars learnable; trained end-to-end.
3. *Why model-driven, not a plain DNN?* — O(10²) parameters vs O(10⁵+); interpretable; less data; generalizes via physics.
4. *Why is the DD channel sparse?* — Few physical paths; each maps to one DD point (smearing only under fractional Doppler).

**Design choices:**
5. *Why ISTA for estimation?* — Natural fit to sparse DD recovery; unfolding adds learnable step/thresholds; off-grid honesty via probe-vector supervision.
6. *Why OAMP for detection?* — Strong classical baseline; clean linear+denoiser structure to unfold; OAMP-Net2 lineage proves the recipe works in MIMO.
7. *Why the weighted-atom operator (Ĥ = Σ ĥ_j H_j)?* — Differentiable path from detection loss to estimator; top-K selection is discontinuous and blocks gradients (superseded from v2.2).
8. *Why supervise CE in operator space?* — Off-grid truth has no unique coefficient label; what matters to the detector is the operator's action.
9. *Why full guards first?* — Isolation must be verified before trusting any reduced-guard trick; leakage measured, not assumed.
10. *Why 32×8?* — Small enough for dense reference solves and fast reproducible experiments; scaling gated on measured solver cost.

**Defense (why not X):**
11. *Why not train the transmitter?* — Tx learning needs closed-loop feedback and would confound the receiver experiment; recorded as future work (pilot-value learning, pre-equalization).
12. *Why not use pretrained models?* — Permitted but not required; no compatible checkpoint verified for our architecture; from-scratch is cleaner and the repo weights (LSTM CE) don't transfer to unfolding.
13. *Why not claim novelty?* — Neighborhood is crowded (PnP, Gao, Lei); our claim is a clean mechanism study; novelty sweep is an open item.
14. *What if joint training shows no gain?* — That is a reportable result: the deliverable is the answer to the question, negative results included.
15. *Isn't state evolution wrong for your channel?* — Yes, approximately — that's why variances are learnable and we report sensitivity instead of assuming SE holds.

**Numbers to have ready:**
16. Grid? **32×8** (decisive), 16×4 debug. Modulation? **QPSK**. Pulses? **Rectangular**.
17. References? **73** (reconciled 1:1 with BibTeX). Audits survived? **4 rounds, all findings accepted/verified.**
18. Stage-1 accuracy? **2.22×10⁻¹⁵** worst-case symbol error, noiseless; transforms unitary (energy 1.0).
19. κ range? **(−½, ½]** (Raviteja Eq. 6). Training SNR? **−5…15 dB; 20 dB = extrapolation**.
20. BER protocol? **≥10⁶ bits AND ≥100 error events/point; 95% Clopper–Pearson CIs.**
21. Seeds? **≥5 independent training seeds** for the paired experiment.
22. Timeline? **10 weeks from 17 Nov 2026**; Stages 2–7 mapped week by week; Stage 1 done.
23. Framework? **NumPy/SciPy now; PyTorch for training** (documented in README).
24. Metrics? **Data-position BER + effective-channel NMSE + latency/peak-memory/FLOPs with documented conventions.**

**If asked something you don't know:** "That's outside what I verified so far — I'd check the primary source before claiming it" — that sentence is worth more than a guess.
