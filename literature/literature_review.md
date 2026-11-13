# Literature Review — Model-Driven Deep Learning for OTFS Modulation

**Project:** MCS, KLH (2nd Year, Even Semester) · **Date compiled:** 12 November 2026 · **Revision:** post-audit round 4
**Scope:** OTFS (Orthogonal Time-Frequency Space) modulation and the model-driven deep-learning paradigm applied to its receiver design. Includes: foundations, data-driven DL for OTFS, model-driven/unfolding approaches, fractional-Doppler channel estimation, MIMO-OTFS/ISAC/NTN extensions, complexity and hardware aspects — and a **full gap analysis**.
**Sources searched:** arXiv API (≈160 candidate records screened), Crossref (IEEE/Springer/Elsevier journal metadata, ~300 candidate records screened), Hugging Face Papers API. Reference list at the end; bracketed numbers like [1] refer to it.
**Verification status: PARTIALLY VERIFIED.** Core entries were DOI-checked against Crossref; most others were checked at abstract/metadata level only. Full texts were **not** systematically read on our side; passages marked "verified" below come either from our abstract-level checks or from an external audit's full-text checks (labeled). See §6, "Evidence standard".

---

## 1. Introduction

Fifth/sixth-generation (5G/6G) wireless systems must support very high mobility — high-speed rail (500 km/h), V2X, UAV swarms, and LEO satellite links — where the channel is *doubly dispersive* (both delay spread and Doppler spread are significant). Orthogonal Frequency Division Multiplexing (OFDM), the 4G/5G waveform, degrades under large Doppler because it destroys subcarrier orthogonality, producing Intercarrier Interference (ICI).

**OTFS modulation** [1] answers this by spreading data symbols over the full time–frequency plane using a two-dimensional transform (ISFFT/SFFT) applied to the *delay–Doppler (DD) domain*. Within one frame, DD-domain coordinates give a **compact representation of a rapidly time-varying multipath channel**: each path appears at one (delay, Doppler) location, and every data symbol interacts with (approximately) the whole channel rather than one fading snapshot. In the idealized case each symbol then sees a (nearly) non-fading, approximately constant-gain channel — but multipath fading and DD-domain inter-symbol coupling still affect detection, and the idealization weakens when fractional Doppler spreads energy across Doppler bins and reduces effective sparsity. How much this matters depends on the pulse shape, cyclic-prefix convention, channel profile, grid size and receiver — it is not a universal law. OTFS is a leading waveform candidate for 6G NTN, ISAC and high-mobility scenarios [3], [60].

The catch: **OTFS receivers are hard**. Optimal ML detection is NP-hard; practical iterative detectors (message passing (MP), approximate message passing (AMP), variational Bayes (VB), Gaussian samplers) rely on sparsity and independence approximations whose accuracy degrades when the DD channel is dense, fractional (off-grid) Doppler shifts are present, or the channel estimate is imperfect [2], [12], [13], [46], [47]. Classical detection does not *inevitably* fail in these regimes — the literature shows degradation that grows with Doppler offset and channel density, and well-damped implementations remain strong baselines.

**Deep learning (DL)** has been applied to OTFS receivers since ~2021. The literature splits into three design philosophies [6]–[9]:

| Paradigm | Idea | Pros | Cons |
|---|---|---|---|
| **Model-based** | Derive algorithms from signal statistics (MP, AMP, VB, MMSE, SBL) | Interpretable, provable, low data need | Approximations degrade in hard regimes; hand-tuned |
| **Data-driven** | Train generic black-box DNN/CNN/GNN/Transformer on (input, target) pairs | Learns without closed-form assumptions | Uninterpretable, poor generalization, needs huge data, no performance guarantees |
| **Model-driven (a.k.a. model-aided, deep unfolding, algorithm unrolling)** | Unfold an iterative *model-based* algorithm into a network whose iterations become layers with a few *learnable* trainable parameters | Small networks, fewer parameters, faster convergence, interpretability, physics fidelity, less data | Design tied to the chosen model; theory still immature |

**Model-driven deep learning** is the focus of this review. The seminal generic works are "Learning to Detect" [7], model-driven MIMO detection (OAMP-Net2 lineage) [8], ComNet [9], learned conjugate-gradient networks [10], and OAMP-based CP-free OFDM receivers [11]. The key attraction for OTFS: an unfolded message-passing or MMSE detector inherits the DD-domain channel structure (2D-convolutional sparsity, delay/Doppler locality) that pure black boxes must relearn from data.

---

## 2. Search Methodology (evidence standard)

- **Databases and dates:** arXiv API (query patterns: `"OTFS" AND "deep learning"`, `"OTFS" AND "detection"`, recent-OTFS by submission date, `"model-driven deep learning"` in eess.SP) and Crossref API (queries: model-driven OTFS, deep unfolding OTFS, GNN OTFS, superimposed-pilot OTFS DL, transformer OTFS, fractional-Doppler OTFS, PAPR OTFS), plus Hugging Face Papers API — all run 12 Nov 2026.
- **Screening stages (recorded as run):** (1) harvest ≈160 arXiv + ≈300 Crossref candidate records; (2) deduplicate by DOI/arXiv ID and title match; (3) metadata/abstract screening for receiver relevance; (4) retention. *Caveat:* exact deduplicated counts were not logged per query; the figures above are stage-level approximations, not auditable counts.
- **Retention:** ~70 works catalogued below; ~30 excluded as off-topic (e.g., OTFS power transfer, non-receiver topics).
- **Counting rule for "model-driven" (G1):** *explicit unfolding* = an iterative model-based algorithm whose iterations become network layers with trainable algorithmic scalars (step sizes, damping, thresholds, denoiser variance). *Hybrid* = model-informed components inside a learned architecture (e.g., PnP, AMP+GNN) without explicit unrolling. *Physics-informed learning* = physics used in loss/features/architecture priors. Gap counts below name which rule produced them.

---

## 3. Thematic Review

### 3.1 OTFS foundations and classical (non-DL) receivers [2017–2026]

Hadani et al. introduced OTFS at WCNC 2017 [1] and demonstrated it for mmWave links the same year [1b]. Raviteja, Phan, Hong and Viterbo formalized the discrete DD-domain input–output model with inter-Doppler/inter-carrier/inter-symbol interference, developed interference-cancellation and iterative (message-passing-style) detection, and established the now-standard rectangular-pulse OTFS framework [2] — noting that this paper assumes **no cyclic prefix** and derives exact I/O relations via phase compensation (Eq. (23)/(27) of [2]). Embedded pilot-aided channel estimation in the DD domain followed [2b], defining the impulse-pilot + guard-region structure (full-guard and reduced-guard variants, §III-B of [2b]) used by most later papers.

The tutorial/overview literature consolidated the field: Wei, Yuan et al. [3] laid out the OAMP and Transform-Domain Message Passing (TMP) detectors — TMP exploits the 2D-convolutional sparsity of the DD channel, OAMP exploits large-random-matrix behavior — and the CRC monograph [4] systematized receivers, channel estimation, and pulse shaping. Classical receiver families now include: message passing / hybrid MP [12], cross-domain MP [13], MMSE-RC zero-padded variants, variational Bayes, Gaussian samplers, EP and SBL-based estimators [65]–[68].

**Known classical limitation (qualified):** the in-phase approximation and on-grid (integer) Doppler assumption. Fractional Doppler spreads path energy across Doppler bins (a Dirichlet-type finite-frame leakage [2], Eq. (17)), reducing effective sparsity and degrading MP-type detectors *in the regimes studied in [46], [47]*. This is a major error source in those studies — though its impact depends on pulse, CP convention, channel profile, grid and receiver (see §6, G3 phrasing).

A notable 2024–2026 theoretical revival is **Zak-OTFS** — reformulating OTFS via the Zak transform so the DD domain becomes a *predictable* (non-fading) physical layer [48], [49], with exact equalization duality [51] and real-time GPU receiver implementations [50]. This line is almost entirely classical/model-based so far, with essentially **no explicit-unfolding treatment** in our corpus (counting rule of §2).

### 3.2 Foundations of deep learning for the physical layer [2017–2020]

O'Shea & Hoydis established end-to-end learned transceivers as autoencoders [6]. Samuel, Diskin & Wiesel's "Learning to Detect" [7] first showed that a DNN trained per-layer on MMSE estimates can approach L-MMSE performance with real-time feasibility. **Model-driven detection** was crystallized by He, Wen, Jin & Li [8] (unfolding OAMP/AMP iteration dynamics into layers — the OAMP-Net2 family; its per-layer linear-filter and variance updates, Eqs. (24)–(27), are the direct ancestor of this project's detector) and by ComNet [9]. Learned CGD [10] unfolded conjugate-gradient iterations; OAMP unfolding for CP-free OFDM [11] is the closest OFDM-domain ancestor of today's OTFS unfolders. Collectively these established the recipe: **identify an iterative algorithm → make key scalars trainable → train end-to-end with the physics fixed**.

### 3.3 Data-driven deep learning for OTFS [2021–2026]

The first OTFS receivers with DL were black boxes:

- **Detection:** "Data-Driven Receiver for OTFS" (IC-NIDC 2021) [14]; the low-complexity DD-symbol DNN (VTC 2021) [15]; DNN transceiver with DD-channel training and IQ-imbalance compensation [16]; environment-robust OTFS detection (GLOBECOM 2022) [17] — its abstract reports adaptation using **one sample from a new environment** (few-shot adaptation claim; audit-checked); a systematic data-driven comparison in China Communications [18]; reservoir-computing (2D-RC) [21]; OTFS-NOMA detection [22].
- **Graph/relational models:** GNN-assisted efficient detection (Comm. Letters 2023) [19] and the AMP-enhanced GNN detector [20] — per the external audit's full-text check (S3): **fractional Doppler is explicitly included (§II)**, and the paper **reports FLOPs and measured runtime (Table I)** and trains at a specified noise level while evaluating BER across SNR; CSI robustness was tested as perfect CSI plus perturbed path gains (Fig. 2). [20] is a *hybrid* (model-informed initialization + learned message functions), not an explicit unrolling.
- **Channel estimation:** superimposed-pilot DD-frequency-domain estimation with DL (VTC 2022) [23]; interleaved-pilot learning [24]; massive-MIMO-OTFS CNN estimation [25]; MIMO-OTFS path prediction + estimation (TVT 2025) [26]; satellite MIMO-OTFS estimation (ICCCN 2024) [28]; multi-snapshot deep denoising [30]; GAN-based [32], RNN-based [33] and super-resolution-network-based [34] fractional-Doppler estimators; a physics-guided dynamic-graph "liquid" transformer (IoT-J 2026) [35].
- **Joint CE + detection (hybrid):** the Plug-and-Play framework of Zhang, Ni, Yuan, Zhang & Quek (IEEE TCOM 2025) [27] plugs learned denoisers into an iterative algorithmic framework covering both tasks; per the audit's full-text check (S4): a CSI-error sweep at fixed 12 dB SNR (Fig. 11(b)), fractional-Doppler BER (Fig. 10(b)), and a *sequential* (not jointly backpropagated) estimator→detector pipeline (Fig. 12). It is a prominent recent framework — priority/superlative language is avoided here because our comparison base does not establish it.
- **Joint CE + detection (end-to-end, data-driven):** Gao, Xu, Jin, Yuan, Zhang & Xu, "Joint Channel Estimation and Data Detection for OTFS Systems: A Lightweight Deep Learning Framework…" (*IEEE Internet of Things J.*, 2025, DOI 10.1109/JIOT.2025.3586012) [71] — a second unified joint CE+detection framework (authors completed from Crossref after audit round 4). Jointly with [27] it confirms that "joint CE+DD" alone is **not** a sufficient novelty axis.
- **End-to-end/learnable transceivers:** modular end-to-end learning over superimposed pilots under fractional Doppler — Lei, Liu, Liu, Wan & Yang, arXiv:2601.22523, **posted 30 Jan 2026** [29] (per the audit's full-text check S5: §III-B optimizes the *complex pilot value* end to end at a designated location); deep iterative-refinement receivers for doubly-dispersive channels [31].

**Assessment:** data-driven OTFS receivers deliver good BER but typically need per-configuration retraining, degrade off-distribution, and are opaque — the weaknesses model-driven design targets. (Where individual papers *do* demonstrate adaptation [17] or SNR transfer [20], that is recorded in the gap tables rather than flattened into a field-wide claim.)

### 3.4 Model-driven deep learning for OTFS — the core topic [2024–2026]

By the §2 counting rule, this is where the explicit-unfolding literature is thin — the central finding of this review. Direct hits:

1. **Yue, Shi, Li, Hu & Tie (2024)** [36] — unfolds a model-based iterative detector whose inputs are the *estimated* DD channel, explicitly trained for robustness to channel-estimation error. ~20 citations — the recognized anchor for model-driven OTFS detection.
2. **Enhanced GNN with prior information (ICT Express 2026)** [37] — "model-driven learning" in the GNN sense: DD-domain priors injected into a graph network (hybrid by the §2 rule).
3. **Phase-noise-robust model-driven detection for V2X (IEEE Comm. Letters 2026)** [38] — extends the model-driven detector to an oscillator-phase-noise impairment; evidence that impairment-aware unfolding exists (relevant to G10's phrasing).
4. **Duan, Liu, Chang & Gui (2026, TVT)** [39] — physics-aware unfolding for OTFS channel estimation under fractional Doppler.
5. **Duan, Chang, Liu & Gui (2026, WCL)** [40] — companion unfolding for on-grid sparse DD estimation (FISTA-based CE).
6. **Deep unfolding for OTFS-ISAC target parameter estimation (WCNC 2024)** [41]; a deep model-based interference-mitigation preprint [43]; sea-surface small-target detection with unfolding (JMSE 2025) [44] — sensing-side unfoldings (sensing-only inference, not joint ISAC receivers — see G7).
7. **Deep-unfolded LAMP with Bernoulli–Gaussian priors for massive MIMO-OTFS channel estimation (ICSIPC 2026)** [42].
8. **Reduced-latency DL-based fractional channel estimation** — Marchese, Wymeersch, Spallaccini, Chinnici & Savazzi, IEEE ICMLCN 2025, arXiv:2503.08234 [45] (bibliographic record completed after audit round 4).

Adjacent, partially model-driven hybrids: PnP [27], AMP-enhanced GNN [20], modular end-to-end learning [29] — all *hybrid* under the §2 rule.

**Assessment:** the explicit-unfolding OTFS literature exists but is *fragmented*: (a) almost all works unfold **one module** (CE or detection) while treating the other as given; (b) "model-driven" is used loosely across the field — several papers use the term for any learned component bolted onto a model, not for genuine algorithm unrolling (hence the §2 counting rule); (c) complexity reporting is uneven (see G8); (d) no shared benchmark (same waveform, frame, pilots, CSI conditions, evaluation protocol) was identified in this audit, although public code repositories exist (see G9).

### 3.5 Fractional Doppler: the recurring crux

Classical estimators for fractional (off-grid) Doppler [46], [47] trade complexity against accuracy; super-resolution, RNN, GAN and SR-network estimators [32]–[34] attack it with data-driven tools; the physics-aware unfolding [39] attacks it with model-driven tools; the hybrid AMP-GNN detector [20] *detects* under fractional Doppler (audit full-text check); modular end-to-end learning [29] and reduced-latency CE [45] attack it from the system side. **What the checked sources do not show:** an *explicitly unfolded detector* trained/validated under fractional Doppler with a time-varying channel jointly — the detection-side unfolding works [36]–[38] were checked at abstract/audit level and report on-grid or pre-equalized assumptions ([20] is hybrid, not unfolded).

### 3.6 Extensions: MIMO, ISAC, NTN, applications

- **MIMO-OTFS:** DL works exist on the *estimation* side (massive MIMO-OTFS CNN [25], path prediction [26], satellite [28], LAMP unfolding [42]); MIMO-OTFS *detection* with explicit unfolding is unexplored in our corpus (Gap G6 — qualified there). Non-DL MIMO-OTFS theory is strong [49].
- **ISAC:** OTFS is attractive for joint radar-communication; prototypes exist [64]; DL used for pre-equalization [52] and unfolded *target-parameter* estimation [41] — sensing-only inference; no unified model-driven ISAC-OTFS *receiver* in our corpus (Gap G7 — qualified there).
- **NTN/LEO:** OTFS vs AFDM vs OCDM comparisons [60]; OTFS-IM autoencoder receivers [54]; grant-free random access with MIMO-OTFS [53]; DL frame synchronization [55].
- **New learning paradigms:** LLM-based CSI prediction [56], conditional variational channel prediction [57], agentic AI for PHY design [58], autonomous algorithm discovery [59] — context for where the field is heading; **not load-bearing for this project's receiver experiment** (G14).
- **Impairments/robustness:** phase noise [38], IQ imbalance [16], synchronization [55], environment adaptation [17] — see the impairment table in G10 for per-receiver coverage.
- **Complexity & hardware:** the hybrid line reports FLOPs and measured runtime ([20], Table I — audit full-text check); real-time *hardware* implementations exist for classical receivers (GPU Zak-OTFS [50], GPU-SDR network-coded OTFS [63]); hardware deployment and over-the-air validation for DL-OTFS receivers were not identified in the checked sources (Gap G8/G11 — qualified there).

---

## 4. Evolution Timeline

- **2017** — OTFS proposed [1]; mmWave demo [1b].
- **2018** — discrete DD I/O model + iterative detection [2]; embedded pilots [2b].
- **2017–2019** — DL-for-PHY foundations [6]–[11].
- **2019–2020** — OTFS tutorials formalize OAMP/TMP receivers [3]; fractional-Doppler problem studied [46], [47].
- **2021–2023** — first data-driven OTFS receivers (DNN/GNN/CNN detection and CE) [14]–[24].
- **2024** — *model-driven* OTFS detection appears (CE-error-robust unfolding) [36]; unfolding enters OTFS-ISAC [41]; AMP+GNN hybrid detector [20] (fractional Doppler; FLOPs/runtime reported).
- **2025** — joint CE+detection frameworks (PnP TCOM [27]; lightweight DL IoT-J [71]); unfolding for sensing [41], [44]; reduced-latency fractional CE (ICMLCN) [45].
- **2026** — physics-aware fractional-Doppler unfolding [39]; phase-noise-robust model-driven detection [38]; GNN-with-priors detection [37]; LAMP unfolding for MIMO-OTFS [42]; modular E2E superimposed-pilot learning posted 30 Jan 2026 [29]; Zak-OTFS theory + GPU real-time receivers [48]–[51]; agentic/foundation-model era begins [58], [59].

*(Post-audit note: [29] was previously listed under 2025; arXiv metadata confirms posting on 30 January 2026 — moved accordingly.)*

---

## 5. Comparative Table — Model-Driven / Hybrid OTFS Works

| Ref | Year/Venue | Module unfolded | Model backbone | Doppler assumption | Robust to CSI error | Impairments | Reports complexity |
|---|---|---|---|---|---|---|---|
| [36] Yue et al. | 2024, Comm. Letters | Detection | Iterative DD detector | On-grid | **Yes (explicit; trained against CE error)** | — | Partial (params/size) |
| [27] Zhang et al. | 2025, TCOM | CE + detection (hybrid, sequential per Fig. 12) | Plug-and-play (optimization + learned priors) | **Incl. fractional-Doppler sparsity loss (abstract, verified); Fig. 10(b) fractional BER (audit)** | **Sweep at fixed 12 dB SNR, Fig. 11(b) (audit)** | — | Partial |
| [20] Zhuang et al. | 2024, preprint | Detection (hybrid AMP+GNN, not unrolled) | AMP + GNN | **Fractional (§II; audit full-text check)** | **Perfect CSI + perturbed path gains (Fig. 2; audit)** | — | **Yes: FLOPs + measured runtime (Table I; audit)** |
| [39] Duan et al. | 2026, TVT | CE | Sparse-recovery unfolding | **Fractional** | unverified | — | Partial |
| [40] Duan et al. | 2026, WCL | CE | Sparse-recovery unfolding (FISTA) | On-grid | unverified | — | Partial |
| [41] WCNC'24 | 2024, WCNC | Sensing (target params) | Unfolding | On-grid | unverified | — | No |
| [42] ICSIPC'26 | 2026, conf. | CE (MIMO) | LAMP unfolding | On-grid | unverified | — | No |
| [37] ICT Express | 2026 | Detection | GNN + DD priors | On-grid | unverified | — | No |
| [38] CL 2026 | 2026, Comm. Letters | Detection | Model-driven iterative | On-grid | unverified | **Phase noise** | Partial |
| [29] Lei et al. | 2026, preprint (posted Jan 2026) | CE + detection (modular E2E) | Modular learned chain | **Fractional** | unverified | — | Partial |
| [71] Gao et al. | 2025, IoT-J | CE + detection (joint, end-to-end) | Lightweight DL + data augmentation | unverified | unverified | — | No |

*Cell policy (one evidence standard):* "— " = not reported in the checked source; "unverified" = claim not checkable at the level audited; bolded cells = verified (by our abstract-level check or the external audit's full-text check, as labeled). An em-dash is **not** a claim of absence. "State of the art" or priority language is used only where a direct comparison supports it (none here).

---

## 6. GAP ANALYSIS

Each gap is stated with its evidence level and — following the audit — with what was *confirmed in the checked sources* vs what remains *unverified*. G6, G7 and G14 are **not load-bearing** for this SISO project's motivation.

### G1 — The niche is young, but "small corpus ≠ small field" (counting-precision gap)
By the §2 *explicit-unfolding* counting rule, ~10 works qualify (Sec. 3.4). But the corpus came from ≈160 arXiv + ≈300 Crossref screened candidates with per-query dedup counts not logged (§2 caveat), and screening was metadata/abstract-level — the retained count bounds what we *found*, not what exists. **Implication:** before any scarcity claim in the report, re-run the searches with logged dedup counts and a full-text screening pass; state the counting rule explicitly.

### G2 — What exactly is *jointly trained* remains an open comparison (narrowed after audits)
No work in the checked sources performs *two explicitly unrolled iterative algorithms co-optimized through a differentiable channel model*. But the neighborhood is crowded and must be compared precisely:
- **[27] PnP (TCOM 2025):** sequential estimator→detector pipeline (Fig. 12, audit full-text check); learned priors inside an optimization skeleton; no joint backprop claimed in checked sources.
- **[71] Gao et al. (IoT-J 2025):** end-to-end joint estimation and detection (abstract-level; which losses reach which modules unverified).
- **[29] Lei et al. (2026):** modular physics-informed end-to-end learning over superimposed pilots (abstract-level; module/loss wiring unverified).
- **[8] He et al.:** joint CE/detection precedent exists in MIMO (OAMP-Net2 lineage) — the OTFS transfer is the open part.
**Comparison axes for the report:** which algorithm is unfolded · which losses reach which module · whether detection feeds back at inference. **The proposed training-time gradient (detection loss → CE parameters through a differentiable operator) is a specific hypothesis; its novelty remains open** pending the full-text sweep (spec §8).

### G3 — Fractional Doppler on the *detector* side (narrowed after audit round 4)
The classical literature shows fractional Doppler degrades on-grid detectors in the studied regimes [46], [47] — impact is configuration-dependent (pulse/CP/channel/grid/receiver). On the DL side: CE-side unfolding [39], hybrid AMP-GNN detection [20] (audit full-text check), and modular E2E [29] handle fractional Doppler. **What remains unverified in the checked sources:** an *explicitly unfolded detector* trained/validated under fractional Doppler with a jointly time-varying channel ([36]–[38] checked at abstract/audit level report on-grid or pre-equalized assumptions). **Implication:** the detector-side fractional-Doppler question is narrower than first stated but still open for the unfolding line.

### G4 — Robustness to CE error: separate *training-time* robustness from *evaluation-time* sweeps (narrowed)
| Work | How CSI error is generated | In training? | Error magnitude swept? |
|---|---|---|---|
| [36] Yue et al. | CE-error model (abstract-level) | **Yes (claimed)** | unverified |
| [20] Zhuang et al. | Perfect CSI + perturbed path gains (Fig. 2, audit) | Not stated in checked sources | unverified |
| [27] Zhang et al. | Normalized CSI-error sweep at fixed 12 dB SNR (Fig. 11(b), audit) | unverified | **Yes (audit)** |
**Implication:** Yue et al. [36] remains the only *unfolded* detector whose checked sources claim training against CE error; systematic training-vs-evaluation robustness studies (with controlled error generation) remain scarce — this project's CSI-error sweep design (spec §5) targets exactly that distinction.

### G5 — Generalization: separate SNR transfer, few-shot adaptation, and channel transfer (replaced field-wide claim)
| Transfer type | Evidence in checked sources |
|---|---|
| **SNR transfer** (train at fixed noise level, evaluate across SNR) | [20] (audit: trains at a specified noise level per modulation, reports BER across SNR) |
| **Few-shot adaptation** (one sample from a new environment) | [17] (abstract: one-sample adaptation; audit-checked) |
| **Channel-statistics / pulse / profile transfer** (train on profile A, zero-shot on profile B) | **Not identified in checked sources** — absence of evidence, not proof of absence |
**Implication:** a *systematic* generalization benchmark with declared train/test splits (Doppler band, path count, SNR extrapolation, profile variant) is unclaimed in the checked sources; this project's hold-out design (spec §5.5) fills that shape.

### G6 — MIMO-OTFS detection without explicit unfolding (qualified; not load-bearing for a SISO project)
DL exists for MIMO-OTFS *channel estimation* [25]–[28], [42] (confirmed at abstract level). MIMO-OTFS *detection* with explicit unfolding: no instance found in our corpus (unverified beyond it — absence of evidence). Non-DL MIMO-OTFS theory is strong [49]. **Implication:** open territory, but **not needed to motivate the SISO experiment**; listed for the report's future-work section.

### G7 — Unified model-driven OTFS-ISAC *receiver* (qualified; not load-bearing)
DL appears at the edges of OTFS-ISAC — pre-equalization [52], unfolded *target-parameter* estimation [41] (sensing-only inference), interference mitigation [43]. A single model-driven architecture jointly performing communication detection **and** radar parameter extraction: not found in checked sources (unverified beyond). **Implication:** strong 6G-flavored direction, heavier workload — future work, not project motivation.

### G8 — Complexity reporting is uneven; hardware deployment unreported (narrowed after audit round 4)
The hybrid line **does** report FLOPs and measured runtime ([20], Table I — audit full-text check). What the checked sources do not show: (i) equally rigorous FLOPs/latency reporting across the *unfolding* line ([36]–[42]: parameter counts at best, "unverified" elsewhere); (ii) hardware deployment on real devices; (iii) over-the-air validation. **Implication:** a rigorous complexity study (FLOPs conventions, measured latency incl. solves and trace terms, peak memory) still differentiates — but the gap must be stated per-line, not field-wide.

### G9 — No *shared benchmark* identified; public code exists (rewritten after audit round 4)
Public resources verified to exist (audit round 4, verified by us on GitHub): **textremo/Phy_Mod_OTFS** — an open OTFS toolbox (MATLAB, 68★) [S12]; **anafreis/OTFS_CE** — an LSTM-based OTFS channel-estimation *reproduction* repository (MATLAB, 57★) [S11]. Search scope of this audit: our corpus searches + the audit's targeted search; neither identified a **shared benchmark** — same waveform, frame structure, pilot pattern, CSI conditions and evaluation protocol — that multiple papers report on. Also distinct: downloadable *checkpoints* (none verified compatible for Yue et al. or UOR-Net), *data generators* (toolboxes exist), and *fixed benchmark splits* (none identified). **Implication:** releasing a standardized simulation/benchmark package (which doubles as the project codebase) is a contribution; claim it as "no shared benchmark identified in this audit", not "no public code exists".

### G10 — Impairments: coverage is per-receiver; build the table (rewritten after audit round 4)
| Receiver | Impairment(s) tested | Model-driven? | Training or evaluation? | Channel source |
|---|---|---|---|---|
| [38] CL 2026 | Phase noise (V2X) | **Yes (model-driven detector)** | unverified | unverified |
| [16] Naikoti & Chockalingam | IQ imbalance | Data-driven | unverified | unverified |
| [55] Men et al. | Synchronization/frame timing | Data-driven | unverified | unverified |
| [17] Cheng et al. | Environment change (one-sample adaptation) | Data-driven | **Adaptation (eval-time; abstract)** | unverified |
| [20] Zhuang et al. | Perturbed path gains | Hybrid | Evaluation (audit) | unverified |
| [27] Zhang et al. | CSI error (normalized sweep) | Hybrid | unverified | unverified |
No receiver in the checked sources is tested on *combined* impairments; whether any used measured channels or an over-the-air link is unverified (→ G11). **Implication:** an impairment-coverage matrix (individual + combined, with declared generation mechanisms) is unclaimed; a model-driven receiver trained with a composite impairment model is unclaimed. Note [38] falsifies any "never inside a model-driven architecture" phrasing.

### G11 — Sim-to-real: no measured-channel or OTA validation identified (softened)
All checked training setups are synthetic (simplified taps or clustered channels); no measured-channel feed or over-the-air validation of a DL-OTFS receiver was identified in the checked sources — **absence of evidence, not proof of absence**. A GPU timing run alone does not quantify a sim-to-real gap. **Implication:** an SDR/measured-channel probe is valuable future work; do not claim "first" without a targeted search.

### G12 — Guarantees: split algorithm-level proofs from learned-receiver guarantees (specified after audit)
Two distinct things: (i) **proofs for the underlying algorithms** — these exist for AMP/LAMP/OAMP lineages [8] and OAMP state evolution (a *modeling approximation* whose assumptions — e.g., right-orthogonal invariance — do not hold exactly for structured sparse DD channels); (ii) **convergence or error guarantees for the *learned* receiver** under off-grid Doppler and estimated CSI — none identified in the checked sources. **Implication:** beyond MCS scope to produce; the project treats OAMP state evolution as an approximation, makes variance parameters learnable to absorb the mismatch, and reports empirical sensitivity (spec §3.4).

### G13 — Pilots: three separate learning claims (tightened after audit)
Per the audit's full-text check of [29] §III-B: Lei et al. optimize the **complex pilot value** end to end **at a designated location**. Three distinct claims must not be conflated: (1) learning the pilot *value* — **verified in [29]**; (2) learning the pilot *location* — not identified in checked sources; (3) learning a *guard pattern* — not identified in checked sources. Fixed embedded pilots [2b] remain the norm in the unfolding line. **Implication:** pilot-value co-training with an unfolded receiver is the nearest-neighbor claim; location/guard-pattern learning is open (future work).

### G14 — Foundation-model / LLM era (demoted; not load-bearing)
LLM CSI prediction [56], agentic PHY design [58] and autonomous algorithm discovery [59] exist for OFDM/general PHY; nothing found for OTFS receivers (unverified beyond). **Implication:** removed from the project's motivating argument (it does not change the receiver experiment); retained as a one-line future-work observation.

### Post-audit corrections (12 Nov 2026) — round 1
- **G2 narrowed:** PnP [27] unifies CE + detection and covers fractional-Doppler sparsity loss (verified against arXiv:2503.11102); it is a *hybrid* (optimization + learned priors), not joint deep unfolding. Novelty for jointly-unfolded CE+detector must be established by systematic search, not assumed.
- **G5 softened** to absence-of-evidence phrasing; **[40] clarified** as a FISTA-based CE network; **[27] table cell corrected**; bibliography marked partially verified.

### Audit round 2 (12 Nov 2026)
- **Gao et al. IoT-J 2025 verified via Crossref and added as [71]** (comparative table row + baseline). Joint CE+DD alone is not a novelty axis.
- **G1 nuance:** the ≲10 count holds for *model-driven/unfolding* works under the §2 rule; [71] is data-driven and does not change it.

### Audit round 3 (12 Nov 2026) — implementation-focused
- **Simulation conventions corrected** (`src/otfs_demo.py` rewritten): ISFFT = FFT-over-delay ∘ IFFT-over-Doppler, ONE frame-level CP, measured-Es SNR, whole-frame LMMSE; noiseless error 2.22e-15 over 5 seeds.
- **Baselines re-scoped** primary/secondary; **reference-verification reminder** for 2026 early-access DOIs.

### Audit round 4 (12 Nov 2026) — evidence standard + full-text checks
- **[20] AMP-GNN row corrected** (was "On-grid"/"Partial" → fractional Doppler §II, FLOPs + measured runtime Table I — audit full-text check S3). G3 and G8 narrowed accordingly.
- **G4/G5 split into evidence tables** (CSI-error generation/training/swept; SNR transfer vs few-shot adaptation vs channel transfer; [17] = Cheng, Shi, Yuan & Zhou — authors verified via Crossref).
- **G9 rewritten:** public code exists (verified: textremo/Phy_Mod_OTFS toolbox; anafreis/OTFS_CE LSTM-CE reproduction); the narrowed claim is "no shared benchmark identified in this audit".
- **G10/G11 rewritten:** per-receiver impairment table (removes the falsified "never inside a model-driven architecture" — [38] is one); synthetic/measured/OTA recorded separately; GPU timing ≠ sim-to-real.
- **G12 split:** algorithm-level proofs vs learned-receiver guarantees under off-grid Doppler + estimated CSI.
- **G13 split:** pilot-value learning verified in [29] §III-B (audit); location and guard-pattern learning unverified.
- **G14 demoted** from the motivating argument.
- **Physical claims in §§1/3 qualified** (no universal constant-gain / dominant-error / inevitable-failure phrasing).
- **Timeline fixed:** [29] moved from 2025 to 2026 (arXiv posting 2026-01-30, verified).
- **Bibliography reconciled:** MD 73 numbered references ↔ BibTeX 73 entries ([1b] added to the bib); author lists completed from Crossref for [17], [29] (+Kun Yang), [45] (+Savazzi; upgraded to ICMLCN 2025 with DOI; = arXiv:2503.08234), [71] (+Jie Zhang, Shugong Xu). "All verified" header claim replaced with the partially-verified standard.
- **Table cell policy** (§5 footnote): "—", "unverified", and "verified" are now distinct labels; no implicit absence claims.

---

## 7. Actionable Directions for This Project (MCS scope)

Ranked by fit (novelty-candidate status × feasibility on a typical college GPU/laptop):

1. **Joint unfolded model-driven OTFS receiver with fractional-Doppler and CSI-error robustness** (targets G2 + G3 + G4; *novelty = open hypothesis*, see G2). Design: unfold a DD-domain detector (OAMP lineage) *and* a sparse-CE block into one trainable network; train under fractional Doppler and across CE-error levels; the decisive comparison is paired separate-vs-joint training of the same architecture (spec §5).
2. **Complexity-transparent benchmark** (targets G8 + G9): release the simulation + benchmark code with BER vs SNR, FLOPs (documented conventions), parameter counts, measured latency (incl. solves/trace), and training time for all baselines.
3. **Generalization study** (targets G5): declared train/test splits — Doppler offset band, path count, SNR extrapolation, generator variant; zero-shot results reported.
4. **Stretch (not project motivation):** MIMO-OTFS detection unfolding (G6), impairment-coverage matrix extension (G10), measured-channel probe (G11).

**Evaluation protocol:** fixed grid for the decisive experiment (32×8, audit round 4), QPSK, per-slot-CP continuous-Doppler time-domain simulator with declared phase convention, embedded pilot with full guards first, νN·T = k + κ parameterization with signed offsets κ ∈ (−½, ½], controlled CSI-error generation, matched SNR grids; metrics: uncoded BER on data positions (+ error/bit counts), effective-channel NMSE, FLOPs/frame, #params, measured latency, peak memory.

---

## 8. Conclusion (empirical form)

This review supports **questions**, not closure. The checked sources establish: (a) the explicit-unfolding line for OTFS is young (~10 works, 2024–2026, under the §2 counting rule); (b) joint CE+detection exists in hybrid ([27]) and end-to-end data-driven ([71]) forms, while *jointly trained explicit unfolding through a differentiable channel operator* remains unverified in the checked sources; (c) fractional Doppler is handled on the estimation side and by at least one hybrid detector, while an unfolded detector validated under fractional Doppler with estimated CSI is not documented there; (d) no shared benchmark was identified in this audit, though public code repositories exist. The experiment this project will run is therefore stated as the testable question it is: **does allowing detection loss to update the unfolded channel estimator improve data-bit BER relative to separate training of the same architecture, under matched fractional-Doppler channels and pilot-estimated CSI?** — reported with error/bit counts, CIs, and negative results included.

---

## References

*(Journal/conference entries include DOI; arXiv preprints include arXiv ID. Status: PARTIALLY VERIFIED — core entries DOI-checked via Crossref; the rest abstract/metadata level. Author lists for [17], [29], [45], [71] completed from Crossref on 2026-11-12 after audit round 4. BibTeX list reconciled 1:1 with this list (73 entries incl. [1b]).)*

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
[17] Q. Cheng, Z. Shi, J. Yuan, and M. Zhou, "Environment-robust Signal Detection for OTFS Systems Using Deep Learning," *IEEE GLOBECOM*, 2022. DOI: 10.1109/GLOBECOM48099.2022.10000940. *(Authors completed via Crossref, audit round 4.)*
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
[29] Y. Lei, Y. Liu, G. Liu, L. Wan, and K. Yang, "Superimposed-Pilot OTFS Under Fractional Doppler: Modular End-to-End Learning," arXiv:2601.22523, posted 30 Jan 2026. *(Author list completed via arXiv metadata, audit round 4.)*
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
[45] M. Marchese, H. Wymeersch, P. Spallaccini, S. Chinnici, and P. Savazzi, "Reduced-latency DL-based Fractional Channel Estimation in OTFS Receivers," *IEEE ICMLCN*, 2025. DOI: 10.1109/ICMLCN64995.2025.11140575; arXiv:2503.08234. *(Record completed via Crossref + arXiv metadata, audit round 4.)*
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
[71] Y. Gao, X. Xu, Y. Jin, W. Yuan, J. Zhang, and S. Xu, "Joint Channel Estimation and Data Detection for OTFS Systems: A Lightweight Deep Learning Framework With a Novel Data Augmentation Method," *IEEE Internet of Things Journal*, 2025. DOI: 10.1109/JIOT.2025.3586012. *(Author list completed via Crossref, audit round 4.)*

[S11] anafreis/OTFS_CE — LSTM-based OTFS channel-estimation reproduction repository (MATLAB), github.com/anafreis/OTFS_CE. *(Existence verified via GitHub API, audit round 4; checkpoint weights availability NOT verified.)*
[S12] textremo/Phy_Mod_OTFS — open OTFS toolbox (MATLAB), github.com/textremo/Phy_Mod_OTFS. *(Existence verified via GitHub API, audit round 4.)*

---

*Compiled from arXiv and Crossref records screened on 2026-11-12. Citation counts noted during collection (e.g., [1] ≈1871, [2] ≈1248, [3] ≈739, [6] ≈2751, [7] ≈474, [36] ≈20, [27] ≈17, [18] ≈24) indicate relative impact; exact counts fluctuate.*
