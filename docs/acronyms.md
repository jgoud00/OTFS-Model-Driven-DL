# Acronyms & Symbols — Glossary

**Project:** Model-Driven Deep Learning for OTFS Modulation · MCS, KLH · 13 Nov 2026
Keep this as an appendix in the report. First use of each term in the report should also be expanded inline.

---

## 1. Waveform & channel

| Term | Full form | Meaning (one line) |
|---|---|---|
| OTFS | Orthogonal Time Frequency Space | Modulation that places data in the delay–Doppler domain; robust in high mobility |
| OFDM | Orthogonal Frequency Division Multiplexing | The 4G/5G waveform; degrades under Doppler (loses subcarrier orthogonality) |
| DD | Delay–Doppler | The 2D domain where OTFS data lives: horizontal = delay, vertical = Doppler |
| ISFFT | Inverse Symplectic Finite Fourier Transform | Transform from delay–Doppler grid → time–frequency grid (transmitter side) |
| SFFT | Symplectic Finite Fourier Transform | Inverse of ISFFT (receiver side) |
| CP | Cyclic Prefix | Copied guard samples that protect against multipath overlap |
| ISI / ICI / IDI | Inter-Symbol / Inter-Carrier / Inter-Doppler Interference | The three interference types in OTFS; rectangular pulses cause ISI+ICI |
| LTV / LTI | Linear Time-Varying / Time-Invariant | Channel that changes within the frame (LTV) vs constant (LTI) |
| AWGN | Additive White Gaussian Noise | The standard background-noise model |
| QPSK | Quadrature Phase Shift Keying | 4 constellation points, 2 bits per symbol (our default modulation) |
| 16-QAM | 16-Quadrature Amplitude Modulation | 16 points, 4 bits per symbol (deferred to later stages) |
| SISO / MIMO | Single / Multiple-Input Multiple-Output | One antenna pair vs several (this project: SISO only) |
| Zak-OTFS | — | Reformulation of OTFS via the Zak transform (2024–26 theory revival) |

## 2. Receiver & estimation algorithms

| Term | Full form | Meaning |
|---|---|---|
| CSI | Channel State Information | Knowledge of the channel at the receiver |
| CE | Channel Estimation | The task of learning the channel before detection |
| MMSE / LMMSE | (Linear) Minimum Mean Squared Error | The optimal linear filter under Gaussian assumptions |
| ZF | Zero Forcing | Simple equalizer that inverts the channel exactly; noise-amplifying |
| LS | Least Squares | Fit that minimizes squared error |
| NMSE | Normalized Mean Squared Error | Error energy ÷ true energy (in dB); our headline CE metric |
| MP | Message Passing | Iterative classical OTFS detector exploiting DD sparsity |
| AMP | Approximate Message Passing | Simpler Gaussian-approximated variant of MP |
| OAMP | Orthogonal AMP | AMP with a decorrelated linear step + divergence-free denoiser |
| TMP | Transform-Domain Message Passing | MP running in a transformed domain for speed |
| VB / EP | Variational Bayes / Expectation Propagation | Other Bayesian iterative detector families |
| SBL | Sparse Bayesian Learning | Bayesian way to recover sparse channels |
| GMM | Gaussian Mixture Model | Multi-component probability model (used as prior) |
| ISTA | Iterative Shrinkage-Thresholding Algorithm | Classic sparse-recovery loop — our Block A backbone |
| FISTA | Fast ISTA | ISTA + momentum (faster convergence) |
| LAMP | Learned AMP | AMP unrolled into a network |
| PnP | Plug-and-Play | Framework: plug learned denoisers into an iterative algorithm |
| ICID | Interference Cancellation & Iterative Detection | Raviteja et al.'s 2018 OTFS detector (our baseline reference) |
| SE | State Evolution | OAMP's theoretical performance-tracking approximation |

## 3. Learning & networks

| Term | Full form | Meaning |
|---|---|---|
| DL / ML | Deep Learning / Machine Learning | — |
| DNN / CNN / GNN / RNN | Deep / Convolutional / Graph / Recurrent Neural Network | The four data-driven architectures in the survey |
| LSTM | Long Short-Term Memory | A recurrent net (the GitHub CE repo uses one) |
| RC | Reservoir Computing | A lightweight recurrent alternative (2D-RC detector) |
| GAN | Generative Adversarial Network | Two-network generative setup (used by some estimators) |
| E2E | End-to-End | Training straight from input to output |
| BCE | Binary Cross-Entropy | The loss on bits (our detection loss) |
| LLR | Log-Likelihood Ratio | log P(bit=0)/P(bit=1) — soft output of the detector |
| STE | Straight-Through Estimator | Gradient trick for non-differentiable selection (removed in v2.3) |
| FLOPs | Floating-Point Operations | Complexity unit for compute cost |

## 4. System, metrics, evaluation

| Term | Full form | Meaning |
|---|---|---|
| BER | Bit Error Rate | Fraction of data bits decoded wrong — the headline metric |
| SNR | Signal-to-Noise Ratio | Signal power ÷ noise power, in dB |
| MSE | Mean Squared Error | Average squared deviation |
| 3GPP | 3rd Generation Partnership Project | Standards body (its channel profiles = TDL models) |
| GPU | Graphics Processing Unit | Training/timing hardware |
| SDR | Software-Defined Radio | Reconfigurable radio used for real-time prototypes |
| OTA | Over-the-Air | Tested on a real wireless link (vs simulation) |
| PAPR | Peak-to-Average Power Ratio | Waveform power-peaking property |
| IQI | In-phase/Quadrature Imbalance | Hardware impairment in mixers |
| DOI | Digital Object Identifier | Permanent paper link (10.1109/…) |

## 5. Applications / domains

| Term | Full form | Meaning |
|---|---|---|
| 5G / 6G | Fifth / Sixth Generation | Mobile-network generations |
| NTN | Non-Terrestrial Networks | Satellite / airborne links (6G focus) |
| LEO | Low Earth Orbit | Satellite class (Starlink-type orbits) |
| ISAC | Integrated Sensing And Communication | One waveform does radar + communications |
| V2X | Vehicle-to-Everything | Car connectivity |
| UAV | Unmanned Aerial Vehicle | Drone |

## 6. Project-specific terms

| Term | Meaning |
|---|---|
| UOR-Net | **U**nified **U**nfolded **O**TFS **R**eceiver **Net**work — our proposed receiver |
| Block A | Unfolded ISTA channel estimator (K layers, learnable γ_k, λ_k) |
| Block B | Differentiable DD channel operator (no learning; weighted atoms) |
| Block C | Unfolded OAMP detector (T layers, learnable variances) |
| Ĥ (H-hat) | The estimated channel operator = Σ ĥ_j · H_j (weighted sum of dictionary atoms) |
| H_j | One dictionary atom: the channel operator for an integer delay + one candidate Doppler |
| ĥ_j | Estimated weight (complex gain) of atom j |
| G = ĤS_d | The detector's channel: estimated operator restricted to data positions |
| S_d, S_p | Selector matrices: S_d inserts data symbols; S_p selects pilot observations |
| y′ = y − Ĥx_p | Received frame with the known pilot contribution removed — detector input |
| W̃_t, W_t | OAMP linear filter at layer t (raw, then trace-normalized) |
| v_t², σ_t² | Residual variance (before denoiser) / denoiser output noise variance (per complex symbol) |
| γ_k | Step size of ISTA layer k (learnable via a sigmoid; initialized to the classical 0.5/L) |
| λ_k | Soft-threshold of ISTA layer k (learnable, kept ≥ 0 via softplus) |
| L | Lipschitz constant (spectral norm) of AᴴA — caps the safe step size |
| S_λ(z) | Soft-threshold function: shrinks small values to exactly 0 |
| L_CE | Channel-estimation loss: operator-space error on probe vectors |
| L_det | Detection loss: BCE on data bits |
| k + κ | Doppler position: integer bin k plus fractional offset κ ∈ (−½, ½] |
| M × N | OTFS grid size: M delay bins × N Doppler bins (decisive grid: 32×8) |
| η | Dictionary oversampling factor (how many candidate Dopplers per bin) |
| D | Number of unknown data symbols in a frame |

## 7. Venue shortcuts (used in references)

| Short | Full name |
|---|---|
| TWC / TSP / TCOM / TVT | IEEE Transactions on Wireless Commun. / Signal Process. / Communications / Vehicular Technology |
| WCL / CL | IEEE Wireless Communications Letters / Communications Letters |
| IoT-J | IEEE Internet of Things Journal |
| WCNC / VTC / GLOBECOM / ICC / ICASSP | IEEE flagship conferences (wireless / vehicular / global comms / int. comms / signal processing) |
| ICMLCN | Int. Conf. on Machine Learning for Communication and Networking |
| IMS | IEEE International Microwave Symposium |
| arXiv | Preprint server (papers posted before journal acceptance) |
