# Model-Driven Deep Learning for OTFS Modulation

MCS project — literature review, gap analysis, and simulation environment for
model-driven deep learning (deep unfolding) applied to OTFS receiver design.

## Repository layout

```
PROJECT/
├── literature/
│   ├── literature_review.md   <- full review + 14-gap analysis (~70 references)
│   └── references.bib         <- BibTeX for the LaTeX report
├── src/                       <- simulation / model code
│   └── otfs_demo.py           <- minimal OTFS transceiver sanity check (numpy only)
├── data/                      <- simulation outputs, datasets
├── docs/                      <- report drafts, figures
└── README.md
```

## Environment setup (Windows, Python 3.12)

```powershell
# one-time
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# optional, for neural network experiments (CPU wheel, ~200 MB):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (for GPU/CUDA training use the plain `pip install torch` instead)
```

## Sanity check

```powershell
.\venv\Scripts\Activate.ps1
python src\otfs_demo.py
```

Expected output: `OTFS roundtrip OK: ... max symbol error ≈ 1e-16` — verifies the
numpy/scipy stack and the OTFS modulation/demodulation + multipath channel chain
used as the base for all later experiments.

## Where to start reading (top 5)

1. Yue et al., *Model-Driven DL Assisted Detector for OTFS With Channel Estimation Error*, IEEE Comm. Letters 2024 — the anchor paper for this exact topic (DOI 10.1109/LCOMM.2024.3362970).
2. Zhang et al., *Deep Learning-based OTFS CE and Symbol Detection with Plug-and-Play*, IEEE TCOM 2025 — state-of-the-art hybrid receiver (arXiv:2503.11102).
3. Wei et al., *OTFS: A Promising Next-Generation Waveform*, IEEE Wireless Commun. 2021 — the field tutorial incl. OAMP/TMP detectors.
4. Duan et al., *Physics-Aware Guided Deep Unfolding for OTFS CE Under Fractional Doppler*, IEEE TVT 2026 — the only unfolding work attacking fractional Doppler.
5. O'Shea & Hoydis, *An Introduction to Deep Learning for the Physical Layer*, IEEE TCCN 2017 — foundational context.

Full annotated list + gap analysis (G1–G14): `literature/literature_review.md` §6.

## Suggested next steps

1. Reimplement the baseline chain in `src/otfs_demo.py` with fractional Doppler and a DD-domain ML/MP detector.
2. Port to PyTorch; build the unfolded detector (layer = one iteration, learnable step/damping).
3. Follow the evaluation protocol in `literature_review.md` §7 (BER vs SNR, FLOPs, #params, generalization splits).
