"""Input/output representation demo for the OTFS project (visualization only).

Reuses the verified Stage-1 chain (otfs_demo.py) on the decisive 32x8 grid and
renders how INPUTS and OUTPUTS are represented:
  fig1: QPSK constellation | DD resource map (pilot/guards/data) | time waveform
  fig2: true sparse channel taps | received DD grid (noiseless, energy spread)
  fig3: received DD grid (10 dB) | detector output constellation with errors
Saved to results/*.png. BER is counted on DATA positions only (guards/pilot excluded).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import otfs_demo as od

Path("results").mkdir(exist_ok=True)

od.M, od.N, od.CP = 32, 8, 8          # decisive grid for this demo

INK, DEEP, GREEN, SOFT = "#001E2B", "#00684A", "#00ED64", "#7FDDBE"
CARD, GOLD, RED, MUT = "#EAF7F0", "#FFC845", "#E5484D", "#43554D"
plt.rcParams.update({"font.family": "Segoe UI", "axes.edgecolor": MUT,
                     "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "figure.facecolor": "white", "axes.titlesize": 11})

M, N = od.M, od.N
rng = np.random.default_rng(42)
qpsk = od.qpsk_grid(rng)

# ---- schematic embedded pilot: impulse + full cross guard (exact size = Stage 3)
mask_data = np.ones((M, N), bool)
mask_data[1, :] = False; mask_data[-1, :] = False      # delay neighbours (circular)
mask_data[:, 1] = False; mask_data[:, -1] = False      # doppler neighbours
mask_data[0, 0] = False                                 # pilot cell
D = int(mask_data.sum())

X = np.where(mask_data, qpsk, 0.0 + 0.0j)
X[0, 0] = 1.0                                           # DD impulse pilot

h = rng.normal(size=4) + 1j * rng.normal(size=4); h /= np.linalg.norm(h)

x_tx = od.transmit(X)
y_clean = np.convolve(x_tx, h)[:x_tx.size]
y, sigma2 = od.add_noise(y_clean, 10.0, np.random.default_rng(99))
X_hat = od.receive_full(y, h, noise_var=sigma2)
X0_hat = od.receive_full(y_clean, h)                    # noiseless view for fig2

def bits(x):
    return x.real < 0, x.imag < 0

b0t, b1t = bits(X); b0r, b1r = bits(X_hat)
sym_err = ((b0t != b0r) | (b1t != b1r)) & mask_data
ber10 = 0.5 * (np.mean((b0t != b0r)[mask_data]) + np.mean((b1t != b1r)[mask_data]))

y20, s2_20 = od.add_noise(y_clean, 20.0, np.random.default_rng(100))
Xh20 = od.receive_full(y20, h, noise_var=s2_20)
c0, c1 = bits(Xh20)
ber20 = 0.5 * (np.mean((b0t != c0)[mask_data]) + np.mean((b1t != c1)[mask_data]))

# ================= FIG 1 — INPUT representations =================
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=160)
a = ax[0]
a.scatter(qpsk[mask_data].real, qpsk[mask_data].imag, s=14, c=SOFT, alpha=.55)
a.axhline(0, color=MUT, lw=.6); a.axvline(0, color=MUT, lw=.6)
a.set_title("QPSK data symbols\n(bits → complex points, 2 bits/symbol)")
a.set_xlabel("I (real)"); a.set_ylabel("Q (imag)"); a.set_aspect("equal")

img = np.full((M, N, 3), matplotlib.colors.to_rgb(CARD))
img[mask_data] = matplotlib.colors.to_rgb(SOFT)
img[0, 0] = matplotlib.colors.to_rgb(GOLD)
a = ax[1]
a.imshow(img, origin="lower", extent=[-0.5, N - 0.5, -0.5, M - 0.5], aspect="auto")
a.set_title(f"DD resource map (M={M} delay × N={N} Doppler)\n"
            f"pilot + guards + {D} data cells")
a.set_xlabel("Doppler bin k"); a.set_ylabel("Delay bin l")
a.legend(handles=[Patch(fc=SOFT, label="data (QPSK)"), Patch(fc=GOLD, label="pilot"),
                  Patch(fc=CARD, label="guard")], loc="upper right", fontsize=8)

a = ax[2]
a.plot(np.abs(x_tx), color=DEEP, lw=.9)
a.axvspan(0, od.CP, color=GREEN, alpha=.25, label="cyclic prefix")
a.set_title(f"Transmit waveform — what goes on air\n({x_tx.size} samples incl. CP)")
a.set_xlabel("time sample n"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/fig1_input.png", bbox_inches="tight")

# ================= FIG 2 — CHANNEL representations =================
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)
a = ax[0]
ml, sl, bl = a.stem(np.arange(4), np.abs(h)); plt.setp(ml, color=DEEP, markersize=8)
plt.setp(sl, color=DEEP, lw=2); plt.setp(bl, color=MUT, lw=.6)
a.set_title("True channel = sparse: 4 taps (dots)\nDoppler = 0 (Stage-1 LTI; fractional → Stage 2)")
a.set_xlabel("delay bin l"); a.set_ylabel("|gain|"); a.set_ylim(0, .8)

im = ax[1].imshow(np.abs(X0_hat).T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[-0.5, M - 0.5, -0.5, N - 0.5])
ax[1].set_title("Received DD grid (noiseless)\nevery symbol sees the whole channel")
ax[1].set_xlabel("delay l"); ax[1].set_ylabel("Doppler k")
fig.colorbar(im, ax=ax[1], shrink=.85)
fig.tight_layout(); fig.savefig("results/fig2_channel.png", bbox_inches="tight")

# ================= FIG 3 — OUTPUT representations =================
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
im = ax[0].imshow(np.abs(X_hat).T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[-0.5, M - 0.5, -0.5, N - 0.5])
ax[0].set_title(f"Received DD grid at SNR 10 dB (noisy)\nnoise var σ² = {sigma2:.2e}")
ax[0].set_xlabel("delay l"); ax[0].set_ylabel("Doppler k")
fig.colorbar(im, ax=ax[0], shrink=.85)

a = ax[1]
ok = mask_data & ~sym_err
a.scatter(X[ok].real, X[ok].imag, s=14, c=SOFT, alpha=.5, label=f"correct ({ok.sum()} symbols)")
if sym_err.any():
    a.scatter(X_hat[sym_err].real, X_hat[sym_err].imag, s=90, c="none", marker="x",
              linewidths=2.2, edgecolors=RED, label=f"errors ({sym_err.sum()} symbols)")
a.axhline(0, color=MUT, lw=.6); a.axvline(0, color=MUT, lw=.6)
a.set_aspect("equal")
a.set_title(f"Detector output: hard bits from LLRs\nBER(data bits) = {ber10:.4f} @ 10 dB")
a.set_xlabel("I"); a.set_ylabel("Q"); a.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("results/fig3_output.png", bbox_inches="tight")

print("---- I/O representation summary (32x8 grid) ----")
print(f"INPUT : bits 2x{D}={2*D}/frame -> QPSK grid X (M={M}, N={N}) complex64-ish"
      f" -> time frame {x_tx.size} samples (incl. CP={od.CP})")
print(f"        + channel h ({h.size} taps, unit energy) + AWGN at chosen SNR")
print(f"OUTPUT: X_hat (M={M}, N={N}) -> LLRs -> hard bits on {D} data cells")
print(f"        BER @10 dB = {ber10:.4f}   |   BER @20 dB = {ber20:.4f}")
print(f"        spectral efficiency ~ {2*D / x_tx.size:.2f} bits/sample (schematic guards)")
print("saved: results/fig1_input.png, results/fig2_channel.png, results/fig3_output.png")
