"""Stage-1 CP-OTFS transceiver sanity check (numpy, float64).

Conventions (Raviteja et al., IEEE TWC 2018 discrete model; Doppler-free
special case here — fractional Doppler is Stage 2, NOT validated by this file):
- Data grid X[l, k]: axis0 = delay l (M points), axis1 = Doppler k (N points).
- ISFFT:  S[m,n] = (1/sqrt(MN)) * sum_{l,k} X[l,k] * exp(+j2pi(k n/N - l m/M))
        = sqrt(N/M) * fft_delay( ifft_doppler(X) )
          (numpy fft = e^{-j2pi...}, numpy ifft = (1/N) e^{+j2pi...})
- Heisenberg (rectangular pulse): per time slot, UNITARY IFFT over subcarriers
  (sqrt(M) * ifft) — unitary normalization per audit round 4; the Wigner
  transform at the receiver is its exact inverse (fft / sqrt(M)), so the
  DD -> time -> DD chain has unit net gain and mean time-sample energy 1.0
  before the channel (was 1/M under the 1/M-normalized Heisenberg).
- ONE cyclic prefix per FRAME (length CP >= max channel delay), matching the
  reduced-CP CP-OTFS convention — not a per-slot CP.
- LTI multipath channel (Doppler = 0): y = h * x + w, h normalized to unit energy.
- Tx power measured per frame: sigma^2 = Es * 10^(-SNR/10), Es = E|y|^2 pre-noise.
- Receiver: frame-CP removal -> whole-frame FFT LMMSE equalization
  (weight conj(H)/(|H|^2 + noise_var); noise_var=0 gives exact ZF) ->
  per-slot FFT (Wigner) -> SFFT.

Checks (audit round 3 — dtype-specific tolerances):
  T1  ISFFT->SFFT transform identity, no channel: < 1e-12 (float64)
  T2  full noiseless chain, 5 seeds, 4-tap channel: < 1e-8 (float64).
      Noiseless ZF is exact for any h_f != 0, so near-null bins do NOT set an
      error floor here (they matter only once noise is added); an exact grid
      zero is measure-zero — reseed if ever hit.
  T3  noisy runs at SNR 10/20 dB with true-variance LMMSE: prints QPSK BER
      (no assert — BER statistics belong to the Stage-4 protocol)

Run:  python src/otfs_demo.py
"""
import numpy as np

M, N = 64, 16      # subcarriers (delay axis) x time slots (Doppler axis)
CP = 8             # frame-level cyclic prefix (>= max delay)
L = 4              # channel taps (delays 0..L-1)
SEEDS = (1, 2, 3, 4, 5)


def qpsk_grid(rng):
    bits = rng.integers(0, 2, (2, M, N))
    return ((2 * bits[0] - 1) + 1j * (2 * bits[1] - 1)) / np.sqrt(2)


def isfft(X):
    """DD -> TF, Raviteja sign convention (Doppler +, delay -)."""
    return np.sqrt(N / M) * np.fft.fft(np.fft.ifft(X, axis=1), axis=0)


def sfft(Y):
    """TF -> DD (inverse of isfft)."""
    return np.sqrt(M / N) * np.fft.fft(np.fft.ifft(Y, axis=0), axis=1)


# ---- T1: pure transform identity ----
X0 = qpsk_grid(np.random.default_rng(0))
err_t1 = np.max(np.abs(sfft(isfft(X0)) - X0))
assert err_t1 < 1e-12, f"T1 transform identity failed: {err_t1}"


def transmit(X):
    s_tf = isfft(X)
    x_time = np.sqrt(M) * np.fft.ifft(s_tf, axis=0)   # unitary per-slot Heisenberg
    frame = x_time.T.reshape(-1)              # slot-serial stream, length M*N
    return np.r_[frame[-CP:], frame]          # ONE frame-level CP (our declared
                                              # LTI-smoke choice; ICID itself
                                              # assumes no CP — see Stage 2)


def add_noise(y_clean, snr_db, rng):
    """Es measured post-channel; returns (noisy y, true noise variance)."""
    sigma2 = np.mean(np.abs(y_clean) ** 2) * 10 ** (-snr_db / 10)
    n = rng.normal(size=y_clean.size) + 1j * rng.normal(size=y_clean.size)
    return y_clean + np.sqrt(sigma2) * n / np.sqrt(2), sigma2


def receive_full(y, h, noise_var=0.0):
    y_frame = y[CP:]                          # remove frame CP -> circular block
    h_f = np.fft.fft(h, y_frame.size)         # circular-channel freq response
    y_f = np.fft.fft(y_frame)
    # LMMSE: noise_var=0 -> exact ZF (noiseless test); = sigma^2 -> true MMSE
    y_eq = y_f * np.conj(h_f) / (np.abs(h_f) ** 2 + noise_var)
    x_eq = np.fft.ifft(y_eq)                  # ~ frame, slot-serial
    slots = x_eq.reshape(N, M)
    y_wigner = np.fft.fft(slots, axis=1).T / np.sqrt(M)  # unitary Wigner (inverse)
    return sfft(y_wigner)


def ber_qpsk(x_true, x_hat):
    b0_t, b1_t = x_true.real < 0, x_true.imag < 0
    b0_r, b1_r = x_hat.real < 0, x_hat.imag < 0
    return 0.5 * (np.mean(b0_t != b0_r) + np.mean(b1_t != b1_r))


# ---- T2: noiseless chain, several seeds ----
rng = np.random.default_rng(7)
h = rng.normal(size=L) + 1j * rng.normal(size=L)
h /= np.linalg.norm(h)

worst = 0.0
for seed in SEEDS:
    X = qpsk_grid(np.random.default_rng(seed))
    x_tx = transmit(X)
    y = np.convolve(x_tx, h)[:x_tx.size]      # noiseless
    X_hat = receive_full(y, h)
    err = np.max(np.abs(X_hat - X))
    worst = max(worst, err)
assert worst < 1e-8, f"T2 noiseless chain failed: worst {worst}"
print("T1 transform identity OK (<1e-12)")
print(f"T2 noiseless chain OK: 5 seeds, worst max-symbol error = {worst:.2e} (<1e-8)")

# ---- T3: noisy runs (report only) ----
for snr_db in (10.0, 20.0):
    X = qpsk_grid(np.random.default_rng(11))
    x_tx = transmit(X)
    y_clean = np.convolve(x_tx, h)[:x_tx.size]
    y, sigma2 = add_noise(y_clean, snr_db, np.random.default_rng(99))
    X_hat = receive_full(y, h, noise_var=sigma2)
    print(f"T3 SNR={snr_db:>4.0f} dB: QPSK BER = {ber_qpsk(X, X_hat):.4f} "
          f"(sanity print; proper BER curves come in Stage 4)")
