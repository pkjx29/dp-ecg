"""Hand-crafted ECG features from a 10 s, 12-lead, 100 Hz recording.

Pipeline for one record (signal shape (1000, 12), units mV):

1. Band-pass 0.5-40 Hz (4th-order Butterworth, zero-phase sosfiltfilt).
   0.5 Hz removes baseline wander from breathing; 40 Hz removes mains
   interference and EMG while keeping the QRS energy (mostly < 30 Hz).
   Zero-phase filtering matters: a causal filter would shift wave timings.
2. R-peak detection on a lead-independent "energy" signal: sum over leads of
   the squared first difference, smoothed with a 100 ms moving average (the
   core idea of Pan-Tompkins, but pooled across 12 leads so a single
   low-amplitude or inverted lead cannot hide the beats).
3. Rhythm features from the RR intervals (heart rate, SDNN, RMSSD, CV).
4. A median beat per lead (window -250..+450 ms around each R peak; the
   median over beats is robust to an occasional ectopic beat), from which
   amplitudes of R, S, ST level and T wave are measured relative to the
   PR-segment baseline.
5. Global features: QRS width, frontal-plane electrical axis, the
   Sokolow-Lyon voltage (an LVH criterion), Welch band powers and spectral
   entropy of lead II, plus age and sex from the metadata.

FEATURE_BOUNDS gives a fixed plausible physiological range for every feature.
The bounds are chosen from physiology, NOT from the data, because the DP
standardiser (mechanisms.dp_standardizer) needs data-independent clipping
bounds for its sensitivity to hold.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from .metadata import FS, LEADS

_SOS = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")

PRE, POST = int(0.25 * FS), int(0.45 * FS)  # median-beat window, samples
R = PRE                                     # index of the R peak in a beat
BASE = slice(R - 10, R - 4)                 # PR segment: -100..-40 ms
QRS = slice(R - 6, R + 7)                   # -60..+60 ms
ST_AT = R + 8                               # +80 ms, roughly J point + 40 ms
T_WIN = slice(R + 15, R + 41)               # +150..+400 ms

PER_LEAD = ["r_amp", "s_amp", "qrs_net", "st", "t_amp"]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "hr_bpm": (20.0, 250.0),
    "sdnn_s": (0.0, 0.5),
    "rmssd_s": (0.0, 0.5),
    "rr_cv": (0.0, 1.0),
    "n_beats": (0.0, 40.0),
    **{f"{f}_{lead}": b for lead in LEADS for f, b in zip(
        PER_LEAD, [(0.0, 4.0), (-4.0, 0.0), (-4.0, 4.0), (-1.0, 1.0), (-2.0, 2.0)])},
    "qrs_width_s": (0.02, 0.24),
    "axis_cos": (-1.0, 1.0),
    "axis_sin": (-1.0, 1.0),
    "sokolow_mv": (0.0, 8.0),
    "bp_low": (0.0, 1.0),
    "bp_mid": (0.0, 1.0),
    "bp_high": (0.0, 1.0),
    "spec_entropy": (0.0, 1.0),
    "age": (0.0, 100.0),
    "sex": (0.0, 1.0),
}
FEATURE_NAMES = list(FEATURE_BOUNDS)


def bandpass(sig: np.ndarray) -> np.ndarray:
    return sosfiltfilt(_SOS, sig, axis=0)


def detect_r_peaks(sig: np.ndarray) -> np.ndarray:
    energy = (np.diff(sig, axis=0) ** 2).sum(axis=1)
    win = int(0.1 * FS)
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(energy, distance=int(0.25 * FS),
                          height=0.3 * np.percentile(energy, 99))
    # The energy peak sits on the steepest QRS slope; snap to the largest
    # absolute deflection within +-50 ms so all beats align on the R wave.
    ref = np.abs(sig).sum(axis=1)
    snapped = [max(p - 5, 0) + int(np.argmax(ref[max(p - 5, 0):p + 6])) for p in peaks]
    return np.unique(np.array(snapped, dtype=int))


def median_beat(sig: np.ndarray, peaks: np.ndarray) -> np.ndarray | None:
    ok = peaks[(peaks >= PRE) & (peaks + POST <= len(sig))]
    if len(ok) == 0:
        return None
    beats = np.stack([sig[p - PRE:p + POST] for p in ok])  # (beats, 70, 12)
    return np.median(beats, axis=0)


def extract(sig: np.ndarray, age: float, sex: float) -> np.ndarray:
    """Return the feature vector (ordered like FEATURE_NAMES); NaN if a
    feature could not be measured (e.g. fewer than two beats found)."""
    x = bandpass(np.nan_to_num(sig))
    peaks = detect_r_peaks(x)
    f: dict[str, float] = {}

    rr = np.diff(peaks) / FS
    if len(rr) >= 2:
        f["hr_bpm"] = 60.0 / rr.mean()
        f["sdnn_s"] = rr.std()
        f["rmssd_s"] = np.sqrt(np.mean(np.diff(rr) ** 2))
        f["rr_cv"] = rr.std() / rr.mean()
    f["n_beats"] = float(len(peaks))

    beat = median_beat(x, peaks)
    if beat is not None:
        beat = beat - beat[BASE].mean(axis=0)  # PR-segment baseline
        qrs = beat[QRS]
        r_amp, s_amp = qrs.max(axis=0), qrs.min(axis=0)
        t_seg = beat[T_WIN]
        t_amp = t_seg[np.abs(t_seg).argmax(axis=0), np.arange(12)]
        for i, lead in enumerate(LEADS):
            f[f"r_amp_{lead}"] = r_amp[i]
            f[f"s_amp_{lead}"] = s_amp[i]
            f[f"qrs_net_{lead}"] = r_amp[i] + s_amp[i]
            f[f"st_{lead}"] = beat[ST_AT, i]
            f[f"t_amp_{lead}"] = t_amp[i]

        # QRS width: span around R where the pooled slope exceeds 20% of max.
        slope = np.abs(np.diff(beat, axis=0)).sum(axis=1)
        above = np.flatnonzero(slope[R - 12:R + 12] > 0.2 * slope[R - 12:R + 12].max())
        f["qrs_width_s"] = (above[-1] - above[0] + 1) / FS

        # Frontal axis from net QRS in lead I (0 deg) and aVF (+90 deg).
        # Encoded as cos/sin so -179 and +179 deg are neighbours.
        ang = np.arctan2(f["qrs_net_aVF"], f["qrs_net_I"])
        f["axis_cos"], f["axis_sin"] = np.cos(ang), np.sin(ang)

        f["sokolow_mv"] = -s_amp[LEADS.index("V1")] + max(r_amp[LEADS.index("V5")],
                                                        r_amp[LEADS.index("V6")])

    freqs, pxx = welch(x[:, LEADS.index("II")], fs=FS, nperseg=256)
    band = (freqs >= 0.5) & (freqs <= 40)
    total = pxx[band].sum()
    if total > 0:
        f["bp_low"] = pxx[(freqs >= 0.5) & (freqs < 4)].sum() / total
        f["bp_mid"] = pxx[(freqs >= 4) & (freqs < 15)].sum() / total
        f["bp_high"] = pxx[(freqs >= 15) & (freqs <= 40)].sum() / total
        p = pxx[band] / total
        f["spec_entropy"] = -(p * np.log(p + 1e-12)).sum() / np.log(band.sum())

    f["age"] = min(age, 90.0)  # PTB-XL codes ages > 89 as 300 for anonymity
    f["sex"] = sex
    return np.array([f.get(n, np.nan) for n in FEATURE_NAMES], dtype=float)


def bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([FEATURE_BOUNDS[n][0] for n in FEATURE_NAMES])
    hi = np.array([FEATURE_BOUNDS[n][1] for n in FEATURE_NAMES])
    return lo, hi


def clip_and_impute(X: np.ndarray) -> np.ndarray:
    """Clip to the fixed bounds and replace NaN by the bounds' midpoint.
    Both steps use only the public bounds, never statistics of the data,
    so they are per-record transformations that cost no privacy."""
    lo, hi = bounds_arrays()
    X = np.where(np.isnan(X), (lo + hi) / 2.0, X)
    return np.clip(X, lo, hi)
