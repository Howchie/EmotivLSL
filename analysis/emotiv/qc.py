"""Acquisition and signal-quality checks, as plain functions returning dicts.

Nothing here plots or writes files; call what you want and put the result in your
own report.  ``integrity`` is the one worth running on every new recording: it is
what catches lost samples, decoder drift and a clock that is not linear, all of
which invalidate timing before any ERP is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, welch

from .devices import FS
from .events import find_eye_intervals
from .loading import Run
from .preprocess import contiguous, flex_counts, flex_deltas, good_mask, rail_fraction

__all__ = ["integrity", "quality_vs_noise", "alpha_spectra", "alpha_scores",
           "cortex_band_power", "pair_stats"]

EYE_TRIM = (3.0, 2.0)  # s dropped at the start/end of each self-timed eye interval


def integrity(run: Run, raw=None) -> dict:
    """Sample loss, decoder consistency and clock linearity, whichever device it is."""

    return _epocx_integrity(run, raw) if run.device == "epocx" else _flex_integrity(run)


def _epocx_integrity(run: Run, raw=None) -> dict:
    extra = run.extra
    res = extra["arrival_residual_ms"]
    out = {
        "device": "epocx",
        "samples_received": int(len(res)),
        "grid_samples": int(len(run.t)),
        "missing_samples": int(run.filled.sum()),
        "duration_s": run.duration_s,
        "dropped_repeats": extra["dropped_repeats"],
        "reader_missing_reports": extra["reader_missing_reports"],
        "reader_resets": extra["reader_resets"],
        "diag_rows_unmatched": extra["diag_rows_unmatched"],
        "headset_rate_hz": extra["segment_rates_hz"],
        "arrival_residual_ms_percentiles": dict(zip(
            ["min", "p1", "p50", "p99", "max"],
            np.percentile(res, [0, 1, 50, 99, 100]).round(2).tolist())),
        "decoder_max_fraction": extra["decoder_max_fraction"],
        "raw_count_range": extra["raw_count_range"],
        "rail_samples": extra["rail_samples"],
    }
    if raw is not None:
        x = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), raw.get_data() * 1e6, axis=1)
        n = int(FS)
        windows = x[:, : (x.shape[1] // n) * n].reshape(x.shape[0], -1, n)
        excursion = 100 * (np.ptp(windows, axis=2) > 400).mean(axis=1)
        out["pct_1s_windows_ptp_over_400uv"] = dict(zip(run.labels, excursion.round(2).tolist()))
    return out


def _flex_integrity(run: Run) -> dict:
    d = run.diag
    filled = run.filled
    starts, stops = contiguous(filled)
    lengths = stops - starts
    reset = d["RESET_FLAG"].astype(bool)

    # Decoder consistency: undoing the leak must give whole-count deltas within the seven-bit range.
    counts = flex_counts(run)
    deltas, real = flex_deltas(run)
    frac = np.abs(deltas[real] - np.round(deltas[real]))
    whole = np.round(deltas[real]).astype(int)
    rails = rail_fraction(run)
    # A lost 8-sample block drops 8 deltas, leaving a step equal to the signal's change over 8 samples.
    step8 = counts[8:] - run.leak ** 8 * counts[:-8]
    clean8 = np.convolve(filled.astype(int), np.ones(9, int), mode="valid") == 0
    step8_rms_uv = run.dev.lsb_uv * np.sqrt(np.mean(step8[clean8] ** 2, axis=0))

    # Timestamp linearity: robust line through each stretch between counter resets.
    edges = [0, *np.flatnonzero(reset).tolist(), len(run.t)]
    residual_ms = np.full(len(run.t), np.nan)
    segments = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1000:
            continue
        idx = np.arange(a, b)
        keep = np.ones(len(idx), bool)
        for _ in range(5):
            fit = np.polyfit(idx[keep], run.t[a:b][keep], 1)
            res = run.t[a:b] - np.polyval(fit, idx)
            keep = np.abs(res - np.median(res[keep])) < 0.005
        residual_ms[a:b] = 1000 * res
        segments.append({"start_s": float(run.t[a]), "end_s": float(run.t[b - 1]),
                         "rate_hz": float(1 / fit[0])})

    run_lengths, run_counts = np.unique(lengths, return_counts=True)
    return {
        "device": "flex",
        "samples": int(len(run.t)),
        "duration_s": run.duration_s,
        "filled_samples": int(filled.sum()),
        "filled_pct": float(100 * filled.mean()),
        "filled_runs": int(len(starts)),
        "filled_run_lengths": {int(k): int(v) for k, v in zip(run_lengths, run_counts)},
        "longest_fill_s": float(lengths.max() / FS) if len(lengths) else 0.0,
        "dropped_repeats": int(((d["GAP_FLAG"] > 0) & (d["MISSING_REPORTS"] == 0)).sum()),
        "counter_resets": int(reset.sum()),
        "counter_steps_not_one": int((np.diff(d["COUNTER"].astype(int)) % 128 != 1).sum()),
        "decoder_max_fraction": float(frac.max()),
        "decoder_delta_range": [int(whole.min()), int(whole.max())],
        "slew_saturated_pct": float(rails.mean()),
        "fill8_step_rms_uv_median_channel": float(np.median(step8_rms_uv)),
        # The per-channel dead-electrode flag; ``Processing.rail_pct`` is the cut.
        "slew_saturated_pct_by_channel": {ch: float(v) for ch, v in rails.items()},
        "timestamp_segments": segments,
        "_residual_ms": residual_ms,
    }


def quality_vs_noise(run: Run, raw) -> dict | None:
    """Do Cortex's per-channel quality numbers track the measured per-channel noise?"""

    if run.eq is None or run.cq is None:
        return None
    x = raw.get_data() * 1e6
    x = x - x.mean(axis=0, keepdims=True)
    hf = sosfiltfilt(butter(4, [1, 40], btype="band", fs=FS, output="sos"), x, axis=1)
    ok = good_mask(raw)
    out = {}
    for name, frame in [("contact", run.cq), ("eeg_quality", run.eq)]:
        noise, quality = [], []
        for _, row in frame.iterrows():
            idx = np.flatnonzero((run.t >= row["time"] - 1.0) & (run.t < row["time"]))
            if len(idx) < 100 or not ok[idx].all():
                continue
            noise.append(np.log(np.mean(hf[:, idx] ** 2, axis=1)))
            quality.append([row[ch] for ch in run.labels])
        if not noise:
            out[name] = None
            continue
        noise = np.array(noise)
        noise -= noise.mean(axis=1, keepdims=True)  # channel-specific part only
        quality = np.array(quality)
        rho = {ch: (float(stats.spearmanr(quality[:, j], noise[:, j])[0])
                    if quality[:, j].std() > 0 else None)
               for j, ch in enumerate(run.labels)}
        out[name] = {
            "spearman_quality_vs_own_noise": rho,
            "pct_below_green": {ch: float(100 * np.mean(frame[ch] < 4)) for ch in run.labels},
            "overall_median": float(frame["OVERALL" if name == "contact" else "overall"].median()),
        }
    return out


# ---------------------------------------------------------------------------
# Eyes closed / open alpha


def alpha_spectra(run: Run, raw, bads: list[str] = ()) -> tuple[pd.DataFrame, dict]:
    """Per-interval spectra on 1-40 Hz average-referenced data; 2-s windows, artifact-free only.

    Returns an empty frame when the recording has no eyes closed/open block.
    """

    intervals = find_eye_intervals(run)
    if not intervals:
        return pd.DataFrame(), {}
    r = raw.copy().filter(1.0, 40.0)
    r.info["bads"] = list(bads)
    # Follow the headset's own reference policy.  An average reference is right
    # for the Flex cap and wrong for the EPOC X ring, the same as for ERPs.
    if run.dev.processing.reference == "average":
        r.set_eeg_reference("average")
    x = r.get_data() * 1e6
    ok = good_mask(r)
    good = [j for j, ch in enumerate(run.labels) if ch not in bads]
    n = int(2 * FS)
    rows, spectra, freqs = [], {}, None
    for iv in intervals:
        a = int((iv["start"] + EYE_TRIM[0]) * FS)
        b = int((iv["end"] - EYE_TRIM[1]) * FS)
        segments, rejected = [], 0
        for s in range(a, b - n + 1, n // 2):
            seg = x[:, s:s + n]
            if not ok[s:s + n].all() or np.ptp(seg[good], axis=1).max() > 200:
                rejected += 1
                continue
            segments.append(seg)
        if not segments:
            continue
        freqs, p = welch(np.stack(segments), fs=FS, nperseg=n, axis=2)
        spectra[(iv["state"], iv["pair"])] = p.mean(axis=0)
        rows.append({
            "run": run.key, "state": iv["state"], "pair": iv["pair"],
            "start_s": iv["start"], "duration_s": iv["end"] - iv["start"],
            "windows": len(segments), "rejected_windows": rejected,
        })
    return pd.DataFrame(rows), {"freqs": freqs, "spectra": spectra}


def alpha_scores(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute 8-13 Hz power (dB) and power relative to the 5-7/14-17 Hz flanks (dB)."""

    band = (freqs >= 8) & (freqs <= 13)
    flank = ((freqs >= 5) & (freqs <= 7)) | ((freqs >= 14) & (freqs <= 17))
    hf = (freqs >= 20) & (freqs <= 40)
    return {
        "alpha_db": 10 * np.log10(psd[:, band].mean(axis=1)),
        "alpha_rel_db": 10 * np.log10(psd[:, band].mean(axis=1) / psd[:, flank].mean(axis=1)),
        "hf_db": 10 * np.log10(psd[:, hf].mean(axis=1)),
    }


def cortex_band_power(run: Run, channels: tuple[str, ...] = ("O1", "O2")) -> dict | None:
    """Does Cortex's own band-power stream show the eyes closed/open alpha change?"""

    intervals = find_eye_intervals(run)
    if run.pow is None or not intervals:
        return None
    out = {}
    for ch in channels:
        if f"{ch}/alpha" not in run.pow.columns:
            continue
        diffs_abs, diffs_rel = [], []
        for pair in range(1, 5):
            vals = {}
            for iv in intervals:
                if iv["pair"] != pair:
                    continue
                sel = ((run.pow["time"] >= iv["start"] + EYE_TRIM[0])
                       & (run.pow["time"] <= iv["end"] - EYE_TRIM[1]))
                f = run.pow[sel]
                alpha = f[f"{ch}/alpha"].median()
                flank = (f[f"{ch}/theta"].median() + f[f"{ch}/betaL"].median()) / 2
                vals[iv["state"]] = (alpha, alpha / flank)
            diffs_abs.append(10 * np.log10(vals["closed"][0] / vals["open"][0]))
            diffs_rel.append(10 * np.log10(vals["closed"][1] / vals["open"][1]))
        out[ch] = {"alpha_closed_minus_open_db": diffs_abs,
                   "alpha_vs_theta_betaL_closed_minus_open_db": diffs_rel}
    return out


def pair_stats(d: np.ndarray) -> dict:
    """Paired summary of a closed-minus-open difference across the four pairs."""

    d = np.asarray(d, float)
    return {
        "closed_minus_open": d.tolist(),
        "mean": float(d.mean()),
        "ci95": [float(v) for v in stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))],
        "pairs_closed_greater": int((d > 0).sum()),
        "sign_test_p": float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue),
    }
