#!/usr/bin/env python3
"""Hardware sanity check for an EPOC X (firmware 0x740) recording, compared with EPOC Flex 1.0.

Same checks as ``flex_sanity.py`` (integrity, noise, eyes closed/open alpha,
auditory ERPs), plus:

* the Go/NoGo block, with button presses taken from the PsychoPy CSV;
* a bound on audio latency from the fastest correct Go responses;
* the latency difference between the two headsets.  Both recordings used the
  same PsychoPy script, PC and speaker, so the lag between their tone-locked
  ERPs over the same 14 scalp sites measures the difference in EEG-chain
  latency (radio, dongle, onboard filtering), whatever the audio latency is.

EPOC X timestamps are arrival times and lost samples are not filled, so the
loader rebuilds the regular sample grid from the packet counter and fits the
arrival times against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy import stats
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

import flex_sanity as fx
from flex_sanity import FS, SEED, Run

mne.set_log_level("ERROR")

EPOCX_XDF = "data/Oddball/data/epochx_0740.xdf"
EPOCX_CSV = "data/Oddball/data/epochx_0740.csv"
LSB_UV = 0.128205128205129
OFFSET_UV = 4201.02564096001  # the reader's value for a raw count of 32768
LABELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
FLEX_EQUIVALENT = {"AF3": "Fp1", "AF4": "Fp2"}  # the Flex cap has no AF3/AF4
EOG = ("AF3", "AF4")
FRONTAL_ROI = ["F3", "F4", "FC5", "FC6"]
INFERIOR_ROI = ["T7", "T8", "P7", "P8"]
POSTERIOR_ROI = ["O1", "O2", "P7", "P8"]
CODES = {"passive/standard": 1, "passive/oddball": 2, "gng/go": 5, "gng/nogo": 6}
LAG_WINDOW = (0.15, 0.55)  # s after the marker; spans the N1-P2 complex in both headsets
UPSAMPLE = 8


# ---------------------------------------------------------------------------
# Loading


def robust_line(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    keep = np.ones(len(x), bool)
    for _ in range(6):
        fit = np.polyfit(x[keep], y[keep], 1)
        res = y - np.polyval(fit, x)
        keep = np.abs(res - np.median(res[keep])) < tol
    return fit


def epocx_grid(eeg: dict, diag_stream: dict) -> dict:
    """Rebuild the regular sample grid from the counter and fit arrival times against it.

    Returns grid timestamps (absolute LSL seconds), grid samples, the mask of
    interpolated (missing) samples and the fit diagnostics.
    """

    t_arrival = np.asarray(eeg["time_stamps"], float)
    x = np.asarray(eeg["time_series"], float)
    diag_names = fx._labels(diag_stream)
    diag_values = np.asarray(diag_stream["time_series"], float)
    diag_t = np.asarray(diag_stream["time_stamps"], float)

    # Diagnostics rows carry the EEG sample's timestamp; match them up (a recording can
    # start a row later on one stream).
    pos = np.clip(np.searchsorted(diag_t, t_arrival), 0, len(diag_t) - 1)
    pos_prev = np.clip(pos - 1, 0, len(diag_t) - 1)
    pos = np.where(np.abs(diag_t[pos_prev] - t_arrival) < np.abs(diag_t[pos] - t_arrival), pos_prev, pos)
    matched = np.abs(diag_t[pos] - t_arrival) < 5e-4
    col = {name: i for i, name in enumerate(diag_names)}
    counter = np.where(matched, diag_values[pos, col["COUNTER"]], np.nan)
    for i in np.flatnonzero(~matched):  # infer from a neighbour
        if i + 1 < len(counter) and matched[i + 1]:
            counter[i] = (counter[i + 1] - 1) % 128
        elif i > 0 and matched[i - 1]:
            counter[i] = (counter[i - 1] + 1) % 128
    counter = counter.astype(int)

    # Regular grid: each report advances by its counter step; a whole lost cycle
    # is invisible to the counter, so long arrival gaps add cycles.
    steps = np.diff(counter) % 128
    steps[steps == 0] = 128
    gap_periods = np.diff(t_arrival) * 128.0656
    extra_cycles = np.maximum(0, np.round((gap_periods - steps) / 128)).astype(int)
    steps = steps + 128 * extra_cycles
    grid_index = np.r_[0, np.cumsum(steps)]
    n_grid = grid_index[-1] + 1
    missing = np.ones(n_grid, bool)
    missing[grid_index] = False
    x_grid = np.column_stack([np.interp(np.arange(n_grid), grid_index, x[:, j]) for j in range(x.shape[1])])

    # Timestamps: arrival times fitted against the grid, separately across long gaps.
    breaks = np.flatnonzero(extra_cycles > 0) + 1
    seg_edges = [0, *breaks.tolist(), len(grid_index)]
    t_grid = np.empty(n_grid)
    residual_ms = np.empty(len(grid_index))
    rates = []
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):
        fit = robust_line(grid_index[a:b].astype(float), t_arrival[a:b], 0.004)
        lo = grid_index[a]
        hi = grid_index[b] if b < len(grid_index) else n_grid
        t_grid[lo:hi] = np.polyval(fit, np.arange(lo, hi))
        residual_ms[a:b] = 1000 * (t_arrival[a:b] - np.polyval(fit, grid_index[a:b]))
        rates.append(float(1 / fit[0]))
    return {"t": t_grid, "x": x_grid, "missing": missing, "grid_index": grid_index, "breaks": breaks,
            "residual_ms": residual_ms, "rates": rates, "matched": matched, "x_received": x,
            "diag_values": diag_values, "diag_col": col}


def load_epocx(path: str, csv_path: str) -> tuple[Run, dict]:
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    eeg = by_name("Epoc X")[0]
    labels = fx._labels(eeg)
    assert labels == LABELS, labels
    grid = epocx_grid(eeg, by_name("Epoc X Packet Diagnostics")[0])
    t_grid, x_grid, missing = grid["t"], grid["x"], grid["missing"]
    grid_index, breaks, residual_ms, rates = grid["grid_index"], grid["breaks"], grid["residual_ms"], grid["rates"]
    matched, x, diag_values, col = grid["matched"], grid["x_received"], grid["diag_values"], grid["diag_col"]
    n_grid = len(t_grid)
    t0 = t_grid[0]

    rows = set()
    for stream in by_name("PsychoPy Markers"):
        for ts, value in zip(stream["time_stamps"], stream["time_series"]):
            rows.add((round(float(ts) - t0, 4), str(value[0])))
    markers = pd.DataFrame(sorted(rows), columns=["time", "value"])

    quality = {}
    for name, short in [("Epoc X Contact Quality", "cq"), ("Epoc X EEG Quality", "eq"), ("Epoc X Band Power", "pow")]:
        stream = by_name(name)
        if stream and len(stream[0]["time_stamps"]):
            frame = pd.DataFrame(np.asarray(stream[0]["time_series"], float), columns=fx._labels(stream[0]))
            frame.insert(0, "time", np.asarray(stream[0]["time_stamps"]) - t0)
            quality[short] = frame
        else:
            quality[short] = None

    reset = np.zeros(n_grid)
    reset[grid_index[breaks]] = 1
    diag = {"FILLED": missing.astype(float), "RESET_FLAG": reset}
    run = Run("epocx", t_grid - t0, x_grid, labels, diag, markers, quality["cq"], quality["eq"])
    run.eye_intervals = fx.find_eye_intervals(markers)
    run.tone_events = epocx_events(run, pd.read_csv(csv_path))

    raw_counts = np.round((x - OFFSET_UV) / LSB_UV)
    extra = {
        "pow": quality["pow"],
        "arrival_residual_ms": residual_ms,
        "arrival_grid_index": grid_index,
        "diag_rows_unmatched": int((~matched).sum()),
        "dropped_repeats": int(((diag_values[:, col["GAP_FLAG"]] > 0) & (diag_values[:, col["MISSING_REPORTS"]] == 0)).sum()),
        "reader_missing_reports": int(diag_values[:, col["MISSING_REPORTS"]].sum()),
        "reader_resets": int(diag_values[:, col["RESET_FLAG"]].sum()),
        "segment_rates_hz": rates,
        "decoder_max_fraction": float(np.abs((x - OFFSET_UV) / LSB_UV - raw_counts).max()),
        "raw_count_range": [int(raw_counts.min()), int(raw_counts.max())],
        "rail_samples": int(((raw_counts <= -32768) | (raw_counts >= 32767)).sum()),
    }
    return run, extra


def epocx_events(run: Run, csv: pd.DataFrame) -> pd.DataFrame:
    """Oddball and Go/NoGo tone markers on the grid, with the CSV's responses attached."""

    gng_rows = csv[csv["stim"].isin(["high", "low"])].reset_index(drop=True)
    rows, gng_i = [], 0
    for time, value in run.markers.itertuples(index=False):
        prefix, _, stim = value.partition("-")
        if prefix == "Oddball" and stim in ("standard", "oddball"):
            row = {"block": "passive", "stim": stim}
        elif prefix == "GnG" and stim in ("high", "low"):
            trial = gng_rows.iloc[gng_i]
            assert trial["stim"] == stim, "GnG markers and CSV rows are out of step"
            gng_i += 1
            pressed = isinstance(trial["choice_resp.keys"], str)
            # choice_resp's clock starts at the routine's first flip; the tone is scheduled
            # `soa` after the routine began, so rt - soa is the RT from the scheduled tone
            # (0-1 frame short).
            rt = float(trial["choice_resp.rt"]) - float(trial["soa"]) if pressed else np.nan
            row = {"block": "gng", "stim": "go" if stim == "high" else "nogo", "pressed": pressed,
                   "correct": pressed == (stim == "high"), "rt_from_tone": rt, "soa": float(trial["soa"])}
        else:
            continue
        sample = int(np.argmin(np.abs(run.t - time)))
        rows.append({"time": time, "prefix": prefix, "sample": sample,
                     "sample_error_ms": 1000 * (run.t[sample] - time), **row})
    assert gng_i == len(gng_rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analyses


def integrity(run: Run, extra: dict, raw: mne.io.BaseRaw) -> dict:
    res = extra["arrival_residual_ms"]
    t_rx = run.t[extra["arrival_grid_index"]]
    x = raw.get_data() * 1e6
    x = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), x, axis=1)
    n = int(FS)
    windows = x[:, : (x.shape[1] // n) * n].reshape(x.shape[0], -1, n)
    excursion = 100 * (np.ptp(windows, axis=2) > 400).mean(axis=1)
    blocks = {}
    for block, ev in run.tone_events.groupby("block"):
        span = (t_rx >= ev["time"].min() - 1) & (t_rx <= ev["time"].max() + 1)
        blocks[block] = {
            "tones": int(len(ev)),
            "duration_s": float(ev["time"].max() - ev["time"].min()),
            "isi_median_s": float(np.median(np.diff(ev["time"]))),
            "isi_range_s": [float(np.min(np.diff(ev["time"]))), float(np.max(np.diff(ev["time"])))],
            "missing_samples": int(run.filled[ev["sample"].min():ev["sample"].max()].sum()),
            "arrival_residual_p99_ms": float(np.percentile(res[span], 99)),
            "arrival_residual_max_ms": float(res[span].max()),
        }
    return {
        "samples_received": int(len(res)),
        "grid_samples": int(len(run.t)),
        "missing_samples": int(run.filled.sum()),
        "duration_s": float(run.t[-1]),
        "dropped_repeats": extra["dropped_repeats"],
        "reader_missing_reports": extra["reader_missing_reports"],
        "reader_resets": extra["reader_resets"],
        "diag_rows_unmatched": extra["diag_rows_unmatched"],
        "headset_rate_hz": extra["segment_rates_hz"],
        "arrival_residual_ms_percentiles": dict(zip(["min", "p1", "p50", "p99", "max"],
                                                    np.percentile(res, [0, 1, 50, 99, 100]).round(2).tolist())),
        "decoder_max_fraction": extra["decoder_max_fraction"],
        "raw_count_range": extra["raw_count_range"],
        "rail_samples": extra["rail_samples"],
        "pct_1s_windows_ptp_over_400uv": dict(zip(run.labels, excursion.round(2).tolist())),
        "blocks": blocks,
    }


def cortex_band_power(run: Run, pow_frame: pd.DataFrame | None) -> dict | None:
    """Does Cortex's own band-power stream show the eyes closed/open alpha change?"""

    if pow_frame is None:
        return None
    out = {}
    for ch in ("O1", "O2"):
        diffs_abs, diffs_rel = [], []
        for pair in range(1, 5):
            vals = {}
            for iv in run.eye_intervals:
                if iv["pair"] != pair:
                    continue
                sel = (pow_frame["time"] >= iv["start"] + fx.EYE_TRIM[0]) & (pow_frame["time"] <= iv["end"] - fx.EYE_TRIM[1])
                f = pow_frame[sel]
                alpha = f[f"{ch}/alpha"].median()
                flank = (f[f"{ch}/theta"].median() + f[f"{ch}/betaL"].median()) / 2
                vals[iv["state"]] = (alpha, alpha / flank)
            diffs_abs.append(10 * np.log10(vals["closed"][0] / vals["open"][0]))
            diffs_rel.append(10 * np.log10(vals["closed"][1] / vals["open"][1]))
        out[ch] = {"alpha_closed_minus_open_db": diffs_abs, "alpha_vs_theta_betaL_closed_minus_open_db": diffs_rel}
    return out


def pair_stats(d: np.ndarray) -> dict:
    return {
        "closed_minus_open": d.tolist(),
        "mean": float(d.mean()),
        "ci95": [float(v) for v in stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))],
        "pairs_closed_greater": int((d > 0).sum()),
        "sign_test_p": float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue),
    }


def n1_p2(epochs: mne.Epochs, frontal: list[str], inferior: list[str], rng) -> dict:
    """Trough/peak latencies of the frontal-minus-inferior waveform (sharpest N1/P2 view)."""

    wave = fx.roi_trials(epochs, frontal) - fx.roi_trials(epochs, inferior)
    times = epochs.times
    grand = wave.mean(axis=0)
    n1_window = (0.05, 0.40)
    n1 = fx.peak_latency(times, grand, n1_window, -1)
    p2_window = (n1 + 0.03, 0.55)
    p2 = fx.peak_latency(times, grand, p2_window, +1)
    return {
        "n_trials": int(len(wave)),
        "n1_latency_s": n1, "n1_ci95": fx.bootstrap_latency(times, wave, n1_window, -1, rng=rng),
        "p2_latency_s": p2, "p2_ci95": fx.bootstrap_latency(times, wave, p2_window, +1, rng=rng),
        "n1_amp_uv": float(grand[times == n1][0]), "p2_amp_uv": float(grand[times == p2][0]),
        "_wave": wave,
    }


def lag_between(times: np.ndarray, ref_trials: np.ndarray, other_trials: np.ndarray, n_boot: int, rng) -> dict:
    """Lag (s) that best aligns ``other``'s average to ``ref``'s; positive = other is later."""

    def lag(ref_avg, other_avg):
        up_t = np.arange(len(times) * UPSAMPLE) / (FS * UPSAMPLE) + times[0]
        r = resample_poly(ref_avg, UPSAMPLE, 1)
        o = resample_poly(other_avg, UPSAMPLE, 1)
        sel = (up_t >= LAG_WINDOW[0]) & (up_t <= LAG_WINDOW[1])
        shifts = np.arange(-int(0.2 * FS * UPSAMPLE), int(0.2 * FS * UPSAMPLE) + 1)
        idx = np.flatnonzero(sel)
        scores = []
        for k in shifts:
            j = idx + k
            ok = (j >= 0) & (j < len(o))
            scores.append(np.corrcoef(r[idx[ok]], o[j[ok]])[0, 1])
        best = int(np.argmax(scores))
        return shifts[best] / (FS * UPSAMPLE), float(scores[best])

    observed, corr = lag(ref_trials.mean(axis=0), other_trials.mean(axis=0))
    boot = []
    for _ in range(n_boot):
        a = ref_trials[rng.integers(0, len(ref_trials), len(ref_trials))].mean(axis=0)
        b = other_trials[rng.integers(0, len(other_trials), len(other_trials))].mean(axis=0)
        boot.append(lag(a, b)[0])
    return {"lag_s": float(observed), "corr_at_lag": corr,
            "ci95_s": [float(v) for v in np.percentile(boot, [2.5, 97.5])], "_boot": np.array(boot)}


def flex_matched(n_perm: int) -> dict:
    """Flex runs reduced to the EPOC X's 14 sites and processed the same way."""

    channels = [FLEX_EQUIVALENT.get(ch, ch) for ch in LABELS]
    epochs, noise, psd_avg = [], {}, []
    for key, path in fx.RUNS.items():
        run = fx.load_run(key, path)
        raw = fx.make_raw(run).pick(channels)
        run.labels = channels
        frame = fx.channel_noise(raw)
        noise[key] = frame
        bads = frame.loc[frame["bad"], "channel"].tolist()
        ep, _ = fx.erp_epochs(run, raw, bads, eog_channels=("Fp1", "Fp2"))
        epochs.append(ep)
        x = raw.get_data() * 1e6
        x = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), x, axis=1)
        good = [j for j, ch in enumerate(channels) if ch not in bads]
        f, p = fx.clean_psd(x - x[good].mean(axis=0, keepdims=True), fx.good_mask(raw))
        psd_avg.append(p)
    ep = mne.concatenate_epochs(epochs, add_offset=True)
    return {"epochs": ep, "noise": noise, "freqs": f, "psd_avg": np.mean(psd_avg, axis=0)}


# ---------------------------------------------------------------------------
# Figures


def fig_integrity(run: Run, extra: dict, integ: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 1]})
    t_rx = run.t[extra["arrival_grid_index"]]
    res = extra["arrival_residual_ms"]
    ax = axes[0, 0]
    ax.plot(t_rx, res, ",", color="tab:blue", alpha=0.4)
    for iv in run.eye_intervals:
        ax.axvspan(iv["start"], iv["end"], ymin=0.93, ymax=1.0, color="navy" if iv["state"] == "closed" else "gold", lw=0)
    for block, ev in run.tone_events.groupby("block"):
        ax.axvspan(ev["time"].min(), ev["time"].max(), ymin=0, ymax=0.06,
                   color="tab:green" if block == "passive" else "tab:purple", lw=0)
        ax.text(ev["time"].mean(), -2.2, {"passive": "passive oddball", "gng": "Go/NoGo"}[block], ha="center", fontsize=8)
    ax.set_ylim(-2.5, 10)
    ax.set_xlim(0, run.t[-1])
    ax.set_ylabel("arrival − fitted headset clock (ms)")
    ax.set_title(f"EPOC X: {integ['missing_samples']} missing samples, {integ['dropped_repeats']} dropped repeats, "
                 f"{integ['reader_resets']} resets; headset at {integ['headset_rate_hz'][0]:.4f} Hz "
                 "(top bar navy = eyes closed, gold = open)", fontsize=9)
    ax = axes[0, 1]
    ax.hist(res, bins=np.arange(-2, 10, 0.1), color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("ms")
    ax.set_title("arrival residuals", fontsize=9)
    ax = axes[1, 0]
    x = run.x_uv - np.median(run.x_uv, axis=0)
    step = 300
    for j, ch in enumerate(run.labels):
        ax.plot(run.t[::4], x[::4, j] - step * j, lw=0.3, color="k")
        ax.text(-5, -step * j, ch, ha="right", va="center", fontsize=7)
    ax.set_xlim(0, run.t[-1])
    ax.set_yticks([])
    ax.set_title(f"as recorded, median removed, {step} µV between channels", fontsize=9)
    ax.set_xlabel("s")
    ax = axes[1, 1]
    exc = pd.Series(integ["pct_1s_windows_ptp_over_400uv"]).sort_values()
    ax.barh(exc.index, exc.values, color="tab:gray")
    ax.set_xlabel("% of 1-s windows > 400 µV p-p")
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(out / "01_integrity.png", dpi=130)
    plt.close(fig)


def fig_spectra(spec: dict, flex: dict, noise: pd.DataFrame, flex_noise: dict, out: Path) -> None:
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax.semilogy(spec["freqs"], np.median(spec["as_recorded"], axis=0), color="tab:red", ls="--", label="EPOC X, as recorded (mastoid CMS/DRL)")
    ax.semilogy(spec["freqs"], np.median(spec["avg_ref"], axis=0), color="tab:red", lw=2, label="EPOC X, average reference")
    ax.semilogy(flex["freqs"], np.median(flex["psd_avg"], axis=0), color="tab:blue", lw=2,
                label="Flex (both runs), same 14 sites, average reference")
    for f0 in (50, 60):
        ax.axvline(f0, color="k", ls=":", lw=0.8)
    ax.set_xlim(0, 64)
    ax.set_xlabel("Hz")
    ax.set_ylabel("µV²/Hz (median channel)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Whole-record spectra (clean spans)")
    ax = fig.add_subplot(gs[0, 2])
    sel = (spec["hires_freqs"] > 44) & (spec["hires_freqs"] < 62)
    ax.semilogy(spec["hires_freqs"][sel], np.median(spec["hires_avg"], axis=0)[sel], lw=0.7, color="tab:red")
    ax.set_title("EPOC X 44–62 Hz, 0.016 Hz resolution", fontsize=9)
    ax.set_xlabel("Hz")
    ax = fig.add_subplot(gs[1, 0])
    frame = noise.sort_values("hf_20_40_uv2")
    ax.barh(frame["channel"], frame["hf_20_40_uv2"], color=np.where(frame["bad"], "tab:red", "tab:gray"))
    ax.set_xscale("log")
    ax.set_xlabel("20–40 Hz power (µV², median ref)")
    ax.set_title("EPOC X per-channel noise (red = flagged)", fontsize=9)
    ax = fig.add_subplot(gs[1, 1:])
    names = [FLEX_EQUIVALENT.get(ch, ch) for ch in LABELS]
    width = 0.25
    xs = np.arange(len(LABELS))
    ax.bar(xs - width, noise.set_index("channel").loc[LABELS, "hf_20_40_uv2"], width, color="tab:red", label="EPOC X")
    for k, (key, frame) in enumerate(flex_noise.items()):
        ax.bar(xs + k * width, frame.set_index("channel").loc[names, "hf_20_40_uv2"], width,
               color=["tab:blue", "lightsteelblue"][k], label=f"Flex {key}")
    ax.set_xticks(xs, [f"{a}" if a not in FLEX_EQUIVALENT else f"{a}/{FLEX_EQUIVALENT[a]}" for a in LABELS], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("20–40 Hz power (µV²)")
    ax.set_title("Per-site noise, same 14 sites and median reference (Flex uses Fp1/Fp2 for AF3/AF4)", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(out / "02_spectra_noise.png", dpi=130)
    plt.close(fig)


def fig_alpha(frame: pd.DataFrame, sp: dict, topo: np.ndarray, labels: list[str], roi: list[str], info, cortex, out: Path) -> None:
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, :2])
    idx = [labels.index(ch) for ch in roi]
    freqs = sp["freqs"]
    for (state, pair), psd in sp["spectra"].items():
        ax.semilogy(freqs, psd[idx].mean(axis=0), color="tab:blue" if state == "closed" else "tab:orange", lw=0.7, alpha=0.5)
    for state, color in [("closed", "tab:blue"), ("open", "tab:orange")]:
        mean = np.mean([psd[idx].mean(axis=0) for (s, _), psd in sp["spectra"].items() if s == state], axis=0)
        ax.semilogy(freqs, mean, color=color, lw=2.2, label=f"eyes {state}")
    ax.axvspan(8, 13, color="0.9", zorder=0)
    ax.set_xlim(1, 40)
    in_band = (freqs >= 1) & (freqs <= 40)
    vals = np.concatenate([psd[idx].mean(axis=0)[in_band] for psd in sp["spectra"].values()])
    ax.set_ylim(vals.min() * 0.7, vals.max() * 1.4)
    ax.set_title(f"EPOC X posterior ROI ({', '.join(roi)}), average reference", fontsize=9)
    ax.set_xlabel("Hz")
    ax.set_ylabel("µV²/Hz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[0, 2:])
    for ch in roi:
        j = labels.index(ch)
        closed = np.mean([psd[j] for (s, _), psd in sp["spectra"].items() if s == "closed"], axis=0)
        opened = np.mean([psd[j] for (s, _), psd in sp["spectra"].items() if s == "open"], axis=0)
        ax.plot(freqs, 10 * np.log10(closed / opened), label=ch)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(8, 13, color="0.9", zorder=0)
    ax.set_xlim(1, 40)
    ax.set_xlabel("Hz")
    ax.set_ylabel("closed / open (dB)")
    ax.set_title("Eyes-closed change per channel", fontsize=9)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 0])
    pivot = frame.pivot_table(index="pair", columns="state", values="roi_alpha_rel_db")
    pivot_abs = frame.pivot_table(index="pair", columns="state", values="roi_alpha_db")
    ax.scatter(np.zeros(4) + np.linspace(-0.1, 0.1, 4), pivot["closed"] - pivot["open"], color="tab:red")
    ax.scatter(np.ones(4) + np.linspace(-0.1, 0.1, 4), pivot_abs["closed"] - pivot_abs["open"], color="tab:red", marker="s")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1], ["alpha vs flanks", "absolute alpha"])
    ax.set_ylabel("closed − open (dB), ROI")
    ax.set_title("Per pair", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    pivot = frame.pivot_table(index="pair", columns="state", values="blinks_per_min")
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row["closed"], row["open"]], color="tab:red", marker="o")
    ax.set_xticks([0, 1], ["closed", "open"])
    ax.set_ylabel("AF3/AF4 blinks / min")
    ax.set_title("Marker check", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    keep = np.flatnonzero(np.isfinite(topo))
    lim = max(np.abs(topo[keep]).max(), 0.5)
    im, _ = mne.viz.plot_topomap(topo[keep], mne.pick_info(info, keep), axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-lim, lim), names=[labels[k] for k in keep], sensors=True)
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("closed − open, alpha vs flanks (dB)", fontsize=9)
    ax = fig.add_subplot(gs[1, 3])
    if cortex:
        for k, ch in enumerate(cortex):
            for m, (metric, marker) in enumerate([("alpha_closed_minus_open_db", "s"), ("alpha_vs_theta_betaL_closed_minus_open_db", "o")]):
                ax.scatter(np.full(4, k + 0.2 * m) + np.linspace(-0.05, 0.05, 4), cortex[ch][metric], marker=marker,
                           color="tab:purple", label=("absolute" if marker == "s" else "vs theta/betaL") if k == 0 else None)
        ax.set_xticks([0.1, 1.1], list(cortex))
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=7)
        ax.set_title("Cortex band-power stream: closed − open alpha (dB)", fontsize=9)
    fig.savefig(out / "03_alpha_eyes.png", dpi=130)
    plt.close(fig)


def fig_erps(epochs: mne.Epochs, latency: dict, tests: dict, out: Path) -> None:
    meta = epochs.metadata
    times = epochs.times
    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    groups = {
        "passive standard": (meta["block"] == "passive") & (meta["stim"] == "standard"),
        "passive oddball": (meta["block"] == "passive") & (meta["stim"] == "oddball"),
        "Go (hit)": (meta["stim"] == "go") & meta["correct"].astype(bool),
        "NoGo (correct)": (meta["stim"] == "nogo") & meta["correct"].astype(bool),
    }
    colors = {"passive standard": "0.35", "passive oddball": "tab:red", "Go (hit)": "tab:green", "NoGo (correct)": "tab:purple"}
    for col, (roi, title) in enumerate([(FRONTAL_ROI, "F3/F4/FC5/FC6"), (INFERIOR_ROI, "T7/T8/P7/P8"), (POSTERIOR_ROI, "O1/O2/P7/P8")]):
        ax = fig.add_subplot(gs[0, col])
        for name, mask in groups.items():
            y = fx.roi_trials(epochs[mask.to_numpy()], roi)
            se = y.std(axis=0, ddof=1) / np.sqrt(len(y))
            ax.plot(times, y.mean(axis=0), color=colors[name], lw=1.4, label=f"{name} (n={len(y)})")
            ax.fill_between(times, y.mean(axis=0) - se, y.mean(axis=0) + se, color=colors[name], alpha=0.15)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(latency["n1_latency_s"], color="tab:red", ls=":", lw=1)
        ax.set_title(f"EPOC X {title} (red dotted = N1 at {1000 * latency['n1_latency_s']:.0f} ms)", fontsize=9)
        ax.set_xlabel("s from tone marker")
        if col == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel("µV")
    evoked = epochs[groups["passive standard"].to_numpy()].average()
    ax = fig.add_subplot(gs[0, 3])
    evoked.plot_topomap(times=[latency["n1_latency_s"]], axes=ax, show=False, colorbar=False, average=0.03, sensors=True)
    ax.set_title(f"standards at N1 ({1000 * latency['n1_latency_s']:.0f} ms)", fontsize=9)

    for row, (name, test) in enumerate(tests.items(), start=1):
        a_mask, b_mask, a_label, b_label = test["masks_def"]
        for col, (roi, title) in enumerate([(FRONTAL_ROI, "frontal"), (INFERIOR_ROI, "inferior"), (POSTERIOR_ROI, "posterior")]):
            ax = fig.add_subplot(gs[row, col])
            for mask, label, color in [(b_mask, b_label, "0.35"), (a_mask, a_label, "tab:red")]:
                y = fx.roi_trials(epochs[mask], roi)
                se = y.std(axis=0, ddof=1) / np.sqrt(len(y))
                ax.plot(times, y.mean(axis=0), color=color, lw=1.4, label=f"{label} (n={len(y)})")
                ax.fill_between(times, y.mean(axis=0) - se, y.mean(axis=0) + se, color=color, alpha=0.2)
            ch_idx = [epochs.ch_names.index(ch) for ch in roi]
            for cmask, p in zip(test["masks"], test["p_values"]):
                if p < 0.05:
                    on = cmask[:, ch_idx].any(axis=1)
                    if on.any():
                        ax.axvspan(test["times"][on].min(), test["times"][on].max(), color="gold", alpha=0.3, zorder=0)
            ax.axvline(0, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{name}: {title} ROI", fontsize=9)
            if col == 0:
                ax.legend(fontsize=7)
                ax.set_ylabel("µV")
        ax = fig.add_subplot(gs[row, 3])
        best = test["clusters"][0] if test["clusters"] else None
        if best:
            diff = mne.combine_evoked([epochs[a_mask].average(), epochs[b_mask].average()], weights=[1, -1])
            centre = (best["t_start"] + best["t_end"]) / 2
            diff.plot_topomap(times=[centre], average=max(best["t_end"] - best["t_start"], 1 / FS), axes=ax,
                              show=False, colorbar=False, sensors=True)
            ax.set_title(f"{a_label} − {b_label}, {1000 * best['t_start']:.0f}–{1000 * best['t_end']:.0f} ms\n"
                         f"strongest cluster p = {best['p']:.3f}", fontsize=8)
    fig.suptitle("EPOC X ERPs (gold = significant cluster touching the ROI, p < .05)", fontsize=11)
    fig.savefig(out / "04_erps.png", dpi=130)
    plt.close(fig)


def fig_latency(times, flex_wave, epx_wave, lag, rts, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    for wave, label, color in [(flex_wave, "Flex 1.0", "tab:blue"), (epx_wave, "EPOC X", "tab:red")]:
        mean = wave.mean(axis=0)
        se = wave.std(axis=0, ddof=1) / np.sqrt(len(wave))
        ax.plot(times, mean, color=color, lw=1.8, label=f"{label} passive standards (n={len(wave)})")
        ax.fill_between(times, mean - se, mean + se, color=color, alpha=0.2)
    ax.plot(times + lag["lag_s"], flex_wave.mean(axis=0), color="tab:blue", ls="--", lw=1,
            label=f"Flex shifted by {1000 * lag['lag_s']:+.0f} ms")
    ax.axvspan(*LAG_WINDOW, color="0.92", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("s from tone marker (same PsychoPy script, PC and speaker)")
    ax.set_ylabel("µV")
    ax.set_title("(F3/F4/FC5/FC6) − (T7/T8/P7/P8), both headsets; grey = alignment window", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(1000 * lag["_boot"], bins=40, color="tab:gray")
    ax.axvline(1000 * lag["lag_s"], color="k")
    ax.set_xlabel("EPOC X lag relative to Flex (ms)")
    ax.set_title(f"bootstrap: {1000 * lag['lag_s']:+.0f} ms [{1000 * lag['ci95_s'][0]:+.0f}, {1000 * lag['ci95_s'][1]:+.0f}]", fontsize=9)
    ax = axes[2]
    ax.hist(rts, bins=np.arange(0.3, 0.85, 0.02), color="tab:green")
    ax.axvline(np.percentile(rts, 5), color="k", ls="--", label="5th percentile")
    ax.set_xlabel("Go RT from scheduled tone (s)")
    ax.set_title("Go responses (correct)", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(out / "05_latency_comparison.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/Oddball/data/epocx_sanity"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results: dict = {}

    print("load EPOC X")
    run, extra = load_epocx(EPOCX_XDF, EPOCX_CSV)
    raw = fx.make_raw(run)
    integ = integrity(run, extra, raw)
    noise = fx.channel_noise(raw)
    bads = noise.loc[noise["bad"], "channel"].tolist()
    results["integrity"] = integ
    results["bad_channels"] = bads
    results["channel_noise"] = noise

    ok = fx.good_mask(raw)
    x = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), raw.get_data() * 1e6, axis=1)
    good = [j for j, ch in enumerate(run.labels) if ch not in bads]
    avg = x - x[good].mean(axis=0, keepdims=True)
    freqs, p_rec = fx.clean_psd(x, ok)
    _, p_avg = fx.clean_psd(avg, ok)
    hi_f, p_hi = fx.clean_psd(avg, ok, nperseg=8192)
    band = (freqs >= 20) & (freqs <= 40)
    spec = {"freqs": freqs, "as_recorded": p_rec, "avg_ref": p_avg, "hires_freqs": hi_f, "hires_avg": p_hi}
    results["psd_20_40_median_channel"] = {"as_recorded": float(np.median(p_rec[:, band])), "avg_ref": float(np.median(p_avg[:, band]))}
    results["line_noise"] = fx.line_check(hi_f, np.median(p_hi, axis=0))
    notch = {}
    for f0 in (50, 60):
        near = np.abs(hi_f - f0) <= 0.5
        flank = (np.abs(hi_f - f0) > 2) & (np.abs(hi_f - f0) < 4)
        notch[f"{f0}Hz_depth_db"] = float(10 * np.log10(np.median(p_hi, axis=0)[near].min() / np.median(np.median(p_hi, axis=0)[flank])))
    results["line_noise"].update(notch)
    results["quality_streams"] = fx.quality_vs_noise(run, raw)
    fig_integrity(run, extra, integ, out)

    print("Flex, matched montage")
    flex = flex_matched(args.permutations)
    results["flex_matched_psd_20_40_avg_ref"] = float(np.median(flex["psd_avg"][:, (flex["freqs"] >= 20) & (flex["freqs"] <= 40)]))
    fig_spectra(spec, flex, noise, flex["noise"], out)

    print("alpha")
    frame, sp = fx.alpha_analysis(run, raw, bads, blink_channels=EOG)
    roi = [ch for ch in POSTERIOR_ROI if ch not in bads]
    idx = [run.labels.index(ch) for ch in roi]
    for metric in ("alpha_rel_db", "alpha_db", "hf_db"):
        frame[f"roi_{metric}"] = [fx.alpha_scores(sp["freqs"], sp["spectra"][(s, p)])[metric][idx].mean()
                                  for s, p in zip(frame["state"], frame["pair"])]
    results["alpha_intervals"] = frame
    results["alpha"] = {}
    for metric in ("roi_alpha_rel_db", "roi_alpha_db", "roi_hf_db", "blinks_per_min"):
        pivot = frame.pivot_table(index="pair", columns="state", values=metric)
        results["alpha"][metric] = pair_stats((pivot["closed"] - pivot["open"]).to_numpy())
    diffs = []
    for pair in range(1, 5):
        c = fx.alpha_scores(sp["freqs"], sp["spectra"][("closed", pair)])["alpha_rel_db"]
        o = fx.alpha_scores(sp["freqs"], sp["spectra"][("open", pair)])["alpha_rel_db"]
        diffs.append(c - o)
    topo = np.mean(diffs, axis=0)
    topo[[run.labels.index(ch) for ch in bads]] = np.nan
    results["alpha"]["topography_alpha_rel_db"] = {ch: (None if not np.isfinite(v) else round(float(v), 2)) for ch, v in zip(run.labels, topo)}
    closed_psd = np.mean([psd[idx].mean(axis=0) for (s, _), psd in sp["spectra"].items() if s == "closed"], axis=0)
    open_psd = np.mean([psd[idx].mean(axis=0) for (s, _), psd in sp["spectra"].items() if s == "open"], axis=0)
    sel = (sp["freqs"] >= 7) & (sp["freqs"] <= 14)
    results["alpha"]["closed_over_open_peak_hz"] = float(sp["freqs"][sel][np.argmax(closed_psd[sel] / open_psd[sel])])
    results["alpha"]["closed_over_open_peak_db"] = float(10 * np.log10((closed_psd[sel] / open_psd[sel]).max()))
    cortex = cortex_band_power(run, extra["pow"])
    results["cortex_band_power_eyes"] = cortex
    fig_alpha(frame, sp, topo, run.labels, roi, raw.info, cortex, out)

    print("ERPs")
    epochs, prep = fx.erp_epochs(run, raw, bads, eog_channels=EOG, codes=CODES)
    results["erp_preprocessing"] = prep
    meta = epochs.metadata
    ev = run.tone_events
    results["erp_trial_counts"] = {
        f"{b}/{s}": {"kept": int(((meta["block"] == b) & (meta["stim"] == s)).sum()), "total": int(((ev["block"] == b) & (ev["stim"] == s)).sum())}
        for b, s in [("passive", "standard"), ("passive", "oddball"), ("gng", "go"), ("gng", "nogo")]}
    gng = ev[ev["block"] == "gng"]
    go_rt = gng.loc[(gng["stim"] == "go") & gng["pressed"], "rt_from_tone"].to_numpy()
    valid_rt = go_rt[go_rt > 0.1]
    results["behaviour"] = {
        "go_hits": int(((gng["stim"] == "go") & gng["pressed"]).sum()), "go_trials": int((gng["stim"] == "go").sum()),
        "nogo_false_alarms": int(((gng["stim"] == "nogo") & gng["pressed"]).sum()), "nogo_trials": int((gng["stim"] == "nogo").sum()),
        "go_rt_from_tone_s": {"p5": float(np.percentile(valid_rt, 5)), "min": float(valid_rt.min()),
                              "median": float(np.median(valid_rt)), "p95": float(np.percentile(valid_rt, 95))},
        "go_presses_before_tone": int((go_rt <= 0.1).sum()),
        "audio_plus_keyboard_latency_bound_s": {
            "if_fastest_true_rt_is_0.25s": float(np.percentile(valid_rt, 5) - 0.25),
            "if_fastest_true_rt_is_0.30s": float(np.percentile(valid_rt, 5) - 0.30),
        },
    }

    passive_std = epochs[((meta["block"] == "passive") & (meta["stim"] == "standard")).to_numpy()]
    latency = n1_p2(passive_std, FRONTAL_ROI, INFERIOR_ROI, rng)
    single = {}
    for roi_name, roi_chs, sign in [("frontal", FRONTAL_ROI, -1), ("inferior", INFERIOR_ROI, +1)]:
        grand = fx.roi_trials(passive_std, roi_chs).mean(axis=0)
        single[roi_name] = float(fx.peak_latency(passive_std.times, grand, (0.05, 0.40), sign))
    latency["roi_extreme_latency_s"] = single
    epx_wave = latency.pop("_wave")
    wave = epx_wave
    sel = (passive_std.times >= 0) & (passive_std.times <= 0.6)
    r = float(np.corrcoef(wave[0::2].mean(axis=0)[sel], wave[1::2].mean(axis=0)[sel])[0, 1])
    latency["split_half_r"] = r
    per_block = {}
    for name, mask in [("gng go", (meta["stim"] == "go") & meta["correct"].astype(bool)),
                       ("gng nogo", (meta["stim"] == "nogo") & meta["correct"].astype(bool))]:
        w = fx.roi_trials(epochs[mask.to_numpy()], FRONTAL_ROI) - fx.roi_trials(epochs[mask.to_numpy()], INFERIOR_ROI)
        per_block[name] = {"n1_latency_s": fx.peak_latency(epochs.times, w.mean(axis=0), (0.05, 0.40), -1)}
    latency["per_block"] = per_block
    results["standards"] = latency

    print("cluster tests")
    passive = (meta["block"] == "passive").to_numpy()
    odd = (meta["stim"] == "oddball").to_numpy()
    go = ((meta["stim"] == "go") & meta["correct"].astype(bool)).to_numpy()
    nogo = ((meta["stim"] == "nogo") & meta["correct"].astype(bool)).to_numpy()
    tests = {}
    for name, a_mask, b_mask, a_label, b_label in [
        ("passive oddball vs standard", passive & odd, passive & ~odd, "oddball", "standard"),
        ("NoGo vs Go", nogo, go, "NoGo", "Go"),
    ]:
        test = fx.cluster_test(epochs[a_mask], epochs[b_mask], n_perm=args.permutations)
        test["masks_def"] = (a_mask, b_mask, a_label, b_label)
        tests[name] = test
    results["condition_tests"] = {name: {k: v for k, v in t.items() if k in ("n_oddball", "n_standard", "clusters")}
                                  for name, t in tests.items()}
    fig_erps(epochs, latency, tests, out)

    print("latency comparison with Flex")
    flex_ep = flex["epochs"]
    fmeta = flex_ep.metadata
    flex_std = flex_ep[((fmeta["block"] == "passive") & (fmeta["stim"] == "standard")).to_numpy()]
    flex_lat = n1_p2(flex_std, FRONTAL_ROI, INFERIOR_ROI, rng)
    flex_wave = flex_lat.pop("_wave")
    lag = lag_between(passive_std.times, flex_wave, epx_wave, args.bootstrap, rng)
    results["latency_comparison"] = {
        "flex_matched_montage": flex_lat,
        "epocx": {k: v for k, v in latency.items() if k.startswith(("n1", "p2", "n_trials"))},
        "epocx_minus_flex_lag_s": lag["lag_s"], "lag_ci95_s": lag["ci95_s"], "corr_at_lag": lag["corr_at_lag"],
        "window_s": list(LAG_WINDOW),
    }
    fig_latency(passive_std.times, flex_wave, epx_wave, lag, valid_rt, out)

    (out / "results.json").write_text(json.dumps(fx.to_jsonable(results), indent=2), encoding="utf-8")
    noise.to_csv(out / "channel_noise.csv", index=False)
    frame.to_csv(out / "alpha_intervals.csv", index=False)
    print(out / "results.json")


if __name__ == "__main__":
    main()
