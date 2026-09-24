#!/usr/bin/env python3
"""EEG-chain latency from a DRT stimulator held against one sensor.

The DRT board drives a tactile/LED output on a millisecond clock that PsychoPy
zeroes with ``resettimer`` (the ``DRTStart`` marker).  Switching the output on
and off couples a sharp electrical transient into the headset.  The time from
the board's stimulus edge to that transient in the recorded EEG timestamps is
the whole EEG-chain latency: onboard filtering, radio, dongle, USB and reader.
That is exactly the offset between EEG timestamps and real time that ERP
analyses need to remove.

Stimulus times are ``DRTStart + OnsetTimeDRT`` from the DRT CSV (every trial,
including the last, whose ``_ON`` marker is pushed after the trial ends); the
``_ON`` markers are checked against them.  If the CSV was not exported, the
corrected ``_ON`` marker timestamps are used directly and that limitation is
recorded in the result.  Each trial samples the transient at a different phase
of the 128 Hz sample clock, so pooling trials by exact time from the stimulus
reconstructs the transient at sub-millisecond resolution.

The edge time is the signed-slope centroid of the pooled waveform, not the
amplitude peak of the filtered transient.  The offset edge is located in the
same onset-locked waveform; its separation from the onset is the measured
pulse width.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import savgol_filter

import emotiv as em

SEED = 7
BASELINE = (-0.30, -0.02)  # s before the stimulus edge
GRID = np.arange(-0.060, 0.200, 0.00025)
KERNEL_S = 0.001
SPIKE_WINDOW = (0.040, 0.120)  # where the main transient is searched for
SHIFT_WINDOW = (0.040, 0.110)  # used to align single trials to the template
EDGE_GRID = np.arange(-0.060, 0.400, 0.00025)
EDGE_SEARCH_WINDOWS = {"onset": (0.030, 0.130), "offset": (0.130, 0.300)}
EDGE_SMOOTH_POINTS = 17  # 4 ms on the 0.25-ms template grid
EDGE_THRESHOLD = 0.10  # retain the main contiguous slope around each edge


def load(xdf: str) -> dict:
    streams, _ = pyxdf.load_xdf(xdf, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    if by_name("Epoc X") and len(by_name("Epoc X")[0]["time_stamps"]):
        eeg = by_name("Epoc X")[0]
        grid = em.epocx_grid(eeg, by_name("Epoc X Packet Diagnostics")[0])
        device, t, x, bad = "EPOC X", grid["t"], grid["x"], grid["missing"]
        clock = "arrival times fitted against the packet counter"
    else:
        eeg = by_name("Epoc Flex 1.0")[0]
        diag = by_name("Epoc Flex 1.0 Packet Diagnostics")[0]
        names = em.loading.stream_labels(diag)
        device, t, x = "Flex 1.0", np.asarray(eeg["time_stamps"], float), np.asarray(eeg["time_series"], float)
        bad = np.asarray(diag["time_series"], float)[:, names.index("FILLED")] > 0
        clock = "reader counter-clock timestamps"
    markers = [(float(ts), str(v[0])) for s in by_name("PsychoPy Markers") for ts, v in zip(s["time_stamps"], s["time_series"])]
    return {"device": device, "clock": clock, "t": t, "x": x, "bad": bad, "labels": em.loading.stream_labels(eeg),
            "markers": pd.DataFrame(sorted(set(markers)), columns=["time", "value"])}


def pooled(t, signal, bad, events, pre=0.35, post=0.25, baseline_events=None):
    """Baseline-corrected samples around each event, with an optional baseline reference.

    The LED pulse in the uploaded run is only 100 ms long.  For its offset edge,
    the samples immediately before the edge are still part of the onset response,
    so the baseline must be taken before the corresponding onset instead of before
    the offset event itself.
    """

    events = np.asarray(events, float)
    baseline_events = events if baseline_events is None else np.asarray(baseline_events, float)
    if len(events) != len(baseline_events):
        raise ValueError("events and baseline_events must have the same length")
    rel, val, trial = [], [], []
    for k, (e, baseline_event) in enumerate(zip(events, baseline_events)):
        a, b = np.searchsorted(t, e - pre), np.searchsorted(t, e + post)
        ba, bb = np.searchsorted(t, baseline_event + BASELINE[0]), np.searchsorted(t, baseline_event + BASELINE[1])
        if a < 1 or b >= len(t) or ba < 0 or bb > len(t) or bad[a:b].any() or bad[ba:bb].any() or bb <= ba:
            continue
        r = t[a:b] - e
        v = signal[a:b] - signal[ba:bb].mean()
        rel.append(r)
        val.append(v)
        trial.append(np.full(len(r), k))
    if not rel:
        raise ValueError("no usable event epochs (check recording coverage and bad samples)")
    return np.concatenate(rel), np.concatenate(val), np.concatenate(trial)


def kernel_template(rel, val, grid=GRID):
    """Gaussian-kernel template without materialising a huge dense matrix.

    The 0.25-ms output grid and 1-ms kernel used by the original analysis make
    contributions more than 4 ms away negligible.  Accumulating only that local
    neighbourhood is numerically equivalent at the precision reported here and
    avoids allocating a multi-hundred-megabyte matrix on every bootstrap draw.
    """

    rel = np.asarray(rel, float)
    val = np.asarray(val, float)
    grid = np.asarray(grid, float)
    if len(grid) > 1:
        step = float(np.median(np.diff(grid)))
    else:
        step = np.inf
    if np.isfinite(step) and step > 0 and np.max(np.abs(np.diff(grid) - step)) < 1e-9:
        radius = max(1, int(np.ceil(4 * KERNEL_S / step)))
        centre = np.floor((rel - grid[0]) / step + 0.5).astype(int)
        num = np.zeros(len(grid), float)
        den = np.zeros(len(grid), float)
        for offset in range(-radius, radius + 1):
            idx = centre + offset
            keep = (idx >= 0) & (idx < len(grid))
            if not np.any(keep):
                continue
            ii = idx[keep]
            w = np.exp(-0.5 * ((rel[keep] - grid[ii]) / KERNEL_S) ** 2)
            num += np.bincount(ii, weights=w * val[keep], minlength=len(grid))
            den += np.bincount(ii, weights=w, minlength=len(grid))
        return num / np.maximum(den, np.finfo(float).tiny)

    # Small/non-uniform grids are used by trial_shifts; the dense operation is
    # cheap there and retains the exact leave-one-out calculation.
    w = np.exp(-0.5 * ((rel[None, :] - grid[:, None]) / KERNEL_S) ** 2)
    return (w @ val) / w.sum(axis=1)


def spike_features(rel, val, sign: int, baseline_window=BASELINE, find_departure=True) -> dict:
    tpl = kernel_template(rel, val)
    base = val[(rel >= baseline_window[0]) & (rel <= baseline_window[1])]
    if len(base) < 2:
        raise ValueError("baseline window contains too few samples")
    sd = base.std()
    sel = (GRID >= SPIKE_WINDOW[0]) & (GRID <= SPIKE_WINDOW[1])
    peak = GRID[sel][np.argmax(sign * tpl[sel])]
    # First departure from baseline (either sign) before the peak: > 3 baseline SDs of
    # the single samples, sustained for 2 ms of the template.
    over = np.abs(tpl) > 3 * sd
    sustained = np.convolve(over.astype(int), np.ones(8, int), mode="same") >= 8
    early = GRID[(GRID > 0) & (GRID < peak) & sustained] if find_departure else np.array([])
    return {"peak_s": float(peak), "peak_amp": float(tpl[GRID == peak][0]),
            "first_departure_s": float(early.min()) if len(early) else None, "baseline_sd": float(sd), "_tpl": tpl}


def bootstrap(rel, val, trial, sign, n_boot, rng, baseline_window=BASELINE, find_departure=True) -> dict:
    ids = np.unique(trial)
    by_trial = {k: np.flatnonzero(trial == k) for k in ids}
    peaks, firsts = [], []
    for _ in range(n_boot):
        pick = np.concatenate([by_trial[k] for k in rng.choice(ids, len(ids))])
        f = spike_features(rel[pick], val[pick], sign, baseline_window, find_departure)
        peaks.append(f["peak_s"])
        if f["first_departure_s"] is not None:
            firsts.append(f["first_departure_s"])
    return {"peak_ci95_s": np.percentile(peaks, [2.5, 97.5]).tolist(),
            "first_departure_ci95_s": np.percentile(firsts, [2.5, 97.5]).tolist() if firsts else None}


def trial_shifts(rel, val, trial, shift_window=SHIFT_WINDOW) -> np.ndarray:
    """Per-trial time shift against a leave-one-out template (least squares, free gain)."""

    shifts = np.arange(-0.012, 0.012001, 0.00025)
    out = []
    for k in np.unique(trial):
        mine = trial == k
        others = ~mine
        r, v = rel[mine], val[mine]
        sel = (r >= shift_window[0]) & (r <= shift_window[1])
        if sel.sum() < 5:
            continue
        sse = []
        for s in shifts:
            tpl = kernel_template(rel[others], val[others], grid=r[sel] - s)
            gain = (tpl @ v[sel]) / (tpl @ tpl)
            sse.append(np.sum((v[sel] - gain * tpl) ** 2))
        out.append(shifts[int(np.argmin(sse))])
    return np.asarray(out)


def edge_features(rel, val, search_window, baseline_window=BASELINE) -> dict:
    """Locate one physical edge from the slope of the pooled recorded waveform.

    A step edge is represented by a broad transient in the EEG, so its amplitude
    peak is not the edge time.  The smoothed derivative is localized around the
    edge; its signed-weight centroid is the timing feature.  The window is only a
    broad search range -- the edge position and pulse width are both estimated
    from the data.
    """

    tpl = kernel_template(rel, val, EDGE_GRID)
    smooth = savgol_filter(tpl, EDGE_SMOOTH_POINTS, 2)
    derivative = np.gradient(smooth, EDGE_GRID)
    selected = (EDGE_GRID >= search_window[0]) & (EDGE_GRID <= search_window[1])
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError(f"edge search window is outside the template: {search_window}")
    peak_i = indices[np.argmax(np.abs(derivative[selected]))]
    sign = 1.0 if derivative[peak_i] >= 0 else -1.0
    peak_abs = abs(float(derivative[peak_i]))
    active = selected & (sign * derivative >= EDGE_THRESHOLD * peak_abs)

    # Keep the contiguous above-threshold run containing the strongest slope;
    # this excludes the opposite edge and the filter's smaller ringing lobes.
    lo = hi = peak_i
    while lo > indices[0] and active[lo - 1]:
        lo -= 1
    while hi < indices[-1] and active[hi + 1]:
        hi += 1
    run = np.arange(lo, hi + 1)
    weights = sign * derivative[run]
    edge_s = float(np.sum(EDGE_GRID[run] * weights) / np.sum(weights))
    base = val[(rel >= baseline_window[0]) & (rel <= baseline_window[1])]
    if len(base) < 2:
        raise ValueError("baseline window contains too few samples")
    return {
        # peak_s is retained as the public field name used by the earlier DRT
        # report; it now contains the edge centroid, not the transient maximum.
        "peak_s": edge_s,
        "edge_s": edge_s,
        "edge_peak_s": float(EDGE_GRID[peak_i]),
        "edge_peak_slope_uv_s": float(derivative[peak_i]),
        "edge_window_s": [float(EDGE_GRID[lo]), float(EDGE_GRID[hi])],
        "baseline_sd": float(base.std()),
        "sign": int(sign),
        "_tpl": tpl,
        "_derivative": derivative,
    }


def bootstrap_edge(rel, val, trial, search_window, n_boot, rng, baseline_window=BASELINE) -> dict:
    """Trial-bootstrap confidence interval for a data-localized edge."""

    ids = np.unique(trial)
    by_trial = {k: np.flatnonzero(trial == k) for k in ids}
    edges = []
    for _ in range(n_boot):
        pick = np.concatenate([by_trial[k] for k in rng.choice(ids, len(ids))])
        edges.append(edge_features(rel[pick], val[pick], search_window, baseline_window)["edge_s"])
    return {"edge_ci95_s": np.percentile(edges, [2.5, 97.5]).tolist()}


def bootstrap_edge_pair(rel, val, trial, n_boot, rng, baseline_window=BASELINE) -> dict:
    """Joint trial-bootstrap intervals for both edges and their measured spacing."""

    ids = np.unique(trial)
    by_trial = {k: np.flatnonzero(trial == k) for k in ids}
    onset, offset = [], []
    for _ in range(n_boot):
        pick = np.concatenate([by_trial[k] for k in rng.choice(ids, len(ids))])
        onset.append(edge_features(rel[pick], val[pick], EDGE_SEARCH_WINDOWS["onset"], baseline_window)["edge_s"])
        offset.append(edge_features(rel[pick], val[pick], EDGE_SEARCH_WINDOWS["offset"], baseline_window)["edge_s"])
    onset, offset = np.asarray(onset), np.asarray(offset)
    width = offset - onset
    return {
        "onset_edge_ci95_s": np.percentile(onset, [2.5, 97.5]).tolist(),
        "offset_edge_ci95_s": np.percentile(offset, [2.5, 97.5]).tolist(),
        "pulse_width_ci95_s": np.percentile(width, [2.5, 97.5]).tolist(),
        "onset_edges_s": onset,
        "offset_edges_s": offset,
    }


def channel_from_filename(path: str, labels: list[str]) -> str | None:
    """Infer a sensor suffix such as ``..._P7.xdf`` when it is present."""

    stem = Path(path).stem.upper()
    for label in labels:
        if re.search(rf"(?:^|[_-]){re.escape(label.upper())}$", stem):
            return label
    return None


def event_reference(markers: pd.DataFrame, start: float, csv_path: str | None) -> tuple[np.ndarray, dict]:
    """Return board-edge references, preferring the exported DRT CSV when present."""

    on_markers = markers.loc[markers["value"].str.endswith("_ON"), "time"].to_numpy(float)
    if not len(on_markers):
        raise ValueError("no DRT *_ON markers found")

    csv = None
    if csv_path is not None:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"DRT CSV not found: {csv_file}")
        csv = pd.read_csv(csv_file)
        if "OnsetTimeDRT" not in csv:
            raise ValueError(f"DRT CSV lacks OnsetTimeDRT: {csv_file}")

    if csv is None:
        return on_markers, {
            "source": "corrected PsychoPy *_ON marker timestamps (DRT CSV not exported)",
            "csv_path": None,
            "trials_in_csv": None,
            "trials_in_markers": int(len(on_markers)),
            "on_markers_minus_csv_onsets_ms": None,
            "csv_lag_ms": None,
        }

    csv_onsets = start + csv["OnsetTimeDRT"].to_numpy(float) / 1000
    n = min(len(on_markers), len(csv_onsets))
    if n == 0:
        raise ValueError("DRT CSV and *_ON marker stream have no overlapping trials")
    deltas = 1000 * (on_markers[:n] - csv_onsets[:n])
    lag = csv["Lag"] if "Lag" in csv else pd.Series(dtype=float)
    lag_stats = None if lag.empty else {
        "min": float(lag.min()), "median": float(lag.median()), "max": float(lag.max())
    }
    return csv_onsets[:n], {
        "source": "DRTStart + OnsetTimeDRT from DRT CSV",
        "csv_path": str(csv_file),
        "trials_in_csv": int(len(csv_onsets)),
        "trials_in_markers": int(len(on_markers)),
        "on_markers_minus_csv_onsets_ms": np.round(deltas, 3).tolist(),
        "on_marker_minus_csv_onset_stats_ms": {
            "mean": float(np.mean(deltas)), "sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "min": float(np.min(deltas)), "median": float(np.median(deltas)), "max": float(np.max(deltas)),
        },
        "csv_lag_ms": lag_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default="data/DRT_Timing/drt_timing_P7.xdf")
    parser.add_argument("--csv", default=None, help="optional DRT export containing OnsetTimeDRT")
    parser.add_argument("--channel", default=None,
                        help="sensor the stimulator was held against (inferred from the XDF name when omitted)")
    parser.add_argument("--duration", type=float, default=None,
                        help="deprecated compatibility hint; offset timing is detected from the waveform")
    parser.add_argument("--out", type=Path, default=Path("data/DRT_Timing/drt_timing"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rec = load(args.xdf)
    t, x, labels = rec["t"], rec["x"], rec["labels"]
    markers = rec["markers"]
    start = markers.loc[markers["value"] == "DRTStart", "time"].to_numpy()
    assert len(start) == 1, "expected one DRTStart marker"
    on_markers = markers.loc[markers["value"].str.endswith("_ON"), "time"].to_numpy(float)
    onsets, reference = event_reference(markers, start[0], args.csv)
    inferred_channel = channel_from_filename(args.xdf, labels)
    channel = args.channel or inferred_channel
    channel_source = "--channel" if args.channel else ("XDF filename suffix" if inferred_channel else "T8 fallback")
    if channel is None:
        channel = "T8" if "T8" in labels else labels[0]
    if channel not in labels:
        raise ValueError(f"channel {channel!r} is not present; choose from {labels}")
    marker_values = markers.loc[markers["value"].str.endswith("_ON"), "value"].to_numpy()
    miss_markers = markers.loc[markers["value"].str.endswith("_MISS"), "time"].to_numpy(float)
    # Both edges are localized in one onset-locked epoch.  The broad offset
    # search range is only a guard against unrelated later activity; no pulse
    # duration is supplied to the timing estimator.
    epoch_post = EDGE_SEARCH_WINDOWS["offset"][1] + 0.05
    in_rec = onsets[(onsets > t[0] + 0.5) & (onsets + epoch_post < t[-1])]
    if not len(in_rec):
        raise ValueError("no stimulus events have enough recording on both sides for analysis")

    results = {
        "xdf": args.xdf,
        "device": rec["device"], "eeg_clock": rec["clock"], "channel": channel,
        "channel_source": channel_source,
        "event_reference": reference,
        "stimulus_marker_values": sorted(set(map(str, marker_values))),
        "method": {
            "edge_timing": "signed-derivative centroid of the pooled onset-locked waveform",
            "edge_search_windows_s": EDGE_SEARCH_WINDOWS,
            "offset_duration": "measured edge-to-edge from the recorded waveform; no fixed duration used",
        },
        "duration_hint_s": None if args.duration is None else float(args.duration),
        "trials_used": int(len(in_rec)),
        "eeg_rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
        "samples": int(len(t)), "duration_s": float(t[-1] - t[0]),
        "missing_samples": int(np.asarray(rec["bad"]).sum()),
        "marker_counts": {str(k): int(v) for k, v in markers["value"].value_counts().items()},
        "on_marker_intervals_ms": {
            "mean": float(np.mean(np.diff(on_markers) * 1000)) if len(on_markers) > 1 else None,
            "sd": float(np.std(np.diff(on_markers) * 1000, ddof=1)) if len(on_markers) > 2 else None,
            "min": float(np.min(np.diff(on_markers) * 1000)) if len(on_markers) > 1 else None,
            "max": float(np.max(np.diff(on_markers) * 1000)) if len(on_markers) > 1 else None,
        },
    }

    trial_table = pd.DataFrame({"trial": np.arange(1, len(onsets) + 1), "onset_reference_s": onsets})
    if len(on_markers) >= len(onsets):
        trial_table["on_marker_s"] = on_markers[:len(onsets)]
    if len(miss_markers) >= len(onsets):
        trial_table["miss_marker_s"] = miss_markers[:len(onsets)]
    trial_table.to_csv(args.out / "trials.csv", index=False)

    j = labels.index(channel)
    others = [k for k in range(len(labels)) if k != j]
    signals = {"electrical (mean of other channels)": x[:, others].mean(axis=1), f"{channel}": x[:, j]}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    feats = {}
    electrical_boot = None
    edge_windows = [("onset", EDGE_SEARCH_WINDOWS["onset"]), ("offset", EDGE_SEARCH_WINDOWS["offset"])]
    for row, (name, sig) in enumerate(signals.items()):
        rel, val, trial = pooled(t, sig, rec["bad"], in_rec, pre=0.35, post=epoch_post)
        boot = bootstrap_edge_pair(rel, val, trial, args.bootstrap, rng)
        if row == 0:
            electrical_boot = boot
        for col, (edge, search_window) in enumerate(edge_windows):
            f = edge_features(rel, val, search_window)
            f["edge_ci95_s"] = boot[f"{edge}_edge_ci95_s"]
            shift_window = SHIFT_WINDOW if edge == "onset" else (0.150, 0.230)
            shifts = trial_shifts(rel, val, trial, shift_window) if row == 0 else None
            if shifts is not None:
                f["trial_shift_sd_ms"] = float(1000 * shifts.std(ddof=1))
                f["trial_shift_range_ms"] = [float(1000 * shifts.min()), float(1000 * shifts.max())]
                f["_shifts"] = shifts
            feats[(edge, name)] = f
            ax = axes[row, col]
            ax.scatter(1000 * rel, val, s=5, c=trial, cmap="viridis", alpha=0.6)
            ax.plot(1000 * EDGE_GRID, f["_tpl"], color="k", lw=1.6)
            ax.axvline(0, color="tab:red", lw=1)
            ax.axvline(1000 * f["edge_s"], color="k", ls="--", lw=0.8)
            ax.set_xlim(-60, 320)
            ymin, ymax = np.percentile(val[(rel > -0.06) & (rel < 0.35)], [0.5, 99.5])
            ax.set_ylim(ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))
            ci = f["edge_ci95_s"]
            ax.set_title(f"{edge} edge, {name}: {1000 * f['edge_s']:.1f} ms "
                         f"[{1000 * ci[0]:.1f}, {1000 * ci[1]:.1f}]", fontsize=9)
            ax.set_xlabel("ms from DRT *_ON marker")
            ax.set_ylabel("µV")
            ax.grid(alpha=0.3)
    ax = axes[0, 2]
    for edge, color in [("onset", "tab:blue"), ("offset", "tab:orange")]:
        s = feats[(edge, "electrical (mean of other channels)")]["_shifts"]
        ax.hist(1000 * s, bins=np.arange(-6, 6.01, 0.5), alpha=0.6, color=color, label=f"{edge} (SD {1000 * s.std(ddof=1):.1f} ms)")
    ax.set_xlabel("single-trial shift vs template (ms)")
    ax.set_title("Trial-to-trial timing jitter (electrical transient)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    k = len(in_rec) // 2
    a, b = np.searchsorted(t, in_rec[k] - 0.5), np.searchsorted(t, in_rec[k] + epoch_post + 0.2)
    for i, ch in enumerate(labels):
        y = x[a:b, i] - np.median(x[a:b, i])
        ax.plot(t[a:b] - in_rec[k], y / (np.ptp(y) + 1e-9) - i, lw=0.6, color="tab:red" if i == j else "k")
        ax.text(-0.52, -i, ch, ha="right", va="center", fontsize=6)
    on_edge = feats[("onset", "electrical (mean of other channels)")]["edge_s"]
    off_edge = feats[("offset", "electrical (mean of other channels)")]["edge_s"]
    for edge_t in (on_edge, off_edge):
        ax.axvline(edge_t, color="tab:red", lw=0.8)
    ax.set_yticks([])
    ax.set_title(f"trial {k + 1}, all channels (each scaled to its range)", fontsize=9)
    ax.set_xlabel("s from DRT *_ON marker")
    fig.suptitle(f"{rec['device']} DRT timing: {len(in_rec)} trials, stimulator on {channel}; "
                 f"measured pulse width {1000 * (off_edge - on_edge):.1f} ms", fontsize=11)
    fig.savefig(args.out / "drt_timing.png", dpi=130)
    plt.close(fig)

    results["features"] = {
        f"{edge} / {name}": {k: v for k, v in f.items() if not k.startswith("_")} for (edge, name), f in feats.items()}
    on = feats[("onset", "electrical (mean of other channels)")]
    off = feats[("offset", "electrical (mean of other channels)")]
    # Preserve the per-trial onset alignment used for the jitter histogram.  It
    # is useful independently of the full diagnostic figure and lets later
    # plots show the absolute marker-to-signal-onset distribution.
    if "_shifts" in on:
        pd.DataFrame({"onset_shift_s": on["_shifts"]}).to_csv(
            args.out / "onset_shifts.csv", index=False
        )
    pulse_width = off["edge_s"] - on["edge_s"]
    chain_from_offset = off["edge_s"] - pulse_width
    results["eeg_chain_latency_s"] = {
        # Keep the old field names as aliases, but make their edge-centroid
        # meaning explicit in the new fields below.
        "onset_peak": on["edge_s"], "offset_peak": off["edge_s"],
        "onset_edge_s": on["edge_s"], "offset_edge_s": off["edge_s"],
        "measured_pulse_width_s": pulse_width,
        "offset_latency_corrected_s": chain_from_offset,
        "mean_of_edges": (on["edge_s"] + chain_from_offset) / 2,
        "onset_edge_ci95_s": on["edge_ci95_s"],
        "offset_edge_ci95_s": off["edge_ci95_s"],
        "pulse_width_ci95_s": electrical_boot["pulse_width_ci95_s"],
    }
    (args.out / "results.json").write_text(json.dumps(em.to_jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps(em.to_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
