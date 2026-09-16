#!/usr/bin/env python3
"""Analyze the two new EPOC X timing/loopback recordings.

The recordings use the same EPOC X timestamp reconstruction as
``analysis/drt_timing.py``.  ``drt-timing.xdf`` has an external DRT CSV, so the
board-clock onset is used as the event time and the [LED.H] marker is checked
against it.  The LED OFF edge is not represented by a separate LSL marker; its
time is estimated from the opposite-polarity transient and the measured pulse
width.  ``marker-timing.xdf`` is the 100-ms audio-jack loopback run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, sosfiltfilt

import drt_timing as dt
import emotiv as em

SEED = 7
FS = 128.0
LABELS = list(em.EPOCX.labels)


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _bootstrap_mean(values: np.ndarray, n: int, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, float)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def load_xdf(path: str) -> dict:
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    eeg = by_name("Epoc X")[0]
    grid = em.epocx_grid(eeg, by_name("Epoc X Packet Diagnostics")[0])
    rows = sorted(
        (float(ts), str(value[0]))
        for stream in by_name("PsychoPy Markers")
        for ts, value in zip(stream["time_stamps"], stream["time_series"])
    )
    return {"streams": streams, "by_name": by_name, "grid": grid, "markers": rows}


def marker_intervals(markers: list[tuple[float, str]], value_or_prefix: str) -> np.ndarray:
    values = np.array(
        [ts for ts, value in markers if value == value_or_prefix or value.startswith(value_or_prefix)],
        float,
    )
    return np.diff(values) * 1000 if len(values) > 1 else np.array([], float)


def template_feature(
    t: np.ndarray,
    x: np.ndarray,
    bad: np.ndarray,
    events: np.ndarray,
    signal: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    rel, val, trial = dt.pooled(t, signal, bad, events, pre=0.35, post=0.25)
    pos = dt.spike_features(rel, val, +1)
    neg = dt.spike_features(rel, val, -1)
    sign = 1 if abs(pos["peak_amp"]) >= abs(neg["peak_amp"]) else -1
    feature = pos if sign > 0 else neg
    boot = dt.bootstrap(rel, val, trial, sign, n_boot, rng)
    shifts = dt.trial_shifts(rel, val, trial)
    peak_i = int(np.argmin(np.abs(dt.GRID - feature["peak_s"])))
    pre = np.flatnonzero((dt.GRID >= feature["peak_s"] - 0.040) & (dt.GRID <= feature["peak_s"] - 0.005))
    post = np.flatnonzero((dt.GRID >= feature["peak_s"] + 0.005) & (dt.GRID <= feature["peak_s"] + 0.040))
    pre_i = pre[np.argmax(-sign * feature["_tpl"][pre])]
    post_i = post[np.argmax(-sign * feature["_tpl"][post])]
    return {
        "n_trials": int(len(np.unique(trial))),
        "peak_ms": 1000 * float(feature["peak_s"]),
        "peak_amp_uv": float(feature["peak_amp"]),
        "peak_ci95_ms": (1000 * np.asarray(boot["peak_ci95_s"])).tolist(),
        "first_departure_ms": 1000 * float(feature["first_departure_s"])
        if feature["first_departure_s"] is not None
        else None,
        "baseline_sd_uv": float(feature["baseline_sd"]),
        "sign": int(sign),
        "trial_shift_sd_ms": float(1000 * shifts.std(ddof=1)),
        "trial_shift_range_ms": (1000 * np.asarray([shifts.min(), shifts.max()])).tolist(),
        "opposite_lobe_before_ms": 1000 * float(dt.GRID[pre_i]),
        "opposite_lobe_after_ms": 1000 * float(dt.GRID[post_i]),
        "rel": rel,
        "val": val,
        "trial": trial,
        "template": feature["_tpl"],
    }


def channel_peaks(t: np.ndarray, x: np.ndarray, bad: np.ndarray, events: np.ndarray) -> dict:
    out = {}
    for j, label in enumerate(LABELS):
        rel, val, _ = dt.pooled(t, x[:, j], bad, events, pre=0.35, post=0.25)
        pos = dt.spike_features(rel, val, +1)
        neg = dt.spike_features(rel, val, -1)
        f = pos if abs(pos["peak_amp"]) >= abs(neg["peak_amp"]) else neg
        out[label] = {"peak_ms": 1000 * float(f["peak_s"]), "amp_uv": float(f["peak_amp"]),
                      "sign": 1 if f is pos else -1}
    return out


def edge_peaks(t: np.ndarray, signal: np.ndarray, events: np.ndarray) -> dict:
    """Per-trial raw edge peaks; the opposite edges also estimate pulse width."""

    hp = sosfiltfilt(butter(4, 5, btype="high", fs=FS, output="sos"), signal)
    dt_sample = float(np.median(np.diff(t)))

    def one(base: float, sign: float) -> float:
        a, b = np.searchsorted(t, base + 0.03), np.searchsorted(t, base + 0.13)
        r, y = t[a:b] - base, sign * hp[a:b]
        k = int(np.argmax(y))
        if 0 < k < len(y) - 1:
            q = y[k - 1:k + 2]
            den = q[0] - 2 * q[1] + q[2]
            frac = 0.5 * (q[0] - q[2]) / den if abs(den) > 1e-12 else 0.0
        else:
            frac = 0.0
        return float(r[k] + frac * dt_sample)

    onset = np.array([one(e, +1) for e in events])
    offset_assumed = np.array([one(e + 1.0, -1) for e in events])
    width = offset_assumed - onset
    # Width is measured from the two edge centers, so corrected OFF latency is
    # offset relative to ON+1 s minus the excess/shortfall in pulse duration.
    off_corrected = offset_assumed - (width.mean())
    return {
        "onset_peak_ms": _stats(onset * 1000),
        "offset_peak_after_on_plus_1s_ms": _stats(offset_assumed * 1000),
        "pulse_width_ms": _stats((1000 + width * 1000)),
        "offset_latency_corrected_ms": _stats(off_corrected * 1000),
        "onset_peak_ci95_ms": _bootstrap_mean(onset * 1000, 10000, np.random.default_rng(SEED)),
        "offset_latency_corrected_ci95_ms": _bootstrap_mean(off_corrected * 1000, 10000, np.random.default_rng(SEED + 1)),
        "onset": onset,
        "offset_assumed": offset_assumed,
        "width": width,
    }


def jack_edges(t: np.ndarray, x: np.ndarray, events: np.ndarray, channel: str = "T8") -> dict:
    """Energy-centroid estimate for the audio loopback edges."""

    j = LABELS.index(channel)
    sig = sosfiltfilt(butter(4, 5, btype="high", fs=FS, output="sos"), x[:, j])
    rows = []
    for event in events:
        a, b = np.searchsorted(t, event - 0.3), np.searchsorted(t, event + 0.5)
        r, power = t[a:b] - event, sig[a:b] ** 2
        noise = power[r < -0.02].mean()
        sel = (r > 0.03) & (r < 0.33)
        w = np.clip(power - noise, 0, None) * sel
        centre = np.sum(r * w) / w.sum()
        left, right = w * (r < centre), w * (r >= centre)
        rows.append((np.sum(r * left) / left.sum(), np.sum(r * right) / right.sum(), np.sqrt(power[sel].max())))
    frame = pd.DataFrame(rows, columns=["onset_burst", "offset_burst", "peak_uv"])
    contact = frame["peak_uv"] > 0.25 * frame["peak_uv"].median()
    good = frame.loc[contact]
    onset = good["onset_burst"].to_numpy() - 0.0025
    offset = good["offset_burst"].to_numpy() + 0.0025 - 0.100
    return {
        "trials": int(len(frame)),
        "trials_with_contact": int(len(good)),
        "onset_burst_after_marker_ms": _stats(good["onset_burst"].to_numpy() * 1000),
        "offset_burst_after_marker_ms": _stats(good["offset_burst"].to_numpy() * 1000),
        "sound_onset_after_marker_ms": _stats(onset * 1000),
        "sound_onset_ci95_ms": _bootstrap_mean(onset * 1000, 10000, np.random.default_rng(SEED + 2)),
        "sound_onset_from_offset_ci95_ms": _bootstrap_mean(offset * 1000, 10000, np.random.default_rng(SEED + 3)),
        "edge_spacing_ms": _stats((good["offset_burst"] - good["onset_burst"]).to_numpy() * 1000),
        "onset": onset,
        "offset": offset,
        "frame": frame,
    }


def analyze_drt(path: str, csv_path: str, n_boot: int, rng: np.random.Generator) -> tuple[dict, dict]:
    rec = load_xdf(path)
    g, markers = rec["grid"], rec["markers"]
    t, x, bad = g["t"], g["x"], g["missing"]
    start = np.array([ts for ts, value in markers if value == "DRTStart"])
    on_markers = np.array([ts for ts, value in markers if value == "[LED.H]_ON"])
    miss_markers = np.array([ts for ts, value in markers if value == "[LED.H]_MISS"])
    csv = pd.read_csv(csv_path)
    events = start[0] + csv["OnsetTimeDRT"].to_numpy()[:len(on_markers)] / 1000
    keep = (events > t[0] + 0.5) & (events + 1.25 < t[-1])
    events = events[keep]
    other = x[:, [j for j, label in enumerate(LABELS) if label != "T8"]].mean(axis=1)
    on = template_feature(t, x, bad, events, other, n_boot, rng)
    off = template_feature(t, x, bad, events + 1.0, other, n_boot, rng)
    edge = edge_peaks(t, other, events)
    on_clean = {k: v for k, v in on.items() if k not in ("rel", "val", "trial", "template")}
    off_clean = {k: v for k, v in off.items() if k not in ("rel", "val", "trial", "template")}
    marker_minus_csv = (on_markers[:len(events)] - events) * 1000
    miss_minus_end = (miss_markers[:len(events)] - (start[0] + csv["EndTime"].to_numpy()[:len(events)] / 1000)) * 1000
    out = {
        "file": path,
        "channel_used_for_latency": "mean of 13 channels other than T8",
        "samples": int(len(t)), "duration_s": float(t[-1] - t[0]),
        "missing_samples": int(bad.sum()), "eeg_rate_hz": float(1 / np.median(np.diff(t))),
        "trials": int(len(events)),
        "marker_intervals_ms": _stats(marker_intervals(markers, "[LED.H]_ON")),
        "on_marker_minus_csv_onset_ms": _stats(marker_minus_csv),
        "miss_marker_minus_csv_end_ms": _stats(miss_minus_end),
        "onset_template": on_clean, "offset_template_assuming_1s": off_clean,
        "edge_peak_duration_fit": {k: v for k, v in edge.items() if k not in ("onset", "offset_assumed", "width")},
        "channel_peaks_onset": channel_peaks(t, x, bad, events),
        "channel_peaks_offset": channel_peaks(t, x, bad, events + 1.0),
        "eeg_chain_latency_ms": {
            "onset_template": on["peak_ms"],
            "offset_template_corrected_for_measured_width": off["peak_ms"] - (edge["pulse_width_ms"]["mean"] - 1000),
            "mean_of_corrected_edges": float((on["peak_ms"] + off["peak_ms"] - (edge["pulse_width_ms"]["mean"] - 1000)) / 2),
            "onset_raw_peak": edge["onset_peak_ms"]["mean"],
        },
        "_plot": {"t": t, "x": x, "events": events, "on": on, "off": off, "edge": edge},
    }
    return out, rec


def analyze_marker(path: str, eeg_latency_ms: float, n_boot: int, rng: np.random.Generator) -> tuple[dict, dict]:
    rec = load_xdf(path)
    g, markers = rec["grid"], rec["markers"]
    t, x, bad = g["t"], g["x"], g["missing"]
    events = np.array([ts for ts, value in markers if value.startswith("Oddball-")])
    jack = jack_edges(t, x, events, "T8")
    audio_onset = jack["sound_onset_after_marker_ms"]["mean"] - eeg_latency_ms
    out = {
        "file": path, "channel": "T8", "stimulus_duration_assumed_ms": 100.0,
        "samples": int(len(t)), "duration_s": float(t[-1] - t[0]), "missing_samples": int(bad.sum()),
        "eeg_rate_hz": float(1 / np.median(np.diff(t))), "trials": int(len(events)),
        "marker_intervals_ms": _stats(np.diff(events) * 1000),
        "jack_edges": {k: v for k, v in jack.items() if k not in ("onset", "offset", "frame")},
        "estimated_audio_latency_ms": audio_onset,
        "estimated_audio_latency_ci95_ms": [v - eeg_latency_ms for v in jack["sound_onset_ci95_ms"]],
        "_plot": {"t": t, "x": x, "events": events, "jack": jack},
    }
    return out, rec


def make_figure(marker: dict, drt: dict, old_latency_ms: float, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    eeg_latency_ms = drt["eeg_chain_latency_ms"]["mean_of_corrected_edges"]
    # LED electrical transient, all trials pooled to 0.25-ms resolution.
    for key, color, label in [("on", "tab:blue", "LED ON"), ("off", "tab:orange", "LED OFF (relative to ON+1 s)")]:
        f = drt["_plot"][key]
        axes[0, 0].plot(1000 * dt.GRID, f["template"], color=color, label=label)
        axes[0, 0].axvline(f["peak_ms"], color=color, ls="--", lw=0.8)
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set(xlim=(-20, 150), xlabel="ms from edge reference", ylabel="µV", title="DRT electrical transient")
    axes[0, 0].legend(fontsize=8)
    # Per-channel onset peak latencies.
    p = drt["channel_peaks_onset"]
    axes[0, 1].bar(np.arange(len(LABELS)), [p[c]["peak_ms"] for c in LABELS], color="0.45")
    axes[0, 1].axhline(drt["eeg_chain_latency_ms"]["onset_template"], color="tab:red", ls="--")
    axes[0, 1].set(xticks=np.arange(len(LABELS)), xticklabels=LABELS, ylim=(60, 80), ylabel="ms", title="Onset peak by channel")
    axes[0, 1].tick_params(axis="x", rotation=70, labelsize=7)
    # Marker-timing audio loopback envelope.
    j = marker["_plot"]["jack"]
    rel, val, _ = dt.pooled(marker["_plot"]["t"],
                             marker["_plot"]["x"][:, LABELS.index("T8")],
                             np.zeros(len(marker["_plot"]["t"]), bool), marker["_plot"]["events"],
                             pre=0.3, post=0.45)
    env_grid = np.arange(-0.05, 0.35, 0.0005)
    env = np.sqrt(np.clip(dt.kernel_template(rel, val ** 2, env_grid), 0, None))
    axes[0, 2].plot(1000 * env_grid, env, color="k")
    axes[0, 2].axvline(j["sound_onset_after_marker_ms"]["mean"], color="tab:blue", ls="--", label="audio onset")
    axes[0, 2].axvline(eeg_latency_ms, color="tab:red", ls=":", label="EEG chain")
    axes[0, 2].set(xlim=(-30, 330), xlabel="ms after tone marker", ylabel="RMS pickup", title="T8 audio-jack loopback")
    axes[0, 2].legend(fontsize=8)
    # Trial jitter histograms.
    for edge, color, label in [("on", "tab:blue", "LED ON"), ("off", "tab:orange", "LED OFF")]:
        trial_ids = drt["_plot"][edge]["trial"]
        # Recompute shifts once; this is only for plotting and not in the JSON.
        sh = dt.trial_shifts(drt["_plot"][edge]["rel"], drt["_plot"][edge]["val"], trial_ids)
        axes[1, 0].hist(1000 * sh, bins=np.arange(-5, 5.01, 0.5), alpha=0.55, label=label, color=color)
    axes[1, 0].set(xlabel="template shift (ms)", title="Trial-to-trial electrical timing")
    axes[1, 0].legend(fontsize=8)
    # Latency summary.
    labels = ["old DRT\n(tactile)", "new LED\n(onset)", "new LED\noff, width-corrected"]
    vals = [old_latency_ms, drt["eeg_chain_latency_ms"]["onset_template"], drt["eeg_chain_latency_ms"]["offset_template_corrected_for_measured_width"]]
    axes[1, 1].bar(labels, vals, color=["0.5", "tab:blue", "tab:orange"])
    axes[1, 1].axhline(np.mean(vals), color="k", ls="--", label=f"mean {np.mean(vals):.1f} ms")
    axes[1, 1].set_ylabel("ms from physical edge")
    axes[1, 1].set_title("EEG-chain delay")
    axes[1, 1].legend(fontsize=8)
    # Pulse-width estimate.
    w = drt["_plot"]["edge"]["width"] * 1000 + 1000
    axes[1, 2].hist(w, bins=10, color="tab:green", alpha=0.75)
    axes[1, 2].axvline(w.mean(), color="k", ls="--", label=f"mean {w.mean():.1f} ms")
    axes[1, 2].set(xlabel="LED ON→OFF edge spacing (ms)", title="Measured pulse width")
    axes[1, 2].legend(fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def strip_private(value):
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [strip_private(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/DRT_Timing/loopback_timing"))
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # The prior DRT run is the same EPOC X and the established reference for
    # the chain delay; its mean-of-edges result is in the existing JSON.
    old = json.loads(Path("data/DRT_Timing/drt_timing/results.json").read_text())
    old_latency_ms = 1000 * old["eeg_chain_latency_s"]["mean_of_edges"]
    drt, _ = analyze_drt("data/DRT_Timing/drt-timing.xdf", "data/DRT_Timing/drt-timing.csv", args.bootstrap, rng)
    # Use the mean of the two corrected LED edges as the common-chain reference
    # for the independent audio-jack test.
    marker, _ = analyze_marker("data/DRT_Timing/marker-timing.xdf", drt["eeg_chain_latency_ms"]["mean_of_corrected_edges"], args.bootstrap, rng)
    result = {
        "method": {
            "timestamp_reconstruction": "EPOC X arrival times fitted to packet counter (analysis/emotiv/loading.py)",
            "latency_feature": "pooled 0.25-ms kernel template of the common electrical transient; T8 excluded for LED latency",
            "jitter_feature": "leave-one-trial-out template alignment",
            "filtering_note": "symmetric pre/post ringing is reported as a filter signature, not subtracted from the total delay",
        },
        "prior_drt_reference": {"mean_of_edges_ms": old_latency_ms},
        "drt_timing": strip_private(drt),
        "marker_timing": strip_private(marker),
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(marker, drt, old_latency_ms, args.out / "loopback_timing.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
