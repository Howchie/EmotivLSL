#!/usr/bin/env python3
"""EEG-chain latency from a DRT tactile stimulator held against one sensor.

The DRT board drives a vibration motor on a millisecond clock that PsychoPy
zeroes with ``resettimer`` (the ``DRTStart`` marker).  Switching the motor on
and off couples a sharp electrical transient into the headset.  The time from
the board's stimulus edge to that transient in the recorded EEG timestamps is
the whole EEG-chain latency: onboard filtering, radio, dongle, USB and reader.
That is exactly the offset between EEG timestamps and real time that ERP
analyses need to remove.

Stimulus times are ``DRTStart + OnsetTimeDRT`` from the DRT CSV (every trial,
including the last, whose ``_ON`` marker is pushed after the trial ends); the
``_ON`` markers are checked against them.  Each trial samples the transient at
a different phase of the 128 Hz sample clock, so pooling trials by exact time
from the stimulus reconstructs the transient at sub-millisecond resolution.
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

import epocx_sanity as ex
import flex_sanity as fx

SEED = 7
BASELINE = (-0.30, -0.02)  # s before the stimulus edge
GRID = np.arange(-0.060, 0.200, 0.00025)
KERNEL_S = 0.001
SPIKE_WINDOW = (0.040, 0.120)  # where the main transient is searched for
SHIFT_WINDOW = (0.040, 0.110)  # used to align single trials to the template


def load(xdf: str) -> dict:
    streams, _ = pyxdf.load_xdf(xdf, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    if by_name("Epoc X") and len(by_name("Epoc X")[0]["time_stamps"]):
        eeg = by_name("Epoc X")[0]
        grid = ex.epocx_grid(eeg, by_name("Epoc X Packet Diagnostics")[0])
        device, t, x, bad = "EPOC X", grid["t"], grid["x"], grid["missing"]
        clock = "arrival times fitted against the packet counter"
    else:
        eeg = by_name("Epoc Flex 1.0")[0]
        diag = by_name("Epoc Flex 1.0 Packet Diagnostics")[0]
        names = fx._labels(diag)
        device, t, x = "Flex 1.0", np.asarray(eeg["time_stamps"], float), np.asarray(eeg["time_series"], float)
        bad = np.asarray(diag["time_series"], float)[:, names.index("FILLED")] > 0
        clock = "reader counter-clock timestamps"
    markers = [(float(ts), str(v[0])) for s in by_name("PsychoPy Markers") for ts, v in zip(s["time_stamps"], s["time_series"])]
    return {"device": device, "clock": clock, "t": t, "x": x, "bad": bad, "labels": fx._labels(eeg),
            "markers": pd.DataFrame(sorted(set(markers)), columns=["time", "value"])}


def pooled(t, signal, bad, events, pre=0.35, post=0.25):
    """Baseline-corrected samples around each event, with their exact times from the event."""

    rel, val, trial = [], [], []
    for k, e in enumerate(events):
        a, b = np.searchsorted(t, e - pre), np.searchsorted(t, e + post)
        if a < 1 or b >= len(t) or bad[a:b].any():
            continue
        r = t[a:b] - e
        v = signal[a:b] - signal[a:b][(r >= BASELINE[0]) & (r <= BASELINE[1])].mean()
        rel.append(r)
        val.append(v)
        trial.append(np.full(len(r), k))
    return np.concatenate(rel), np.concatenate(val), np.concatenate(trial)


def kernel_template(rel, val, grid=GRID):
    w = np.exp(-0.5 * ((rel[None, :] - grid[:, None]) / KERNEL_S) ** 2)
    return (w @ val) / w.sum(axis=1)


def spike_features(rel, val, sign: int) -> dict:
    tpl = kernel_template(rel, val)
    base = val[(rel >= BASELINE[0]) & (rel <= BASELINE[1])]
    sd = base.std()
    sel = (GRID >= SPIKE_WINDOW[0]) & (GRID <= SPIKE_WINDOW[1])
    peak = GRID[sel][np.argmax(sign * tpl[sel])]
    # First departure from baseline (either sign) before the peak: > 3 baseline SDs of
    # the single samples, sustained for 2 ms of the template.
    over = np.abs(tpl) > 3 * sd
    sustained = np.convolve(over.astype(int), np.ones(8, int), mode="same") >= 8
    early = GRID[(GRID > 0) & (GRID < peak) & sustained]
    return {"peak_s": float(peak), "peak_amp": float(tpl[GRID == peak][0]),
            "first_departure_s": float(early.min()) if len(early) else None, "baseline_sd": float(sd), "_tpl": tpl}


def bootstrap(rel, val, trial, sign, n_boot, rng) -> dict:
    ids = np.unique(trial)
    by_trial = {k: np.flatnonzero(trial == k) for k in ids}
    peaks, firsts = [], []
    for _ in range(n_boot):
        pick = np.concatenate([by_trial[k] for k in rng.choice(ids, len(ids))])
        f = spike_features(rel[pick], val[pick], sign)
        peaks.append(f["peak_s"])
        if f["first_departure_s"] is not None:
            firsts.append(f["first_departure_s"])
    return {"peak_ci95_s": np.percentile(peaks, [2.5, 97.5]).tolist(),
            "first_departure_ci95_s": np.percentile(firsts, [2.5, 97.5]).tolist() if firsts else None}


def trial_shifts(rel, val, trial) -> np.ndarray:
    """Per-trial time shift against a leave-one-out template (least squares, free gain)."""

    shifts = np.arange(-0.012, 0.012001, 0.00025)
    out = []
    for k in np.unique(trial):
        mine = trial == k
        others = ~mine
        r, v = rel[mine], val[mine]
        sel = (r >= SHIFT_WINDOW[0]) & (r <= SHIFT_WINDOW[1])
        if sel.sum() < 5:
            continue
        sse = []
        for s in shifts:
            tpl = kernel_template(rel[others], val[others], grid=r[sel] - s)
            gain = (tpl @ v[sel]) / (tpl @ tpl)
            sse.append(np.sum((v[sel] - gain * tpl) ** 2))
        out.append(shifts[int(np.argmin(sse))])
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default="data/DRT_Timing/drt.xdf")
    parser.add_argument("--csv", default="data/DRT_Timing/drt.csv")
    parser.add_argument("--channel", default="T8", help="sensor the stimulator was held against")
    parser.add_argument("--duration", type=float, default=1.0, help="stimulus duration (s) for the offset edge")
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
    csv = pd.read_csv(args.csv)
    onsets = start[0] + csv["OnsetTimeDRT"].to_numpy() / 1000
    on_markers = markers.loc[markers["value"].str.endswith("_ON"), "time"].to_numpy()
    in_rec = onsets[(onsets > t[0] + 0.5) & (onsets + args.duration + 0.3 < t[-1])]

    results = {
        "device": rec["device"], "eeg_clock": rec["clock"], "channel": args.channel,
        "trials_in_csv": int(len(onsets)), "trials_used": int(len(in_rec)),
        "on_markers_minus_csv_onsets_ms": np.round(1000 * (on_markers - onsets[:len(on_markers)]), 3).tolist(),
        "csv_lag_ms": {"min": int(csv["Lag"].min()), "median": float(csv["Lag"].median()), "max": int(csv["Lag"].max())},
        "eeg_rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
    }

    j = labels.index(args.channel)
    others = [k for k in range(len(labels)) if k != j]
    signals = {"electrical (mean of other channels)": x[:, others].mean(axis=1), f"{args.channel}": x[:, j]}
    edges = {"onset": in_rec, "offset": in_rec + args.duration}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    feats = {}
    for col, (edge, events) in enumerate(edges.items()):
        for row, (name, sig) in enumerate(signals.items()):
            rel, val, trial = pooled(t, sig, rec["bad"], events)
            # Sign of the main spike: whichever extreme is larger in the onset template.
            probe = spike_features(rel, val, +1)
            probe_neg = spike_features(rel, val, -1)
            sign = +1 if abs(probe["peak_amp"]) >= abs(probe_neg["peak_amp"]) else -1
            f = probe if sign > 0 else probe_neg
            f.update(bootstrap(rel, val, trial, sign, args.bootstrap, rng))
            shifts = trial_shifts(rel, val, trial) if row == 0 else None
            if shifts is not None:
                f["trial_shift_sd_ms"] = float(1000 * shifts.std(ddof=1))
                f["trial_shift_range_ms"] = [float(1000 * shifts.min()), float(1000 * shifts.max())]
                f["_shifts"] = shifts
            f["sign"] = sign
            feats[(edge, name)] = f
            ax = axes[row, col]
            ax.scatter(1000 * rel, val, s=5, c=trial, cmap="viridis", alpha=0.6)
            ax.plot(1000 * GRID, f["_tpl"], color="k", lw=1.6)
            ax.axvline(0, color="tab:red", lw=1)
            ax.axvline(1000 * f["peak_s"], color="k", ls="--", lw=0.8)
            if f["first_departure_s"] is not None:
                ax.axvline(1000 * f["first_departure_s"], color="0.5", ls=":", lw=0.8)
            ax.set_xlim(-60, 200)
            ymin, ymax = np.percentile(val[(rel > -0.06) & (rel < 0.2)], [0.5, 99.5])
            ax.set_ylim(ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))
            ax.set_title(f"{edge}, {name}: peak {1000 * f['peak_s']:.1f} ms "
                         f"[{1000 * f['peak_ci95_s'][0]:.1f}, {1000 * f['peak_ci95_s'][1]:.1f}]", fontsize=9)
            ax.set_xlabel("ms from DRT stimulus edge (board clock on LSL time)")
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
    a, b = np.searchsorted(t, in_rec[k] - 0.5), np.searchsorted(t, in_rec[k] + args.duration + 0.5)
    for i, ch in enumerate(labels):
        y = x[a:b, i] - np.median(x[a:b, i])
        ax.plot(t[a:b] - in_rec[k], y / (np.ptp(y) + 1e-9) - i, lw=0.6, color="tab:red" if i == j else "k")
        ax.text(-0.52, -i, ch, ha="right", va="center", fontsize=6)
    for edge_t in (0, args.duration):
        ax.axvline(edge_t, color="tab:red", lw=0.8)
    ax.set_yticks([])
    ax.set_title(f"trial {k + 1}, all channels (each scaled to its range)", fontsize=9)
    ax.set_xlabel("s from onset")
    fig.suptitle(f"{rec['device']} DRT timing: {len(in_rec)} trials, stimulator on {args.channel}", fontsize=11)
    fig.savefig(args.out / "drt_timing.png", dpi=130)
    plt.close(fig)

    results["features"] = {
        f"{edge} / {name}": {k: v for k, v in f.items() if not k.startswith("_")} for (edge, name), f in feats.items()}
    on = feats[("onset", "electrical (mean of other channels)")]
    off = feats[("offset", "electrical (mean of other channels)")]
    results["eeg_chain_latency_s"] = {
        "onset_peak": on["peak_s"], "offset_peak": off["peak_s"],
        "mean_of_edges": (on["peak_s"] + off["peak_s"]) / 2,
        "earliest_departure": min(v for v in (on["first_departure_s"], off["first_departure_s"]) if v is not None),
    }
    (args.out / "results.json").write_text(json.dumps(fx.to_jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps(fx.to_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
