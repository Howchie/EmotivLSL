#!/usr/bin/env python3
"""Plot the absolute marker-to-signal-onset distributions for both timing tests.

The DRT run's source XDF is not part of the repository, but its onset-shift
histogram is retained in ``drt_timing.png``.  The bin counts below are that
histogram's onset series, translated by the measured DRT edge latency.  When a
rerun of :mod:`drt_timing` has produced ``onset_shifts.csv``, those per-trial
values are preferred automatically.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DRT_RESULTS = Path("data/DRT_Timing/drt_timing/results.json")
DRT_SHIFTS = Path("data/DRT_Timing/drt_timing/onset_shifts.csv")
HEADPHONE_TRIALS = Path("data/Headphone_Timing/headphone_timing/trials.csv")
DEFAULT_OUT = Path("data/timing_onset_histogram.png")

# Counts read from the onset (blue) series in the checked-in DRT diagnostic.
# Edges are the same 0.5-ms edges used by analysis/drt_timing.py.
DRT_SHIFT_EDGES_MS = np.arange(-6.0, 6.5, 0.5)
DRT_SHIFT_COUNTS = np.array(
    [0, 0, 2, 2, 4, 1, 6, 5, 9, 13, 23, 23,
     21, 31, 19, 20, 17, 5, 4, 1, 1, 0, 0, 0],
    dtype=int,
)


def load_drt(results_path: Path, shifts_path: Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    latency_ms = 1000.0 * results["eeg_chain_latency_s"]["onset_edge_s"]
    jitter_ms = results["features"]["onset / electrical (mean of other channels)"][
        "trial_shift_sd_ms"
    ]

    if shifts_path.exists():
        shifts = pd.read_csv(shifts_path)["onset_shift_s"].dropna().to_numpy(float)
        values_ms = latency_ms + 1000.0 * shifts
        source = "per-trial onset shifts"
    else:
        values_ms = None
        source = "onset histogram retained in drt_timing.png"

    return {
        "label": "DRT_timing",
        "mean_ms": latency_ms,
        "sd_ms": float(jitter_ms),
        "n": int(DRT_SHIFT_COUNTS.sum()) if values_ms is None else int(values_ms.size),
        "values_ms": values_ms,
        "source": source,
    }


def load_headphone(trials_path: Path) -> dict:
    frame = pd.read_csv(trials_path)
    if "contact" in frame:
        contact = frame["contact"].astype(str).str.lower().eq("true")
        frame = frame.loc[contact]
    values_ms = 1000.0 * frame["sound_onset_s"].dropna().to_numpy(float)
    return {
        "label": "Headphone_timing",
        "mean_ms": float(values_ms.mean()),
        "sd_ms": float(values_ms.std(ddof=1)),
        "n": int(values_ms.size),
        "values_ms": values_ms,
        "source": "trials.csv: sound_onset_s",
    }


def draw_histogram(ax, timing: dict, color: str, bins_ms: np.ndarray) -> None:
    mean = timing["mean_ms"]
    sd = timing["sd_ms"]
    ax.axvspan(mean - sd, mean + sd, color=color, alpha=0.15, lw=0, zorder=0)

    if timing["values_ms"] is None:
        # Draw the retained DRT bins directly so no artificial observations are
        # introduced when the original source recording is unavailable.
        edges = mean + DRT_SHIFT_EDGES_MS
        ax.bar(
            edges[:-1],
            DRT_SHIFT_COUNTS,
            width=np.diff(edges),
            align="edge",
            color=color,
            alpha=0.78,
            linewidth=0,
            zorder=2,
        )
    else:
        ax.hist(timing["values_ms"], bins=bins_ms, color=color, alpha=0.78,
                linewidth=0, zorder=2)

    ax.axvline(mean, color="0.15", ls="--", lw=1.2, zorder=3)
    handles = [
        Patch(facecolor=color, edgecolor="none", alpha=0.78,
              label=f"{timing['label']} (n={timing['n']})"),
        Line2D([], [], color="0.15", ls="--", lw=1.2,
               label=f"mean {mean:.1f} ms; SD {sd:.1f} ms"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)


def plot(drt: dict, headphone: dict, out: Path) -> None:
    bins_ms = np.arange(60.0, 90.5, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True, sharey=True)
    draw_histogram(axes[0], drt, "tab:blue", bins_ms)
    draw_histogram(axes[1], headphone, "tab:orange", bins_ms)

    for ax in axes:
        ax.set_xlim(60, 90)
        ax.set_xticks(np.arange(60, 91, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3)
    axes[0].set_ylabel("trials")
    fig.supxlabel("signal onset from marker (ms)")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.98, wspace=0.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drt-results", type=Path, default=DRT_RESULTS)
    parser.add_argument("--drt-shifts", type=Path, default=DRT_SHIFTS)
    parser.add_argument("--headphone-trials", type=Path, default=HEADPHONE_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    drt = load_drt(args.drt_results, args.drt_shifts)
    headphone = load_headphone(args.headphone_trials)
    plot(drt, headphone, args.out)
    print(f"DRT_timing: {drt['mean_ms']:.3f} ± {drt['sd_ms']:.3f} ms (n={drt['n']}; {drt['source']})")
    print(f"Headphone_timing: {headphone['mean_ms']:.3f} ± {headphone['sd_ms']:.3f} ms (n={headphone['n']})")
    print(args.out)


if __name__ == "__main__":
    main()
