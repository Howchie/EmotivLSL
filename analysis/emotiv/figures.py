"""Standard two-condition ERP figures, shared by every task script.

One layout serves any paired contrast -- a Go/NoGo P3, an oddball MMN -- so the
axes and the shading mean the same thing in every report.

**The figures carry no numbers.**  A panel is labelled with its ROI and the
electrodes it was measured from, nothing else: effect sizes, CIs, peak latencies,
FWHMs and cluster p-values are printed by the task script and stored in
``results.json``, which is where they can be read exactly rather than squinted at.
The only shading is the measurement window, which says which part of the axis the
summary came from without asserting anything about it.

Blocks are never pooled in ``block_figure``.  A session can change task mid-way,
and a pooled waveform is then a fair summary of neither half.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["COLORS", "contrast_figure", "block_figure"]

COLORS = {"test": "#b2182b", "ref": "#2166ac", "diff": "black"}


def _labels(res: dict) -> tuple[str, str]:
    test, ref = res["conditions"]
    return test.capitalize(), ref.capitalize()


def _plot_pair(ax, res: dict) -> None:
    t = 1000 * res["_times"]
    test, ref = _labels(res)
    for key, label, colour, n in (("_test", test, COLORS["test"], res["n_test"]),
                                  ("_ref", ref, COLORS["ref"], res["n_ref"])):
        mean, lo, hi = res[key]
        ax.plot(t, mean, color=colour, lw=1.6, label=f"{label} (n={n})")
        ax.fill_between(t, lo, hi, color=colour, alpha=0.18, lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.legend(fontsize=8, framealpha=0.9)


def _plot_diff(ax, res: dict, windows) -> None:
    t = 1000 * res["_times"]
    diff, lo, hi = res["_diff"]
    ax.plot(t, diff, color=COLORS["diff"], lw=1.8)
    ax.fill_between(t, lo, hi, color=COLORS["diff"], alpha=0.18, lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    for w in windows:
        ax.axvspan(1000 * w[0], 1000 * w[1], color="tab:green", alpha=0.08)
    ax.set_xlabel("ms after the sound")


def contrast_figure(result: dict, rois: dict, path: Path, title: str) -> None:
    """One column per ROI: the two conditions above, their difference below."""

    n = len(rois)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.0), squeeze=False, sharex=True)
    windows = result.get("windows_s") or [result["window_s"]]
    for col, (name, res) in enumerate(rois.items()):
        ax = axes[0, col]
        _plot_pair(ax, res)
        ax.set_title(f"{name}: {' '.join(res['roi'])}", fontsize=10)
        ax.set_ylabel("µV" if col == 0 else "")

        ax = axes[1, col]
        _plot_diff(ax, res, windows)
        test, ref = _labels(res)
        ax.set_title(f"{test} − {ref}", fontsize=10)
        ax.set_ylabel("µV" if col == 0 else "")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def block_figure(result: dict, blocks: dict, path: Path, title: str) -> None:
    """One column per block, so a session that changed task mid-way can be read."""

    n = len(blocks)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.0), squeeze=False,
                             sharey="row", sharex=True)
    windows = result.get("windows_s") or [result["window_s"]]
    for col, res in enumerate(blocks.values()):
        kind = {True: "button", False: "silent counting, no response",
                None: None}[res["responding"]]
        ax = axes[0, col]
        _plot_pair(ax, res)
        ax.set_title(f"block {res['block']}" + (f" — {kind}" if kind else ""), fontsize=10)
        ax.set_ylabel("µV" if col == 0 else "")

        ax = axes[1, col]
        _plot_diff(ax, res, windows)
        test, ref = _labels(res)
        ax.set_title(f"{test} − {ref}", fontsize=10)
        ax.set_ylabel("µV" if col == 0 else "")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
