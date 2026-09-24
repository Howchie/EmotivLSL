#!/usr/bin/env python3
"""Does the Flex cap hold up over an hour, at the sites we would actually use?

``device_quality.py`` asks which headset is better.  This asks the only question
that decides whether a long Flex session is feasible: **on one wetting, does the
good part of the cap get worse as the minutes pass?**  It is a different question
and it needs different handling, so it is a separate script::

    python analysis/flex_endurance.py

Three decisions define it.

1. **The two Flex recordings are treated as one run.**  ``gng_flex.xdf`` and
   ``oddball_flex.xdf`` were recorded on the same cap and the same wetting, and
   their LSL clocks are the same clock -- the loaders now publish the absolute
   origin as ``run.extra["lsl_t0"]`` -- so every row here carries ``elapsed_min``,
   minutes since the first sample of ``gng_flex``.  The cap is on the head
   continuously, but the recorder is not: there is a **30-minute gap with no
   data** between the end of one file and the start of the other.  Nothing can be
   measured across it, which is why the trend is reported three ways (see below).

2. **The rim is dropped, not measured.**  PO9/PO10/O1/Oz/O2 and the temporal and
   prefrontal ring are excluded by construction: they are badly positioned on
   this cap and their failure is a seating problem, not an ageing one, so
   including them would put a constant offset into every trend and hide what the
   usable electrodes are doing.  ``CORE`` is the nine sites a midline analysis
   uses; ``INNER`` adds the four FC/CP midline neighbours.

3. **Task time only.**  Between-block breaks are self-paced and uncontrolled --
   the participant may be working, coding or moving -- so a break is not evidence
   about the cap.  The two eyes-closed/eyes-open rest blocks per session *are*
   controlled, and they are reported separately as timed anchors: four matched
   eyes-closed blocks spread across the hour are the cleanest drift probe here,
   because they hold posture and task constant while time varies.

**Time and session are confounded and the script never pretends otherwise.**  The
late chunks are all oddball and the early chunks are all Go/NoGo, so a rise could
be the cap drying or it could be the task.  Every metric is therefore reported as
(a) a slope *within* ``gng_flex``, (b) a slope *within* ``oddball_flex``, and (c)
the step between the two session medians.  Only the within-session slopes are
clean evidence about time; the step carries the task change with it.  A cap that
is drying out has to show a positive within-session slope, in both sessions.

Outputs go to ``data/device_quality/``: ``flex_endurance_chunks.csv`` (one row per
channel per chunk of task time), ``flex_endurance_bands.csv`` (band power per
channel per chunk), ``flex_endurance_trends.csv`` (the slopes and their
confidence intervals), ``flex_endurance_rest.csv`` (the rest-block anchors),
``flex_endurance_bounds.csv`` (the same-condition rate of change between the
first and last rest block, which is the only rate here worth extrapolating),
``flex_endurance.json`` and ``flex_endurance.png``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr, theilslopes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import emotiv as em  # noqa: E402
import device_quality as dq  # noqa: E402
import gng_erp  # noqa: E402
import oddball_erp  # noqa: E402

OUT = Path("data/device_quality")

# The same cap, the same wetting, in the order recorded.  ``anchor`` is the
# session whose first sample is t = 0 on the shared timeline.
SESSIONS = [
    {"key": "gng_flex", "xdf": "data/GnG/gng_flex.xdf", "csv": "data/GnG/gng_flex.csv",
     "task": "gng", "anchor": True},
    {"key": "oddball_flex", "xdf": "data/Oddball/oddball_flex.xdf", "csv": None,
     "task": "oddball", "anchor": False},
]

# The nine sites a midline ERP analysis actually reads, and the four inner
# neighbours that a centro-parietal ROI averages in.  Everything else on the cap
# -- the prefrontal, temporal, inferior and occipital rim -- is excluded here on
# purpose: see the module docstring.
CORE = ("F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4")
INNER = CORE + ("FC1", "FC2", "CP1", "CP2")
ROWS = {"F": ("F3", "Fz", "F4"), "C": ("C3", "Cz", "C4"), "P": ("P3", "Pz", "P4")}

# A chunk is a fixed amount of *accepted task time*, not a fixed amount of wall
# clock, so every chunk is measured to the same precision and a chunk that
# happens to straddle a break is not a thinner estimate than its neighbours.  Its
# position on the timeline is the median time of the windows in it.
CHUNK_TASK_S = 60.0
MIN_CHUNK_WINDOWS = 60  # half a chunk; a shorter tail is dropped rather than reported

# Frequency bands.  The top band stops at 30 Hz because the Flex ERP passband
# does (``FLEX_PROCESSING.erp_band``); the separate 20-45 Hz noise figure below is
# measured on unfiltered data, where it is still there to measure.
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
BAND_TOTAL = (1.0, 30.0)
PSD_NPERSEG = 512  # 4 s at 128 Hz -> 0.25 Hz bins
MIN_SEGMENT_S = 4.0  # a contiguous clean run shorter than one window is unusable

N_BOOT = 2000


# ---------------------------------------------------------------------------
# Loading


def session_events(spec: dict, run: em.Run):
    if spec["task"] == "gng":
        events, _ = gng_erp.task_events(run, spec["csv"])
        return events, gng_erp.task_mask(run, events)
    events, _ = oddball_erp.task_events(run)
    return events, oddball_erp.task_mask(run, events)


def stimulus_mask(run: em.Run, events: pd.DataFrame, task: np.ndarray) -> np.ndarray:
    """Task time with the between-block breaks cut out.

    ``task_mask`` spans the first to the last stimulus, which includes the rest
    breaks in the middle.  Both tasks run at SOAs of a few seconds, so any gap
    over 20 s between consecutive stimuli is a break; those seconds are removed
    here because they are uncontrolled and must not enter a stability trend.
    """

    stim = np.sort(events["time"].to_numpy())
    mask = task.copy()
    for i in np.flatnonzero(np.diff(stim) > 20.0):
        a, b = np.searchsorted(run.t, [stim[i], stim[i + 1]])
        mask[a:b] = False
    return mask


def rail_mask(run: em.Run) -> np.ndarray | None:
    """Per-sample, per-channel mask of seven-bit delta rails (channels x samples).

    ``rail_fraction`` collapses this to one number per session; the whole point
    here is to watch it move, so the mask is kept.  Index 0 is padded False
    because a delta needs two samples.
    """

    if run.device != "flex":
        return None
    deltas, real = em.flex_deltas(run)
    whole = np.round(deltas).astype(int)
    limit = em.devices.FLEX_DELTA_LIMIT
    railed = ((whole <= -run.delta_zero) | (whole >= 2 * limit + 1 - run.delta_zero)) & real[:, None]
    return np.vstack([np.zeros((1, railed.shape[1]), bool), railed]).T


def prepare(spec: dict) -> dict:
    """Load one session and everything the chunk loop reads out of it."""

    run = em.load(spec["xdf"], key=spec["key"])
    events, task = session_events(spec, run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = em.clean(run, task_mask=task)

    # Two views of the data, and the difference matters.  ``clean`` is the
    # 0.1-30 Hz ICA-cleaned signal an analysis actually sees, and is what the
    # contact and band-power numbers are measured on.  ``broad`` is unfiltered,
    # so it still contains the slow wander and the 30-45 Hz noise that the ERP
    # filter removes -- which is exactly where a drying electrode shows itself.
    x_clean, labels = dq.median_reference(cleaned)
    broad_raw = em.make_raw(run)
    x_broad = broad_raw.get_data() * 1e6
    # Each channel is levelled on its own median BEFORE the across-channel median
    # reference, and it is not optional.  The Flex decodes as a pure accumulator
    # with no DC restore, so every channel sits at a large, near-constant offset
    # of its own; on unfiltered data the rank order across channels is therefore
    # set by those offsets and never changes.  Measured on gng_flex.xdf: F3 and
    # F4 were ranks 16 and 17 of 32 in 99.7% and 99.9% of samples, so the median
    # was always (F3 + F4) / 2 and subtracting it left F3 = -F4 (r = -0.96) --
    # two electrodes reduced to one difference signal with, by construction,
    # identical noise in every band.  Levelling first makes the ranks follow the
    # signal (F3/F4 correlate +0.97 afterwards, as two frontal electrodes
    # should).  ``device_quality.py`` is not affected: it references the 0.1-30 Hz
    # filtered data, where the offsets are already gone.
    x_recorded = x_broad - np.median(x_broad, axis=1, keepdims=True)
    x_broad = x_recorded - np.median(x_recorded, axis=0, keepdims=True)

    return {
        "spec": spec, "run": run, "cleaned": cleaned, "events": events,
        "labels": labels,
        "x_clean": x_clean, "x_broad": x_broad, "x_recorded": x_recorded,
        "stim": stimulus_mask(run, events, task),
        "good": em.good_mask(cleaned.raw),
        "rails": rail_mask(run),
        "t0_min": None,  # filled in once the anchor is known
    }


# ---------------------------------------------------------------------------
# Per-chunk measurement


def chunk_edges(t: np.ndarray, starts: np.ndarray, keep: np.ndarray) -> list[tuple[float, float, int]]:
    """Group accepted analysis windows into chunks of ``CHUNK_TASK_S`` of task time."""

    per_chunk = int(round(CHUNK_TASK_S / em.preprocess.STEP_S))
    out = []
    for i in range(0, len(keep), per_chunk):
        group = keep[i:i + per_chunk]
        if len(group) < MIN_CHUNK_WINDOWS:
            continue
        times = t[starts[group]]
        out.append((float(times[0]), float(times[-1] + em.preprocess.WIN_S), len(group)))
    return out


def band_powers(x: np.ndarray, mask: np.ndarray, sfreq: float = em.FS) -> dict | None:
    """Welch band powers over the contiguous clean runs inside ``mask``.

    Averaged over segments weighted by length, so a chunk broken into several
    clean runs is not dominated by its shortest one.  Returns None when the
    chunk has under one PSD window of usable data.
    """

    n_min = int(round(MIN_SEGMENT_S * sfreq))
    starts, stops = em.contiguous(mask)
    segs = [(a, b) for a, b in zip(starts, stops) if b - a >= n_min]
    if not segs:
        return None
    psds, weights = [], []
    for a, b in segs:
        seg = x[:, a:b]
        nper = min(PSD_NPERSEG, seg.shape[1])
        freqs, psd = welch(seg, fs=sfreq, nperseg=nper, detrend="linear", axis=1)
        psds.append(psd)
        weights.append(b - a)
    # Interpolating to a common grid would smear the shortest segments, so a
    # short segment simply gets the resolution it can support and is resampled
    # onto the longest segment's grid.
    grid = max(psds, key=lambda p: p.shape[1])
    freqs = np.linspace(0, sfreq / 2, grid.shape[1])
    aligned = [p if p.shape[1] == grid.shape[1] else
               np.vstack([np.interp(freqs, np.linspace(0, sfreq / 2, p.shape[1]), row) for row in p])
               for p in psds]
    psd = np.average(aligned, axis=0, weights=weights)

    def integrate(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return np.trapezoid(psd[:, sel], freqs[sel], axis=1)

    total = integrate(*BAND_TOTAL)
    out = {"total_uv2": total, "seconds": sum(weights) / sfreq}
    for name, (lo, hi) in BANDS.items():
        p = integrate(lo, hi)
        out[f"{name}_uv2"] = p
        out[f"{name}_rel"] = p / total
    return out


def band_variability(x: np.ndarray, mask: np.ndarray, sfreq: float = em.FS) -> dict | None:
    """Spread of band power across 4-s windows inside a chunk, as a robust CV.

    "Is it getting more variable" is a separate question from "is it getting
    noisier": a cap that is intermittently losing contact gets more variable long
    before its median moves.  (p75 - p25) / median is used rather than sd/mean so
    that one bad window cannot create the effect it is supposed to detect.
    """

    n = int(round(MIN_SEGMENT_S * sfreq))
    starts, stops = em.contiguous(mask)
    windows = [(a + i, a + i + n) for a, b in zip(starts, stops)
               for i in range(0, b - a - n + 1, n)]
    if len(windows) < 4:
        return None
    rows = {name: [] for name in BANDS}
    for a, b in windows:
        freqs, psd = welch(x[:, a:b], fs=sfreq, nperseg=n, detrend="linear", axis=1)
        for name, (lo, hi) in BANDS.items():
            sel = (freqs >= lo) & (freqs < hi)
            rows[name].append(np.trapezoid(psd[:, sel], freqs[sel], axis=1))
    out = {"n_windows": len(windows)}
    for name, vals in rows.items():
        arr = np.asarray(vals)  # windows x channels
        q25, med, q75 = np.percentile(arr, [25, 50, 75], axis=0)
        out[f"{name}_cv"] = (q75 - q25) / med
    return out


def wander(x: np.ndarray, mask: np.ndarray, t: np.ndarray, sfreq: float = em.FS) -> dict:
    """Slow baseline movement, from one-second means of the usable samples.

    Deliberately not a filter: a filter rings across the DC steps a lost Flex
    packet leaves behind, and those steps are one of the things being counted.
    ``wander_uv`` is the p10-p90 spread of the per-second level and ``drift`` its
    Theil-Sen slope, so a channel that walks steadily in one direction is told
    apart from one that wobbles.
    """

    n = int(round(sfreq))
    edges = np.arange(0, x.shape[1] - n + 1, n)
    levels, times = [], []
    for a in edges:
        sel = mask[a:a + n]
        if sel.mean() < 0.8:
            continue
        levels.append(x[:, a:a + n][:, sel].mean(axis=1))
        times.append(t[a] / 60.0)
    if len(levels) < 5:
        return {}
    lv = np.asarray(levels)  # seconds x channels
    tm = np.asarray(times)
    q10, q90 = np.percentile(lv, [10, 90], axis=0)
    slope = np.array([theilslopes(lv[:, j], tm)[0] for j in range(lv.shape[1])])
    # A lost Flex packet leaves a permanent DC step of ~17 uV, so one dropout
    # inflates the p10-p90 excursion of a whole chunk and would be read as
    # electrode drift.  The median second-to-second movement is the same quantity
    # per unit time and a single step cannot move a median, so the two together
    # separate "this electrode is walking" from "the radio dropped a block".
    rate = np.median(np.abs(np.diff(lv, axis=0)), axis=0)
    return {"wander_uv": q90 - q10, "wander_rate_uv_per_s": rate,
            "drift_uv_per_min": slope, "n_seconds": len(levels)}


def chunk_table(prep: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per channel per chunk: contact, noise, rails, dropouts, drift; and bands."""

    run, cleaned = prep["run"], prep["cleaned"]
    labels, x_clean, x_broad = prep["labels"], prep["x_clean"], prep["x_broad"]
    stim, good = prep["stim"], prep["good"]
    t0_min = prep["t0_min"]
    sfreq = float(run.sfreq_hz)

    p2p, starts = em.sliding_p2p(x_clean, sfreq=sfreq)
    end = np.minimum(starts + int(round(em.preprocess.WIN_S * sfreq)) - 1, len(stim) - 1)
    keep = np.flatnonzero(stim[starts] & stim[end])

    hf = em.preprocess.sosfiltfilt(
        em.preprocess.butter(4, [20, 45], btype="band", fs=sfreq, output="sos"), x_broad, axis=1)
    # The EEG band beside the noise band.  A contact failure raises both; muscle
    # and a rising sensor floor raise mostly the high one, so the ratio of the
    # two says which kind of change a session is showing.
    lf = em.preprocess.sosfiltfilt(
        em.preprocess.butter(4, [1, 20], btype="band", fs=sfreq, output="sos"), x_broad, axis=1)
    pops = cleaned.pops["times_s"]
    rails = prep["rails"]
    filled = run.filled
    fill_starts, fill_stops = em.contiguous(filled)

    rows, band_rows = [], []
    for k, (a, b, n_win) in enumerate(chunk_edges(run.t, starts, keep), start=1):
        i, j = np.searchsorted(run.t, [a, b])
        in_chunk = np.zeros(len(run.t), bool)
        in_chunk[i:j] = True
        usable = in_chunk & stim & good
        if usable.sum() < MIN_SEGMENT_S * sfreq:
            continue
        sel = keep[(run.t[starts[keep]] >= a) & (run.t[starts[keep]] < b)]
        minutes = usable.sum() / sfreq / 60.0
        elapsed = t0_min + (a + b) / 2 / 60.0

        bands = band_powers(x_clean, usable, sfreq)
        var = band_variability(x_clean, usable, sfreq)
        wan = wander(x_broad, usable, run.t, sfreq)

        gaps = [(s, e) for s, e in zip(fill_starts, fill_stops) if i <= s < j]
        for c, ch in enumerate(labels):
            row = {
                "session": run.key, "chunk": k, "elapsed_min": elapsed,
                "t_start_s": a, "t_end_s": b, "task_minutes": minutes,
                "channel": ch, "row": next((r for r, chans in ROWS.items() if ch in chans), ""),
                "in_core": ch in CORE, "in_inner": ch in INNER,
                "marked_bad": ch in cleaned.bads,
                # Contact, measured exactly as device_quality measures it.
                "p2p_median_uv": float(np.median(p2p[c, sel])),
                "p2p_p99_over_median": float(np.percentile(p2p[c, sel], 99)
                                             / np.median(p2p[c, sel])),
                "pct_windows_over_common": float(100 * (p2p[c, sel] > dq.COMMON_P2P_UV).mean()),
                # Noise floor, unfiltered so the ERP passband does not hide it.
                "hf_20_45_uv_rms": float(np.sqrt(np.mean(hf[c, usable] ** 2))),
                "rms_1_20_uv": float(np.sqrt(np.mean(lf[c, usable] ** 2))),
                "hf_over_lf": float(np.sqrt(np.mean(hf[c, usable] ** 2))
                                    / np.sqrt(np.mean(lf[c, usable] ** 2))),
                "pops_per_min": float(sum(a <= p < b for p in pops[ch]) / minutes),
                "n_windows": int(len(sel)),
            }
            if rails is not None:
                row["rail_pct"] = float(100 * rails[c, i:j][stim[i:j]].mean())
            if wan:
                row["wander_uv"] = float(wan["wander_uv"][c])
                row["wander_rate_uv_per_s"] = float(wan["wander_rate_uv_per_s"][c])
                row["drift_uv_per_min"] = float(wan["drift_uv_per_min"][c])
            # Dropouts are a radio property, not an electrode property, so they
            # are the same for every channel in the chunk; carried on each row so
            # the table can be filtered to the core set without losing them.
            row["gaps_per_min"] = len(gaps) / minutes
            row["lost_samples"] = int(sum(e - s for s, e in gaps))
            row["has_gap"] = bool(gaps)
            rows.append(row)

            if bands is not None:
                brow = {"session": run.key, "chunk": k, "elapsed_min": elapsed,
                        "channel": ch, "in_core": ch in CORE,
                        "psd_seconds": bands["seconds"],
                        "total_uv2": float(bands["total_uv2"][c])}
                for name in BANDS:
                    brow[f"{name}_uv2"] = float(bands[f"{name}_uv2"][c])
                    brow[f"{name}_db"] = float(10 * np.log10(bands[f"{name}_uv2"][c]))
                    brow[f"{name}_rel_pct"] = float(100 * bands[f"{name}_rel"][c])
                    brow[f"{name}_cv"] = float(var[f"{name}_cv"][c]) if var else float("nan")
                band_rows.append(brow)

    return pd.DataFrame(rows), pd.DataFrame(band_rows)


# ---------------------------------------------------------------------------
# The rest-block anchors


def rest_table(prep: dict) -> pd.DataFrame:
    """The eyes-closed/eyes-open blocks, per channel, placed on the timeline.

    These are the controlled probes: same instruction, same posture, four of each
    spread over the hour.  If the cap is drying, an eyes-closed block at minute 60
    must be worse than the one at minute 3, and unlike task time there is no task
    difference to explain it away.
    """

    run = prep["run"]
    labels, x_clean, x_broad = prep["labels"], prep["x_clean"], prep["x_broad"]
    good, t0_min = prep["good"], prep["t0_min"]
    sfreq = float(run.sfreq_hz)
    intervals = em.find_eye_intervals(run)
    if not intervals:
        return pd.DataFrame()

    p2p, starts = em.sliding_p2p(x_clean, sfreq=sfreq)
    t_win = run.t[np.clip(starts, 0, len(run.t) - 1)]
    hf = em.preprocess.sosfiltfilt(
        em.preprocess.butter(4, [20, 45], btype="band", fs=sfreq, output="sos"), x_broad, axis=1)
    rails = prep["rails"]

    rows = []
    for iv in intervals:
        a, b = iv["start"] + em.qc.EYE_TRIM[0], iv["end"] - em.qc.EYE_TRIM[1]
        sel = np.flatnonzero((t_win >= a) & (t_win < b))
        i, j = np.searchsorted(run.t, [a, b])
        usable = np.zeros(len(run.t), bool)
        usable[i:j] = True
        usable &= good
        if len(sel) < 20 or usable.sum() < MIN_SEGMENT_S * sfreq:
            continue
        bands = band_powers(x_clean, usable, sfreq)
        wan = wander(x_broad, usable, run.t, sfreq)
        for c, ch in enumerate(labels):
            row = {"session": run.key, "state": iv["state"], "pair": iv["pair"],
                   "elapsed_min": t0_min + (a + b) / 2 / 60.0, "duration_s": b - a,
                   "channel": ch, "in_core": ch in CORE,
                   "p2p_median_uv": float(np.median(p2p[c, sel])),
                   "p2p_p99_over_median": float(np.percentile(p2p[c, sel], 99)
                                                / np.median(p2p[c, sel])),
                   "pct_windows_over_common": float(100 * (p2p[c, sel] > dq.COMMON_P2P_UV).mean()),
                   "hf_20_45_uv_rms": float(np.sqrt(np.mean(hf[c, usable] ** 2)))}
            if rails is not None:
                row["rail_pct"] = float(100 * rails[c, i:j].mean())
            if wan:
                row["wander_uv"] = float(wan["wander_uv"][c])
                row["wander_rate_uv_per_s"] = float(wan["wander_rate_uv_per_s"][c])
            if bands is not None:
                for name in BANDS:
                    row[f"{name}_db"] = float(10 * np.log10(bands[f"{name}_uv2"][c]))
                    row[f"{name}_rel_pct"] = float(100 * bands[f"{name}_rel"][c])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trends


def trend(x: np.ndarray, y: np.ndarray, per: float = 60.0) -> dict:
    """Theil-Sen slope per ``per`` minutes, its CI, and Spearman rho.

    Theil-Sen rather than least squares because a single chunk containing one
    movement is exactly the failure mode here and would drag a least-squares
    line; Spearman beside it because the question is monotone degradation, not a
    particular functional form.
    """

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    blank = {"n": int(len(x)), "slope": float("nan"), "lo": float("nan"), "hi": float("nan"),
             "rho": float("nan"), "p": float("nan"), "constant": False,
             "median": float(np.median(y)) if len(y) else float("nan")}
    if len(x) < 5 or np.ptp(x) == 0:
        return blank
    if np.ptp(y) == 0:
        # A metric that never moves is an answer, not a missing value: on the core
        # nine both the over-threshold share and the pop rate are zero in every
        # chunk of both sessions.  Reported as constant rather than as a slope,
        # because a correlation with a constant is undefined.
        return {**blank, "slope": 0.0, "lo": 0.0, "hi": 0.0, "constant": True}
    slope, _, lo, hi = theilslopes(y, x, alpha=0.95)
    rho, p = spearmanr(x, y)
    return {"n": int(len(x)), "slope": float(slope * per), "lo": float(lo * per),
            "hi": float(hi * per), "rho": float(rho), "p": float(p), "constant": False,
            "median": float(np.median(y))}


# metric, label, which table it lives in, and whether it is on a ratio scale.
# The last flag exists because a "percent change" in a quantity already expressed
# in decibels is meaningless: those rows report the session step in dB instead.
METRICS = [
    ("hf_20_45_uv_rms", "20-45 Hz noise (uV rms)", "chunks", True),
    ("rms_1_20_uv", "1-20 Hz rms (uV)", "chunks", True),
    ("hf_over_lf", "20-45 Hz / 1-20 Hz", "chunks", True),
    ("p2p_median_uv", "1-s p2p median (uV)", "chunks", True),
    ("p2p_p99_over_median", "p2p p99 / own median", "chunks", True),
    ("pct_windows_over_common", "% of seconds over 120 uV", "chunks", False),
    ("rail_pct", "% of samples railing the delta encoder", "chunks", False),
    ("pops_per_min", "electrode pops / min", "chunks", False),
    ("wander_uv", "baseline excursion, p10-p90 of 1-s level (uV)", "chunks", True),
    ("wander_rate_uv_per_s", "baseline movement (uV/s, median)", "chunks", True),
    ("gaps_per_min", "radio dropouts / min", "chunks", False),
    ("delta_db", "1-4 Hz power (dB)", "bands", False),
    ("theta_db", "4-8 Hz power (dB)", "bands", False),
    ("alpha_db", "8-13 Hz power (dB)", "bands", False),
    ("beta_db", "13-30 Hz power (dB)", "bands", False),
    ("theta_rel_pct", "theta, % of 1-30 Hz", "bands", False),
    ("alpha_rel_pct", "alpha, % of 1-30 Hz", "bands", False),
    ("theta_cv", "theta variability (IQR/median over 4-s windows)", "bands", True),
    ("alpha_cv", "alpha variability (IQR/median over 4-s windows)", "bands", True),
]

# The scopes: the sites a midline analysis reads, those plus their FC/CP
# neighbours, and -- as the control -- the rim that this script otherwise
# excludes.  The rim is here for one reason: a change that appears on the core
# and on the rim at the same moment is the participant, because those electrodes
# are on the other side of the head and share nothing but the subject and the
# amplifier.
SCOPES = {"core": CORE, "inner": INNER, "rim": None}


def scope_channels(frame: pd.DataFrame, scope: str) -> list[str]:
    if SCOPES[scope] is not None:
        return list(SCOPES[scope])
    bad = set(frame.loc[frame.get("marked_bad", False) == True, "channel"]) if "marked_bad" in frame else set()
    return [c for c in frame["channel"].unique() if c not in INNER and c not in bad]


def session_trend(per_chunk: pd.DataFrame, metric: str, key: str) -> dict:
    """Within-session slope, plus the two diagnostics that stop a step being read as drift.

    ``settled`` is the same slope with the session's first chunk dropped, because
    the first minute after a rest block is the participant settling into the task
    and not the cap doing anything.  ``largest_step_share`` is the biggest
    chunk-to-chunk jump as a fraction of the session's whole range: near 1 means
    the "trend" is one step and a flat line either side of it, which is what a
    posture change looks like and is not what a drying electrode looks like.
    """

    s = per_chunk[per_chunk["session"] == key].sort_values("elapsed_min")
    x, y = s["elapsed_min"].to_numpy(), s[metric].to_numpy()
    t = trend(x, y)
    out = {f"{key}_n": t["n"], f"{key}_median": t["median"],
           f"{key}_first": float(y[0]) if len(y) else float("nan"),
           f"{key}_last": float(y[-1]) if len(y) else float("nan"),
           f"{key}_slope_per_h": t["slope"], f"{key}_lo": t["lo"], f"{key}_hi": t["hi"],
           f"{key}_rho": t["rho"], f"{key}_p": t["p"], f"{key}_constant": t["constant"]}
    settled = trend(x[1:], y[1:])
    out[f"{key}_settled_slope_per_h"] = settled["slope"]
    out[f"{key}_settled_lo"], out[f"{key}_settled_hi"] = settled["lo"], settled["hi"]
    rng = float(np.ptp(y)) if len(y) else 0.0
    out[f"{key}_largest_step_share"] = (float(np.abs(np.diff(y)).max() / rng)
                                        if len(y) > 1 and rng > 0 else float("nan"))
    return out


def trend_table(chunks: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Within-session slopes, the between-session step, and the pooled slope.

    The pooled slope is reported last and with a warning attached, because the
    30-minute recording gap means it is fitted across a session change: it
    answers "did the numbers get worse over the hour", not "did the cap get
    worse over the hour", and only the within-session columns answer the second.
    """

    rows = []
    for metric, label, source, ratio in METRICS:
        frame = chunks if source == "chunks" else bands
        if metric not in frame:
            continue
        for scope in SCOPES:
            chans = scope_channels(chunks, scope)
            sub = frame[frame["channel"].isin(chans)]
            if sub.empty:
                continue
            per_chunk = (sub.groupby(["session", "chunk", "elapsed_min"])[metric]
                         .median().reset_index())
            row = {"metric": metric, "label": label, "scope": scope, "ratio_scale": ratio}
            for key in ("gng_flex", "oddball_flex"):
                row.update(session_trend(per_chunk, metric, key))
            row["step"] = row["oddball_flex_median"] - row["gng_flex_median"]
            row["step_pct"] = (100 * row["step"] / row["gng_flex_median"]
                               if ratio and row["gng_flex_median"] else float("nan"))
            pooled = trend(per_chunk["elapsed_min"].to_numpy(), per_chunk[metric].to_numpy())
            row["pooled_slope_per_h"] = pooled["slope"]
            row["pooled_lo"], row["pooled_hi"] = pooled["lo"], pooled["hi"]
            row["pooled_rho"], row["pooled_p"] = pooled["rho"], pooled["p"]
            # Both within-session slopes pointing the same way, with neither CI
            # touching zero, is the only pattern here that the session change
            # cannot produce on its own.  Everything else is a candidate, not a
            # finding.
            a, b = row["gng_flex_slope_per_h"], row["oddball_flex_slope_per_h"]
            row["both_ci_exclude_zero"] = bool(
                np.isfinite(a) and np.isfinite(b)
                and (row["gng_flex_lo"] > 0 or row["gng_flex_hi"] < 0)
                and (row["oddball_flex_lo"] > 0 or row["oddball_flex_hi"] < 0)
                and np.sign(a) == np.sign(b))
            rows.append(row)
    return pd.DataFrame(rows)


def channel_trends(chunks: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The same within-session slopes, per electrode, for the core nine."""

    rows = []
    for ch in CORE:
        sub = chunks[chunks["channel"] == ch]
        row = {"channel": ch}
        for key in ("gng_flex", "oddball_flex"):
            s = sub[sub["session"] == key].sort_values("elapsed_min")
            t = trend(s["elapsed_min"].to_numpy(), s[metric].to_numpy())
            settled = trend(s["elapsed_min"].to_numpy()[1:], s[metric].to_numpy()[1:])
            row[f"{key}_median"] = t["median"]
            row[f"{key}_slope_per_h"] = t["slope"]
            row[f"{key}_lo"], row[f"{key}_hi"] = t["lo"], t["hi"]
            row[f"{key}_settled_slope_per_h"] = settled["slope"]
        rows.append(row)
    return pd.DataFrame(rows)


# Metrics for which the rest-block anchors give a same-condition rate of change.
BOUND_METRICS = ("hf_20_45_uv_rms", "p2p_median_uv", "pct_windows_over_common",
                 "rail_pct", "wander_rate_uv_per_s")


def rest_bound(rest: pd.DataFrame, target_min: float = 150.0,
               n_boot: int = N_BOOT, seed: int = em.SEED) -> pd.DataFrame:
    """Rate of change between the two sessions' rest blocks, matched pair for pair.

    **This is the only clean rate of change in the data, and it is the one to
    quote.**  The within-task slopes cannot be extrapolated: each is fitted over
    ten to twelve minutes, the two sessions are two different tasks, and -- as the
    trend table shows -- they disagree in sign, so projecting either one measures
    the task, not the cap.  Two eyes-closed blocks 46 minutes apart hold posture,
    instruction and eye state constant while the only thing that differs is how
    long the saline has been on the head.

    The blocks are matched by their **position in the session** as well as by
    state: the first eyes-closed block of ``gng_flex`` against the first of
    ``oddball_flex``, the second against the second.  Without that, a comparison
    of the first block of one session with the last of the other would carry
    whatever settling happens inside a rest sequence.  The confidence interval is
    bootstrapped over the nine core electrodes, so it answers "would a different
    electrode have told a different story", which is the relevant uncertainty for
    a cap rather than for a recording.
    """

    sub = rest[rest["in_core"]]
    if sub.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for (state, pair), block in sub.groupby(["state", "pair"]):
        early = block[block["session"] == "gng_flex"]
        late = block[block["session"] == "oddball_flex"]
        if early.empty or late.empty:
            continue
        hours = (late["elapsed_min"].iloc[0] - early["elapsed_min"].iloc[0]) / 60.0
        for metric in BOUND_METRICS:
            if metric not in block:
                continue
            u = early.set_index("channel")[metric]
            v = late.set_index("channel")[metric]
            chans = [c for c in CORE if c in u.index and c in v.index]
            d = (v[chans] - u[chans]).to_numpy()
            boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            base = float(np.median(u[chans]))
            rows.append({
                "metric": metric, "state": state, "pair": int(pair),
                "n_channels": len(chans), "hours_apart": hours,
                "early_min": float(early["elapsed_min"].iloc[0]),
                "late_min": float(late["elapsed_min"].iloc[0]),
                "early_value": base, "late_value": float(np.median(v[chans])),
                "change": float(np.mean(d)), "change_lo": float(lo), "change_hi": float(hi),
                "per_hour": float(np.mean(d) / hours),
                "per_hour_lo": float(lo / hours), "per_hour_hi": float(hi / hours),
                "worse_channels": int((d > 0).sum()),
                # The bound: the worst end of the interval, carried to the target.
                "value_at_target_worst_case": base + float(hi / hours) * (target_min / 60.0),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["metric", "state", "pair"])


def poisson_upper(k: int, exposure: float, conf: float = 0.95) -> float:
    """Upper confidence bound on a rate from ``k`` events in ``exposure`` units.

    Exact (chi-squared) rather than the rule of three, because these counts are
    small but not zero: the core nine do lose the occasional second, and rounding
    that to "never" would overstate the case as badly as ignoring how rare it is.
    """

    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (k + 1)) / 2 / exposure) if exposure else float("nan")


def reference_check(preps: list[dict], chunks: pd.DataFrame) -> pd.DataFrame:
    """Repeat the core-nine noise trend in the recorded reference, with no re-referencing.

    Every number above is measured in the median-across-channels reference, which
    is right for comparing electrodes but is computed from the data it is
    measuring: a change shared by the whole cap partly cancels, so "the core rose
    and the rim did not" could in principle be an artifact of the reference rather
    than a fact about the electrodes.  The recorded reference (CMS = TP9) cannot
    cancel anything, so if the two references agree about the direction of a
    within-session trend, the trend is in the data.

    The absolute levels are not comparable between the two -- a single-ended
    reference puts the CMS electrode's own noise into every channel, which is why
    ``devices.py`` measures noise in the median reference in the first place.
    Only the signs and the slopes are being compared here.
    """

    rows = []
    for prep in preps:
        run, key = prep["run"], prep["run"].key
        hf = em.preprocess.sosfiltfilt(
            em.preprocess.butter(4, [20, 45], btype="band", fs=run.sfreq_hz, output="sos"),
            prep["x_recorded"], axis=1)
        idx = [prep["labels"].index(c) for c in CORE if c in prep["labels"]]
        usable_base = prep["stim"] & prep["good"]
        sub = chunks[(chunks["session"] == key) & (chunks["channel"] == CORE[0])]
        for r in sub.itertuples():
            i, j = np.searchsorted(run.t, [r.t_start_s, r.t_end_s])
            u = np.zeros(len(run.t), bool)
            u[i:j] = True
            u &= usable_base
            rows.append({"session": key, "chunk": r.chunk, "elapsed_min": r.elapsed_min,
                         "hf_recorded_ref": float(np.median(
                             [np.sqrt(np.mean(hf[m, u] ** 2)) for m in idx]))})
    return pd.DataFrame(rows)


def failure_rates(chunks: pd.DataFrame) -> pd.DataFrame:
    """How often the core nine actually failed, early half against late half.

    The two halves are split on the timeline, not within a session, so the early
    half is Go/NoGo and the late half is the oddball.  A cap that is drying has to
    put more failures in the late half; the exposure is large enough that a real
    doubling would be visible, and the Poisson intervals say how much smaller a
    change could still be hiding.
    """

    core = chunks[chunks["in_core"]].copy()
    core["over_n"] = core["pct_windows_over_common"] / 100 * core["n_windows"]
    core["pops_n"] = core["pops_per_min"] * core["task_minutes"]
    mid = core["elapsed_min"].median()
    rows = []
    for name, sub in (("early (Go/NoGo)", core[core["elapsed_min"] <= mid]),
                      ("late (oddball)", core[core["elapsed_min"] > mid]),
                      ("all", core)):
        windows, minutes = float(sub["n_windows"].sum()), float(sub["task_minutes"].sum())
        over, pops = int(round(sub["over_n"].sum())), int(round(sub["pops_n"].sum()))
        rows.append({
            "half": name,
            "first_min": float(sub["elapsed_min"].min()), "last_min": float(sub["elapsed_min"].max()),
            "channel_windows": int(windows), "channel_minutes": minutes,
            "over_threshold": over,
            "pct_windows_over_common": 100 * over / windows if windows else float("nan"),
            "pct_upper_95": 100 * poisson_upper(over, windows),
            "pops": pops, "pops_per_min": pops / minutes if minutes else float("nan"),
            "pops_upper_95": poisson_upper(pops, minutes),
            "channels_affected": int(sub.loc[(sub["over_n"] > 0) | (sub["pops_n"] > 0),
                                             "channel"].nunique()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure


def figure(chunks: pd.DataFrame, bands: pd.DataFrame, rest: pd.DataFrame, path: Path,
           gap: tuple[float, float] | None = None) -> None:
    """Task chunks as lines, rest blocks as markers, on one timeline.

    The rest markers are the point of the figure.  A within-task line that climbs
    and a rest marker at the end of the same climb that sits back at its starting
    value cannot both be the cap; putting them on the same axes is what makes that
    visible without any fitting.
    """

    core = chunks[chunks["channel"].isin(CORE)]
    core_b = bands[bands["channel"].isin(CORE)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    colour = {"gng_flex": "#2166ac", "oddball_flex": "#b2182b"}

    def panel(ax, frame, metric, label):
        for key, sub in frame.groupby("session"):
            per = sub.groupby(["chunk", "elapsed_min"])[metric].median().reset_index()
            ax.plot(per["elapsed_min"], per[metric], "o-", ms=4, lw=1.2,
                    color=colour.get(key, "#444"), label=key if metric == "hf_20_45_uv_rms" else None)
            if len(per) >= 5:
                sl, ic, _, _ = theilslopes(per[metric], per["elapsed_min"])
                xs = np.array([per["elapsed_min"].min(), per["elapsed_min"].max()])
                ax.plot(xs, ic + sl * xs, "--", color=colour.get(key, "#444"), lw=1.0, alpha=0.7)
        if not rest.empty and metric in rest:
            for state, marker in (("closed", "^"), ("open", "v")):
                sub = rest[(rest["state"] == state) & rest["in_core"]]
                if sub.empty:
                    continue
                per = sub.groupby("elapsed_min")[metric].median()
                ax.plot(per.index, per.values, marker, color="k", ms=8,
                        mfc="k" if state == "closed" else "none",
                        label=f"rest, eyes {state}" if metric == "hf_20_45_uv_rms" else None)
        if gap:
            ax.axvspan(*gap, color="0.85", zorder=0)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25)

    panel(axes[0, 0], core, "hf_20_45_uv_rms", "20–45 Hz noise (µV rms)")
    panel(axes[0, 1], core, "p2p_median_uv", "1-s p2p median (µV)")
    panel(axes[1, 0], core, "rail_pct", "% samples railing")
    panel(axes[1, 1], core, "wander_rate_uv_per_s", "baseline movement (µV/s)")
    panel(axes[2, 0], core_b, "theta_db", "4–8 Hz power (dB)")
    panel(axes[2, 1], core_b, "alpha_db", "8–13 Hz power (dB)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    for ax in axes[2]:
        ax.set_xlabel("minutes since the Flex cap started recording", fontsize=9)
    fig.suptitle("Flex 1.0, one wetting: the core nine midline sites over 65 minutes\n"
                 "occipital and rim electrodes excluded, task time only; "
                 "the shaded band is between recordings and was not measured", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    preps = []
    for spec in SESSIONS:
        print(f"loading {spec['key']} ...", flush=True)
        preps.append(prepare(spec))
    anchor = next(p for p in preps if p["spec"]["anchor"])["run"].extra["lsl_t0"]
    for p in preps:
        p["t0_min"] = (p["run"].extra["lsl_t0"] - anchor) / 60.0

    chunk_frames, band_frames, rest_frames = [], [], []
    for p in preps:
        print(f"measuring {p['run'].key} ...", flush=True)
        c, b = chunk_table(p)
        chunk_frames.append(c)
        band_frames.append(b)
        rest_frames.append(rest_table(p))
    chunks = pd.concat(chunk_frames, ignore_index=True)
    bands = pd.concat(band_frames, ignore_index=True)
    rest = pd.concat([r for r in rest_frames if not r.empty], ignore_index=True)

    trends = trend_table(chunks, bands)
    span = chunks["elapsed_min"].max() - chunks["elapsed_min"].min()

    print("\n=== timeline ===")
    for p in preps:
        r = p["run"]
        print(f"  {r.key:<13} starts {p['t0_min']:6.1f} min, "
              f"{r.t[-1] / 60:5.1f} min long, "
              f"{p['stim'].sum() / p['run'].sfreq_hz / 60:5.1f} min of stimulus time")
    print(f"  spanned {span:.1f} min of chunk centres; "
          f"{chunks.groupby(['session', 'chunk']).ngroups} chunks of {CHUNK_TASK_S:.0f} s task time")
    bad = sorted({c for p in preps for c in p["cleaned"].bads})
    print(f"  channels marked bad in either session: {bad}")
    print(f"  of the core nine: {[c for c in CORE if c in bad] or 'none'}")

    core = chunks[chunks["in_core"]]
    print("\n=== core nine, per chunk (medians over the nine channels) ===")
    show = (core.groupby(["session", "chunk", "elapsed_min"])
            [["hf_20_45_uv_rms", "rms_1_20_uv", "p2p_median_uv", "p2p_p99_over_median",
              "pct_windows_over_common", "rail_pct", "pops_per_min",
              "wander_rate_uv_per_s", "wander_uv", "gaps_per_min"]]
            .median().round(3).reset_index())
    print(show.to_string(index=False))

    print("\n=== trends: within-session slope per hour [95% CI], and the session step ===")
    cols = ["metric", "gng_flex_median", "gng_flex_slope_per_h", "gng_flex_lo",
            "gng_flex_hi", "oddball_flex_median", "oddball_flex_slope_per_h",
            "oddball_flex_lo", "oddball_flex_hi", "oddball_flex_settled_slope_per_h",
            "oddball_flex_largest_step_share", "step", "step_pct", "both_ci_exclude_zero"]
    print(trends[trends["scope"] == "core"][cols].round(3).to_string(index=False))

    print("\n=== the same, on the excluded rim: does a change reach the whole cap at once? ===")
    keep = ["hf_20_45_uv_rms", "rms_1_20_uv", "p2p_median_uv", "rail_pct", "theta_db", "alpha_db"]
    side = trends[trends["metric"].isin(keep) & trends["scope"].isin(["core", "rim"])]
    print(side[["metric", "scope", "gng_flex_median", "gng_flex_first", "gng_flex_last",
                "oddball_flex_first", "oddball_flex_median", "oddball_flex_last",
                "step"]].round(3).to_string(index=False))

    print("\n=== per electrode, 20-45 Hz noise, slope per hour ===")
    per_ch = channel_trends(chunks, "hf_20_45_uv_rms")
    print(per_ch.round(3).to_string(index=False))

    if not rest.empty:
        print("\n=== rest-block anchors, core nine (medians over channels) ===")
        r = (rest[rest["in_core"]]
             .groupby(["session", "state", "pair", "elapsed_min"])
             [["hf_20_45_uv_rms", "p2p_median_uv", "pct_windows_over_common",
               "rail_pct", "wander_rate_uv_per_s", "alpha_db", "theta_db"]]
             .median().round(3).reset_index().sort_values("elapsed_min"))
        print(r.to_string(index=False))

    bounds = rest_bound(rest) if not rest.empty else pd.DataFrame()
    if not bounds.empty:
        print("\n=== rest blocks 46 min apart, matched by state and by position in the session ===")
        print(bounds[["metric", "state", "pair", "early_min", "late_min", "early_value",
                      "late_value", "change", "change_lo", "change_hi", "worse_channels",
                      "value_at_target_worst_case"]].round(3).to_string(index=False))

    print("\n=== by electrode row, early half vs late half (medians over the row) ===")
    half = chunks[chunks["in_core"]].copy()
    mid = half["elapsed_min"].median()
    half["half"] = np.where(half["elapsed_min"] <= mid, "early", "late")
    rows = (half.groupby(["row", "half"])
            [["hf_20_45_uv_rms", "rms_1_20_uv", "p2p_median_uv", "p2p_p99_over_median",
              "rail_pct", "wander_rate_uv_per_s"]].median()
            .unstack("half").round(3))
    print(rows.to_string())

    ref = reference_check(preps, chunks)
    print("\n=== robustness: the same core-nine noise trend in the recorded (CMS) reference ===")
    for key, sub in ref.groupby("session"):
        sub = sub.sort_values("elapsed_min")
        t = trend(sub["elapsed_min"].to_numpy(), sub["hf_recorded_ref"].to_numpy())
        m = trends[(trends["metric"] == "hf_20_45_uv_rms") & (trends["scope"] == "core")].iloc[0]
        print(f"  {key:<13} recorded ref {sub['hf_recorded_ref'].iloc[0]:.2f} -> "
              f"{sub['hf_recorded_ref'].iloc[-1]:.2f} uV, slope {t['slope']:+.2f}/h "
              f"[{t['lo']:+.2f}, {t['hi']:+.2f}]   "
              f"median ref slope {m[f'{key}_slope_per_h']:+.2f}/h")

    fails = failure_rates(chunks)
    print("\n=== how often the core nine actually failed, early half vs late half ===")
    print(fails.round(4).to_string(index=False))
    replicated = trends[trends["both_ci_exclude_zero"] & (trends["scope"] == "core")]
    print(f"\n  metrics whose within-session slope is non-zero in BOTH sessions with the "
          f"same sign: {list(replicated['metric']) or 'NONE'}")

    chunks.to_csv(args.out / "flex_endurance_chunks.csv", index=False)
    bands.to_csv(args.out / "flex_endurance_bands.csv", index=False)
    trends.to_csv(args.out / "flex_endurance_trends.csv", index=False)
    rest.to_csv(args.out / "flex_endurance_rest.csv", index=False)
    if not bounds.empty:
        bounds.to_csv(args.out / "flex_endurance_bounds.csv", index=False)
    fails.to_csv(args.out / "flex_endurance_failures.csv", index=False)
    gap = (float(preps[0]["run"].t[-1] / 60), float(preps[1]["t0_min"]))
    figure(chunks, bands, rest, args.out / "flex_endurance.png", gap=gap)
    summary = {
        "question": "does the Flex cap's usable core degrade within one wetting",
        "core_channels": list(CORE), "inner_channels": list(INNER),
        "excluded": "prefrontal, temporal, inferior and occipital rim",
        "chunk_task_seconds": CHUNK_TASK_S,
        "timeline": {p["run"].key: {"start_min": p["t0_min"],
                                    "minutes": float(p["run"].t[-1] / 60),
                                    "stimulus_minutes": float(p["stim"].sum() / p["run"].sfreq_hz / 60)}
                     for p in preps},
        "unobserved_gap_min": float(preps[1]["t0_min"] - preps[0]["t0_min"]
                                    - preps[0]["run"].t[-1] / 60),
        "bad_channels": bad,
        "trends": em.to_jsonable(trends.to_dict("records")),
        "channel_noise_trends": em.to_jsonable(per_ch.to_dict("records")),
        "rest_anchor_bounds_150_min": em.to_jsonable(bounds.to_dict("records")
                                                     if not bounds.empty else []),
        "failure_rates": em.to_jsonable(fails.to_dict("records")),
        "recorded_reference_check": em.to_jsonable(ref.to_dict("records")),
        "replicated_trends": list(replicated["metric"]),
    }
    (args.out / "flex_endurance.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/flex_endurance_{{chunks,bands,trends,rest}}.csv, "
          f"flex_endurance.json, flex_endurance.png")


if __name__ == "__main__":
    main()
