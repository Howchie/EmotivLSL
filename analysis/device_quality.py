#!/usr/bin/env python3
"""Head-to-head signal quality: EPOC X 14-sensor ring vs Flex 1.0 32-electrode cap.

The question this answers is not "which headset has a better ERP" -- the task
scripts already report that -- but "what does each headset's *data* cost you",
so that the Flex's extra 18 electrodes can be weighed against its older radio
and its saline cap.  Everything here comes out of the ``emotiv`` package, so a
number means the same thing it means in ``gng_erp.py`` and ``oddball_erp.py``::

    python analysis/device_quality.py

Three things had to be made device-neutral before the two caps could be put in
the same table, and each of them is a deliberate departure from the per-device
processing profile:

1. **The reference.**  Each headset is analysed in its own recorded reference
   (``devices.py`` explains why), but a recorded reference puts the CMS
   electrode's own noise into every channel, and the two headsets have different
   CMS sites.  Every noise number below is therefore measured in the **median
   reference across channels**, which is what ``preprocess.channel_noise``
   already uses: it is robust, it needs no electrode to be good, and it behaves
   the same way on 14 channels as on 32.
2. **The thresholds.**  ``Processing.residual_uv`` is 100 uV on the EPOC X and
   120 on the Flex, each a percentile of its own headset's amplitude, so the
   "% of task time flagged" numbers in the two results.json files are *not*
   comparable.  The exceedance rates here are recomputed at one common
   threshold, and a self-normalised ratio (p99 / median of each channel's own
   1-s peak-to-peak) is reported next to them, which cancels any gain,
   reference or LSB difference between the headsets entirely.
3. **Bad channels.**  The Flex profile interpolates them and the EPOC X drops
   them.  Interpolation would launder a dead electrode into a plausible-looking
   trace, so epochs here are always cut with ``bad_channels="drop"``: every
   channel in the per-channel tables carries its own electrode's data.

What comes out is four tables per session -- per channel, per scalp region, per
quarter of the recording, and per matched site -- plus the one number that
decides whether a component is detectable at a site: the standard error of the
evoked mean, which is single-trial noise divided by the square root of the
trials that site kept.

Outputs go to ``data/device_quality/``: ``channels.csv`` (one row per channel
per session), ``regions.csv``, ``matched_sites.csv``, ``quarters.csv``,
``timeline.csv``, ``summary.json``, ``device_quality.png`` and ``timeline.png``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import emotiv as em  # noqa: E402
import gng_erp  # noqa: E402
import oddball_erp  # noqa: E402

OUT = Path("data/device_quality")

# The sessions, in the order they were recorded.  ``order`` is what makes the
# cap-ageing question answerable: gng_flex and oddball_flex are the same cap,
# the same wetting, 45 minutes apart.
SESSIONS = [
    {"key": "gng_epocx", "xdf": "data/GnG/gng.xdf", "csv": "data/GnG/gng.csv",
     "task": "gng", "day": "2026-09-15", "order": 1,
     "note": "EPOC X Go/NoGo, first session"},
    {"key": "gng_flex", "xdf": "data/GnG/gng_flex.xdf", "csv": "data/GnG/gng_flex.csv",
     "task": "gng", "day": "2026-09-16", "order": 1,
     "note": "Flex Go/NoGo, ~15 min after the cap was wetted"},
    {"key": "oddball_flex", "xdf": "data/Oddball/oddball_flex.xdf", "csv": None,
     "task": "oddball", "day": "2026-09-16", "order": 2,
     "note": "Flex oddball, ~45 min after gng_flex on the same wetting"},
    {"key": "gng_epocx_2", "xdf": "data/GnG/epoc_gng_timingfixed.xdf",
     "csv": "data/GnG/epoc_gng_timingfixed.csv", "task": "gng", "day": "2026-09-16",
     "order": 2, "note": "EPOC X Go/NoGo re-run with the audio path fixed"},
]

# One common 1-s peak-to-peak threshold for both headsets, in the median
# reference.  120 uV is the Flex profile's own residual cut and the higher of the
# two, so neither headset is judged against a line derived from the other's noise
# floor; the self-normalised ratio beside it is the threshold-free version.
COMMON_P2P_UV = 120.0

# Scalp regions along the anterior-posterior axis, which is the axis the contact
# question is actually about.  Both montages are classified by the same rule, so
# "frontal" means the same band of the head on either headset even though the
# electrodes in it differ.
REGIONS = {
    "prefrontal": ("Fp1", "Fp2", "AF3", "AF4"),
    "frontal": ("F7", "F3", "Fz", "F4", "F8"),
    "fronto-central": ("FT9", "FC5", "FC1", "FC2", "FC6", "FT10"),
    "central": ("T7", "C3", "Cz", "C4", "T8"),
    "centro-parietal": ("CP5", "CP1", "CP2", "CP6"),
    "parietal": ("P7", "P3", "Pz", "P4", "P8"),
    "occipital": ("PO9", "O1", "Oz", "O2", "PO10"),
}
REGION_ORDER = list(REGIONS)
REGION_OF = {ch: name for name, chans in REGIONS.items() for ch in chans}

# The 14 EPOC X sites, with the two the Flex cap does not have mapped to their
# nearest equivalent (devices.FLEX_EQUIVALENT).  Comparing the headsets only on
# these answers "is the Flex's electronics/radio worse at the same site", which
# is a different question from "is the whole cap worse".
MATCHED_SITES = [(c, em.devices.FLEX_EQUIVALENT.get(c, c)) for c in em.EPOCX.labels]

N_QUARTERS = 4
# Bin for the fine timeline.  A Go/NoGo session has a ~24 s rest break in the
# middle and the oddball has two; at quarter resolution a break and a drift look
# the same, and they are the two explanations that have to be told apart.
TIMELINE_S = 15.0


# ---------------------------------------------------------------------------
# Per-session measurement


def session_events(spec: dict, run: em.Run):
    """This session's events and task mask, from the task script that owns them."""

    if spec["task"] == "gng":
        events, _ = gng_erp.task_events(run, spec["csv"])
        return events, gng_erp.task_mask(run, events)
    events, _ = oddball_erp.task_events(run)
    return events, oddball_erp.task_mask(run, events)


def median_reference(cleaned: em.Cleaned) -> tuple[np.ndarray, list[str]]:
    """ERP-band data in the median-across-channels reference, in microvolts.

    The median rather than the mean so that one bad electrode cannot move the
    reference, and so that the reference does not depend on how many electrodes
    the cap has -- which is the whole point of the comparison.
    """

    x = cleaned.raw.get_data() * 1e6
    return x - np.median(x, axis=0, keepdims=True), list(cleaned.raw.ch_names)


def break_windows(run: em.Run, events: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    """Mask of analysis windows that fall in a rest break between blocks.

    Both tasks run at SOAs of a few seconds, so a gap over 20 s between
    consecutive stimuli is a break and nothing else is.  The distinction matters
    because a break is when the participant moves, and an ERP never uses those
    seconds: a stability number computed over them describes how the cap
    survives head movement, which is worth knowing separately but must not be
    mixed into how the cap behaves while a trial is being recorded.
    """

    stim = np.sort(events["time"].to_numpy())
    gaps = np.diff(stim)
    mask = np.zeros(len(starts), bool)
    t_win = run.t[np.clip(starts, 0, len(run.t) - 1)]
    for i in np.flatnonzero(gaps > 20.0):
        mask |= (t_win >= stim[i]) & (t_win < stim[i + 1])
    return mask


def channel_table(run: em.Run, cleaned: em.Cleaned, epochs, timing: dict,
                  events: pd.DataFrame) -> pd.DataFrame:
    """One row per electrode: noise, stability, contact failures, ERP yield.

    Stability is reported twice, on stimulus time and on the rest breaks, because
    the two answer different questions and one 24 s break can otherwise dominate
    a whole session's numbers -- it does exactly that on gng_flex.xdf.
    """

    x, labels = median_reference(cleaned)
    task = cleaned.task_mask
    sfreq = float(run.sfreq_hz)
    p2p, starts = em.sliding_p2p(x, sfreq=sfreq)
    end = np.minimum(starts + int(round(em.preprocess.WIN_S * sfreq)) - 1, len(task) - 1)
    win = task[starts] & task[end]
    rest = break_windows(run, events, starts)
    p2p, in_break = p2p[:, win], rest[win]
    stim_only = p2p[:, ~in_break]

    noise = em.channel_noise(cleaned.raw).set_index("channel")
    rails = cleaned.rails
    pops = cleaned.pops["per_min"]

    # Single-trial noise and the standard error of the evoked mean, per channel,
    # counting only the trials in which that electrode was usable.  This is the
    # quantity a component has to beat, and it is what the extra electrodes have
    # to be paid for in.
    base = (epochs.times >= em.BASELINE[0]) & (epochs.times < em.BASELINE[1])
    data = epochs.get_data() * 1e6
    usable = ~epochs.metadata[[f"bad_{c}" for c in epochs.ch_names]].to_numpy(dtype=bool)

    rows = []
    for j, ch in enumerate(labels):
        row = {
            "session": run.key, "device": run.device, "channel": ch,
            "region": REGION_OF.get(ch, "other"),
            # Noise floor above the EEG, where the headset's own noise lives.
            "hf_20_40_uv_rms": float(np.sqrt(noise.loc[ch, "hf_20_40_uv2"])),
            "rms_1_20_uv": float(noise.loc[ch, "rms_1_20_uv"]),
            # Amplitude stability on task time, at one threshold for both headsets.
            "p2p_median_uv": float(np.median(stim_only[j])),
            "p2p_p99_uv": float(np.percentile(stim_only[j], 99)),
            # Threshold-free: how far the bad seconds sit above this channel's own
            # typical second.  Immune to gain, LSB and reference differences.
            "p2p_p99_over_median": float(np.percentile(stim_only[j], 99)
                                         / np.median(stim_only[j])),
            "pct_windows_over_common": float(100 * (stim_only[j] > COMMON_P2P_UV).mean()),
            # The same channel during the rest breaks: how it survives movement.
            "pct_break_windows_over_common": (float(100 * (p2p[j, in_break] > COMMON_P2P_UV).mean())
                                              if in_break.any() else float("nan")),
            # Everything including the breaks, which is what a naive pass reports.
            "pct_all_windows_over_common": float(100 * (p2p[j] > COMMON_P2P_UV).mean()),
            "pops_per_min": float(pops[ch]),
            "rail_pct": float(rails[ch]) if len(rails) else float("nan"),
            "marked_bad": ch in cleaned.bads or ch in timing["bad_channels"],
        }
        if ch in epochs.ch_names:
            k = epochs.ch_names.index(ch)
            keep = usable[:, k]
            trials = data[keep][:, k][:, base]
            row["erp_trials"] = int(keep.sum())
            # Across-trial SD at each baseline sample: the single-trial noise the
            # average has to divide down.
            row["single_trial_noise_uv"] = (float(np.sqrt(np.mean(trials.std(axis=0) ** 2)))
                                            if keep.sum() > 2 else float("nan"))
            row["evoked_sem_uv"] = (row["single_trial_noise_uv"] / np.sqrt(keep.sum())
                                    if keep.sum() > 2 else float("nan"))
        else:  # dropped before epoching
            row.update(erp_trials=0, single_trial_noise_uv=float("nan"),
                       evoked_sem_uv=float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def quarter_table(run: em.Run, cleaned: em.Cleaned) -> pd.DataFrame:
    """The same stability measure in each quarter of task time, per channel.

    A saline cap that is drying out should show this rising through the session;
    a cap that was simply seated badly from the start should not.
    """

    x, labels = median_reference(cleaned)
    task = cleaned.task_mask
    sfreq = float(run.sfreq_hz)
    p2p, starts = em.sliding_p2p(x, sfreq=sfreq)
    end = np.minimum(starts + int(round(em.preprocess.WIN_S * sfreq)) - 1, len(task) - 1)
    keep = np.flatnonzero(task[starts] & task[end])
    pops = cleaned.pops["times_s"]
    t0, t1 = run.t[np.flatnonzero(task)[[0, -1]]]
    edges = np.linspace(t0, t1, N_QUARTERS + 1)

    rows = []
    for q in range(N_QUARTERS):
        a, b = edges[q], edges[q + 1]
        sel = keep[(run.t[starts[keep]] >= a) & (run.t[starts[keep]] < b)]
        if not len(sel):
            continue
        minutes = (b - a) / 60.0
        for j, ch in enumerate(labels):
            rows.append({
                "session": run.key, "device": run.device, "channel": ch,
                "region": REGION_OF.get(ch, "other"), "quarter": q + 1,
                "start_min": a / 60.0,
                "p2p_median_uv": float(np.median(p2p[j, sel])),
                "pct_windows_over_common": float(100 * (p2p[j, sel] > COMMON_P2P_UV).mean()),
                "pops_per_min": float(sum(a <= t < b for t in pops[ch]) / minutes),
            })
    return pd.DataFrame(rows)


def timeline_table(run: em.Run, cleaned: em.Cleaned, events: pd.DataFrame) -> pd.DataFrame:
    """Exceedance in short bins, flagged for whether the bin is inside a rest break.

    Quarter resolution cannot separate a cap that is drying out from a cap that
    lost contact once, during the movement in a rest break.  A bin between the
    last stimulus of one block and the first of the next is marked ``break``;
    everything else is stimulus time, and that is where a drift would have to
    show up to matter.
    """

    x, labels = median_reference(cleaned)
    task = cleaned.task_mask
    sfreq = float(run.sfreq_hz)
    p2p, starts = em.sliding_p2p(x, sfreq=sfreq)
    end = np.minimum(starts + int(round(em.preprocess.WIN_S * sfreq)) - 1, len(task) - 1)
    keep = np.flatnonzero(task[starts] & task[end])
    t_win = run.t[starts[keep]]

    # Rest breaks: any gap between consecutive stimuli longer than 20 s.  Both
    # tasks run at SOAs of one to a few seconds, so nothing else comes close.
    stim = np.sort(events["time"].to_numpy())
    gaps = np.diff(stim)
    breaks = [(stim[i], stim[i + 1]) for i in np.flatnonzero(gaps > 20.0)]

    t0, t1 = run.t[np.flatnonzero(task)[[0, -1]]]
    edges = np.arange(t0, t1 + TIMELINE_S, TIMELINE_S)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = keep[(t_win >= a) & (t_win < b)]
        if not len(sel):
            continue
        block = p2p[:, sel]
        rows.append({
            "session": run.key, "device": run.device,
            "t_min": a / 60.0, "minutes_into_task": (a - t0) / 60.0,
            "in_break": any(s < b and a < e for s, e in breaks),
            # Averaged over channels, so a 14- and a 32-channel cap are on the
            # same scale: the share of channel-seconds that were unusable.
            "pct_channel_seconds_over_common": float(100 * (block > COMMON_P2P_UV).mean()),
            "worst_channel_pct": float(100 * (block > COMMON_P2P_UV).mean(axis=1).max()),
            "p2p_median_uv": float(np.median(block)),
        })
    return pd.DataFrame(rows)


def measure(spec: dict) -> dict:
    """Load, clean, epoch and measure one session."""

    run = em.load(spec["xdf"], key=spec["key"])
    events, task = session_events(spec, run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = em.clean(run, task_mask=task,
                           manual_bads=gng_erp.MANUAL_BADS.get(Path(spec["xdf"]).stem, {}))
        # Dropped, never interpolated: a per-channel quality table must not
        # contain a channel whose samples are a spline of its neighbours.
        epochs, timing = em.erp_epochs(
            run, cleaned, events, bad_channels="drop",
            chain_latency_s=(gng_erp.AUDIO_OFFSET_S.get(Path(spec["xdf"]).stem, 0.0)
                             if spec["task"] == "gng" else 0.0))
    return {
        "spec": spec, "run": run, "cleaned": cleaned, "timing": timing,
        "integrity": em.integrity(run, cleaned.raw),
        "channels": channel_table(run, cleaned, epochs, timing, events),
        "quarters": quarter_table(run, cleaned),
        "timeline": timeline_table(run, cleaned, events),
        "n_epochs": len(epochs),
    }


# ---------------------------------------------------------------------------
# Reporting


def region_summary(channels: pd.DataFrame) -> pd.DataFrame:
    """Per-session, per-region medians, in anterior-to-posterior order."""

    agg = (channels.groupby(["session", "device", "region"])
           .agg(n_channels=("channel", "size"),
                n_bad=("marked_bad", "sum"),
                hf_uv=("hf_20_40_uv_rms", "median"),
                p2p_median_uv=("p2p_median_uv", "median"),
                p2p_ratio=("p2p_p99_over_median", "median"),
                pct_over_common=("pct_windows_over_common", "median"),
                pct_over_in_breaks=("pct_break_windows_over_common", "median"),
                pops_per_min=("pops_per_min", "median"),
                single_trial_uv=("single_trial_noise_uv", "median"),
                evoked_sem_uv=("evoked_sem_uv", "median"))
           .reset_index())
    agg["region"] = pd.Categorical(agg["region"], REGION_ORDER + ["other"], ordered=True)
    return agg.sort_values(["session", "region"])


def matched_sites(channels: pd.DataFrame) -> pd.DataFrame:
    """The 14 EPOC X sites on both headsets, side by side.

    The two sites the Flex cap does not have (AF3/AF4) are represented by their
    nearest equivalents, Fp1/Fp2, which are lower on the forehead and see more of
    the eyes -- so a Flex disadvantage at those two rows is expected and is not
    evidence about the hardware.
    """

    lookup = {(r.session, r.channel): r for r in channels.itertuples()}
    rows = []
    for site, flex_ch in MATCHED_SITES:
        for session in channels["session"].unique():
            device = channels.loc[channels["session"] == session, "device"].iloc[0]
            ch = flex_ch if device == "flex" else site
            rec = lookup.get((session, ch))
            if rec is None:
                continue
            rows.append({"site": site, "session": session, "device": device, "channel": ch,
                         "hf_20_40_uv_rms": rec.hf_20_40_uv_rms,
                         "p2p_p99_over_median": rec.p2p_p99_over_median,
                         "pct_windows_over_common": rec.pct_windows_over_common,
                         "pops_per_min": rec.pops_per_min,
                         "single_trial_noise_uv": rec.single_trial_noise_uv,
                         "marked_bad": rec.marked_bad})
    frame = pd.DataFrame(rows)
    frame["site"] = pd.Categorical(frame["site"], [s for s, _ in MATCHED_SITES], ordered=True)
    return frame.sort_values(["site", "session"])


def figure(channels: pd.DataFrame, quarters: pd.DataFrame, path: Path) -> None:
    """Four panels: noise and stability by region, ERP yield, and drift in time."""

    sessions = list(channels["session"].unique())
    colour = {s: c for s, c in zip(sessions, ["#2166ac", "#b2182b", "#d6604d", "#4393c3"])}
    marker = {"epocx": "o", "flex": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    regions = region_summary(channels)
    xpos = {r: i for i, r in enumerate(REGION_ORDER)}
    for ax, col, label in (
            (axes[0, 0], "hf_uv", "20–40 Hz noise (µV rms, median reference)"),
            (axes[0, 1], "p2p_ratio", "1-s p2p: p99 / own median (contact instability)"),
            (axes[1, 0], "evoked_sem_uv", "SEM of the evoked mean (µV)")):
        for session in sessions:
            sub = regions[(regions["session"] == session) & regions["region"].isin(REGION_ORDER)]
            device = sub["device"].iloc[0]
            ax.plot([xpos[r] for r in sub["region"]], sub[col], marker=marker[device],
                    color=colour[session], lw=1.6, ms=6, label=session)
        ax.set_xticks(range(len(REGION_ORDER)))
        ax.set_xticklabels(REGION_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)

    ax = axes[1, 1]
    for session in sessions:
        sub = quarters[quarters["session"] == session]
        device = sub["device"].iloc[0]
        by_q = sub.groupby("quarter")["p2p_median_uv"].median()
        ax.plot(by_q.index, by_q.values, marker=marker[device], color=colour[session],
                lw=1.6, ms=6, label=session)
    ax.set_xticks(range(1, N_QUARTERS + 1))
    ax.set_xlabel("quarter of task time")
    ax.set_ylabel("median 1-s p2p across channels (µV)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("EPOC X vs Flex 1.0 — signal quality by scalp region and time on head",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline_figure(timeline: pd.DataFrame, path: Path) -> None:
    """Unusable channel-seconds against time on head, with the rest breaks shaded.

    This is the panel that separates the two stories: a cap drying out is a
    rising floor between the breaks, while a cap that lets go when the head moves
    is spikes that sit on them.
    """

    sessions = list(timeline["session"].unique())
    fig, axes = plt.subplots(len(sessions), 1, figsize=(11, 2.3 * len(sessions)),
                             sharex=False, squeeze=False)
    for ax, session in zip(axes[:, 0], sessions):
        sub = timeline[timeline["session"] == session]
        ax.plot(sub["minutes_into_task"], sub["pct_channel_seconds_over_common"],
                color="#b2182b", lw=1.3)
        for row in sub[sub["in_break"]].itertuples():
            ax.axvspan(row.minutes_into_task, row.minutes_into_task + TIMELINE_S / 60,
                       color="0.6", alpha=0.35, lw=0)
        ax.set_title(f"{session} ({sub['device'].iloc[0]}) — grey: rest break", fontsize=9)
        ax.set_ylabel("% ch-s\nover threshold", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("minutes into the task")
    fig.suptitle(f"Unusable channel-seconds (1-s p2p > {COMMON_P2P_UV:.0f} µV, "
                 "median reference) against time on head", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def acquisition_row(result: dict) -> dict:
    """The radio/decoder side of the comparison, in units both headsets share."""

    integrity, run = result["integrity"], result["run"]
    if run.device == "epocx":
        lost = integrity["missing_samples"]
        runs = {}
        worst = 0.0
    else:
        lost = integrity["filled_samples"]
        runs = integrity["filled_run_lengths"]
        worst = integrity["longest_fill_s"]
    return {
        "session": run.key, "device": run.device,
        "duration_min": integrity["duration_s"] / 60,
        "task_min": result["cleaned"].summary()["task_minutes"],
        "lost_samples": int(lost),
        "lost_pct": 100 * lost / len(run.t),
        "lost_seconds_total": lost / run.sfreq_hz,
        "gap_events": (integrity["filled_runs"] if run.device == "flex"
                       else int(np.sum(np.diff(np.flatnonzero(run.filled)) > 1) + bool(lost))),
        "gap_sample_lengths": runs,
        "longest_gap_s": worst,
        "gap_step_rms_uv": integrity.get("fill8_step_rms_uv_median_channel"),
        "dropped_repeats": integrity["dropped_repeats"],
        "rate_hz": (integrity["headset_rate_hz"][0] if run.device == "epocx"
                    else integrity["timestamp_segments"][0]["rate_hz"]),
        "lsb_uv": run.dev.lsb_uv,
        "bad_channels": result["timing"]["bad_channels"],
        "n_channels": len(run.labels),
        "n_good": len(run.labels) - len(result["timing"]["bad_channels"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", default=None,
                        help="comma-separated session keys, for a quick re-run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = SESSIONS
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in SESSIONS if s["key"] in wanted]

    results = []
    for spec in specs:
        print(f"measuring {spec['key']} ({spec['xdf']}) ...", flush=True)
        results.append(measure(spec))

    channels = pd.concat([r["channels"] for r in results], ignore_index=True)
    quarters = pd.concat([r["quarters"] for r in results], ignore_index=True)
    timeline = pd.concat([r["timeline"] for r in results], ignore_index=True)
    regions = region_summary(channels)
    matched = matched_sites(channels)
    acquisition = pd.DataFrame([acquisition_row(r) for r in results])

    channels.to_csv(args.out / "channels.csv", index=False)
    quarters.to_csv(args.out / "quarters.csv", index=False)
    timeline.to_csv(args.out / "timeline.csv", index=False)
    regions.to_csv(args.out / "regions.csv", index=False)
    matched.to_csv(args.out / "matched_sites.csv", index=False)
    figure(channels, quarters, args.out / "device_quality.png")
    timeline_figure(timeline, args.out / "timeline.png")

    summary = {
        "sessions": [r["spec"] for r in results],
        "common_p2p_uv": COMMON_P2P_UV,
        "reference_for_noise": "median across channels",
        "acquisition": acquisition.to_dict("records"),
        "regions": regions.to_dict("records"),
        "matched_sites": matched.to_dict("records"),
        "timeline_break_vs_stimulus": (
            timeline.groupby(["session", "device", "in_break"])
            ["pct_channel_seconds_over_common"].agg(["size", "mean", "max"])
            .reset_index().to_dict("records")),
        "channels": channels.to_dict("records"),
    }
    (args.out / "summary.json").write_text(
        json.dumps(em.to_jsonable(summary), indent=2), encoding="utf-8")

    pd.set_option("display.width", 200, "display.max_rows", 400,
                  "display.float_format", lambda v: f"{v:.2f}")
    print("\n=== acquisition ===")
    print(acquisition.drop(columns=["gap_sample_lengths", "bad_channels"]).to_string(index=False))
    for row in acquisition.itertuples():
        print(f"  {row.session}: gaps {row.gap_sample_lengths}, bad {row.bad_channels}")
    print("\n=== by region (medians) ===")
    print(regions.to_string(index=False))
    print("\n=== matched EPOC X sites ===")
    print(matched.to_string(index=False))
    print("\n=== per channel ===")
    print(channels.sort_values(["session", "region", "channel"]).to_string(index=False))
    print("\n=== stimulus time vs rest breaks (% of channel-seconds over "
          f"{COMMON_P2P_UV:.0f} uV p2p) ===")
    print(timeline.groupby(["session", "device", "in_break"])
          ["pct_channel_seconds_over_common"].agg(["size", "mean", "max"]).round(2).to_string())
    print(f"\nwrote {args.out}/channels.csv, quarters.csv, timeline.csv, regions.csv, "
          "matched_sites.csv, summary.json, device_quality.png, timeline.png")


if __name__ == "__main__":
    main()
