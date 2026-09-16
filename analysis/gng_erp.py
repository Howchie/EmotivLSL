#!/usr/bin/env python3
"""Go/NoGo ERPs from an EPOC X or Flex 1.0 recording.

Everything device-shaped -- loading, the sample-grid rebuild, the reference, the
artifact thresholds, the bad-channel policy, ICA, timing correction -- lives in
the ``emotiv`` package and comes from that headset's ``Processing`` profile in
``devices.py``.  The EPOC X pipeline is documented in ``EPOCX_ERP_PIPELINE.md``.
What is left here is the part that is specific to this task: which markers are Go
and NoGo, which of them the participant got right, and the ERP contrast worth
plotting.  The same script therefore runs on either headset::

    python analysis/gng_erp.py                                     # EPOC X
    python analysis/gng_erp.py --xdf data/GnG/gng_flex.xdf \
        --csv data/GnG/gng_flex.csv --out data/GnG/gng_flex_erp    # Flex 1.0

The ROIs come from the device too, and the two headsets do not have the same
ones.  The 32-channel Flex cap has midline sites, so it gets the canonical
centro-parietal P3 ROI and reports that as its primary contrast; the 14-sensor
EPOC X ring has no midline and falls back to its frontal ROI.  An ROI the device
does not define is skipped rather than faked.

Epochs are locked to the physical sound: ``emotiv.erp_epochs`` shifts the events
by the headset's measured chain latency, so a latency read off these waveforms is
a real latency and needs no further correction.
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

import emotiv as em

XDF = "data/GnG/gng.xdf"
CSV = "data/GnG/gng.csv"
OUT = Path("data/GnG/gng_erp")

# Known-bad channels are a property of one recording, not of the headset, so they
# are keyed by the file.  Everything else about a channel is decided from the data.
MANUAL_BADS = {
    # F4's saline pad was failing during this recording: a ~200 ms ramp, a sharp
    # spike, then a ~50 uV offset recovering over ~2 s, recurring every 30-45 s.
    "gng": {"F4": "old saline pad: recurring electrode pops"},
}

# ROIs are reported in this order; one the device leaves empty is skipped.  The
# first one present is the primary contrast quoted in the summary and the title.
ROI_ORDER = ("central", "frontal", "posterior", "inferior")

CODES = {"go": 5, "nogo": 6}
COLORS = {"go": "#b2182b", "nogo": "#2166ac", "diff": "black"}

# The window the Go-NoGo positivity is summarised over.  It is descriptive -- it
# was chosen after looking at the topography -- and the cluster test below is the
# inference.  It is per recording because ``erp_epochs`` corrects the EEG chain
# and nothing else, so a session whose audio path delayed the sound after the
# marker sits on a shifted time base:
#
# * ``gng``, the EPOC X session of 2026-09-15, played through the USB speaker on
#   the pre-WASAPI PsychoPy path.  Neither delay was measured for that session,
#   so its epochs are ~130 ms early relative to the physical sound.  Three
#   independent estimates agree: its keypress RT from the marker is 128 ms longer
#   than the Flex session's, and its NoGo N1 and P2 are both 133 ms later.  Its
#   window is kept where it was originally placed, on that shifted base.
# * ``gng_flex``, the Flex session of 2026-09-16, played through the headphone
#   jack on WASAPI exclusive mode, whose output latency is 0.2 ms [-0.6, 0.9]
#   (analysis/headphone_timing.py).  Its markers need no audio term, so it gets
#   the canonical post-stimulus window.
#
# The two sessions' latencies are therefore NOT directly comparable.
DEFAULT_P3_WINDOW = (0.30, 0.50)
P3_WINDOW = {
    "gng": (0.34, 0.49),  # 300-500 ms on that session's ~130 ms-shifted time base
    "gng_flex": DEFAULT_P3_WINDOW,
}


# ---------------------------------------------------------------------------
# The task


def task_events(run: em.Run, csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Go/NoGo events, keeping only the trials the participant got right.

    Rare Go = 1500 Hz ("high", keypress), frequent NoGo = 1000 Hz ("low").  The
    markers carry the stimulus; whether a key was pressed only exists in the
    PsychoPy CSV, so the two are matched in order and checked.
    """

    csv = pd.read_csv(csv_path)
    rows = csv[csv["stim"].isin(["high", "low"])].reset_index(drop=True)
    events = em.events_from_markers(run, em.split_condition({"GnG": "gng"}, ("high", "low")))
    if len(events) != len(rows):
        raise RuntimeError(f"{len(events)} GnG markers but {len(rows)} CSV task rows")
    marker_stim = events["condition"].str.split("/").str[-1].to_numpy()
    if not (marker_stim == rows["stim"].to_numpy()).all():
        raise RuntimeError("GnG markers and CSV rows are out of step")

    pressed = rows["choice_resp.keys"].apply(lambda v: isinstance(v, str)).to_numpy()
    is_go = rows["stim"].to_numpy() == "high"
    # choice_resp's clock starts at the routine's first flip and the tone is
    # scheduled `soa` after that, so rt - soa is the RT from the *marker*, 0-1
    # frame short.  It is not the RT from the sound: that would need this
    # session's audio-output latency, which was never measured, and it is not the
    # chain latency -- that delays the EEG, not the keypress.
    rt = np.where(pressed, rows["choice_resp.rt"].to_numpy(float) - rows["soa"].to_numpy(float), np.nan)
    events = events.assign(condition=np.where(is_go, "go", "nogo"), pressed=pressed,
                           correct=pressed == is_go, rt_from_marker_s=rt,
                           block=rows["blocks.thisN"].astype(int).to_numpy())
    hit_rt = rt[is_go & pressed]
    behaviour = {
        "go_trials": int(is_go.sum()), "nogo_trials": int((~is_go).sum()),
        "hits": int((is_go & pressed).sum()), "false_alarms": int((~is_go & pressed).sum()),
        "hit_rate": float((pressed[is_go]).mean()),
        "false_alarm_rate": float((pressed[~is_go]).mean()),
        "rt_reference": "from the tone marker; audio-output latency for this session is unmeasured",
        "rt_median_s": float(np.nanmedian(hit_rt)),
        "rt_iqr_s": [float(v) for v in np.nanpercentile(hit_rt, [25, 75])],
    }
    return events[events["correct"]].copy(), behaviour


def task_mask(run: em.Run, events: pd.DataFrame, pad_s: float = 2.0) -> np.ndarray:
    """Samples belonging to the task, so idle time does not count towards QC rates."""

    mask = np.zeros(len(run.t), bool)
    a, b = np.searchsorted(run.t, [events["time"].min() - pad_s, events["time"].max() + pad_s])
    mask[a:b] = True
    return mask


# ---------------------------------------------------------------------------
# Measurement


def device_rois(run: em.Run, epochs) -> dict[str, list[str]]:
    """The ROIs this headset defines, in report order, keeping present channels only."""

    out = {}
    for name in ROI_ORDER:
        roi = [c for c in getattr(run.dev, f"{name}_roi", ()) if c in epochs.ch_names]
        if roi:
            out[name] = roi
    if not out:
        raise RuntimeError(f"none of {run.device}'s ROI channels survived cleaning")
    return out


def contrast(epochs, roi: list[str], rng, p3_window: tuple[float, float]) -> dict:
    """Go, NoGo and their difference over an ROI, with trial-bootstrap CIs."""

    present = [c for c in roi if c in epochs.ch_names]
    go, nogo = em.roi_trials(epochs["go"], present), em.roi_trials(epochs["nogo"], present)
    times = epochs.times
    diff, lo, hi, boots = em.boot_diff_ci(go, nogo, rng)
    window = (times >= p3_window[0]) & (times <= p3_window[1])
    peak_i = int(np.flatnonzero(window)[np.argmax(diff[window])])
    peak_lat = times[np.flatnonzero(window)[np.argmax(boots[:, window], axis=1)]]
    baseline = (times >= -0.2) & (times < 0)
    return {
        "roi": present,
        "n_go": int(len(go)), "n_nogo": int(len(nogo)),
        "go_minus_nogo_uv": float(diff[window].mean()),
        "ci95_uv": [float(lo[window].mean()), float(hi[window].mean())],
        "window_s": list(p3_window),
        "peak_latency_s": float(times[peak_i]),
        "peak_latency_ci95_s": [float(v) for v in np.percentile(peak_lat, [2.5, 97.5])],
        "peak_uv": float(diff[peak_i]),
        "baseline_rms_uv": float(np.sqrt((go.mean(0)[baseline] ** 2).mean())),
        "plus_minus_rms_uv": em.plus_minus_rms(go, baseline, rng),
        "_times": times, "_go": em.boot_mean_ci(go, rng), "_nogo": em.boot_mean_ci(nogo, rng),
        "_diff": (diff, lo, hi), "_peak_i": peak_i,
    }


def topography(epochs, peak_i: int, reference: str) -> dict:
    """Go - NoGo at every good channel at the peak.

    The check that matters on the EPOC X ring: if every channel is positive, an
    average reference would subtract the effect and flip the inferior sites
    negative, which is why its recorded reference is kept.  Under an average
    reference the channels sum to zero by construction, so the same-sign test
    cannot be positive and says nothing -- ``reference`` is recorded here so the
    number is not read as a finding.
    """

    go = epochs["go"].get_data()[:, :, peak_i] * 1e6
    nogo = epochs["nogo"].get_data()[:, :, peak_i] * 1e6
    per = {c: float(go[:, j].mean() - nogo[:, j].mean()) for j, c in enumerate(epochs.ch_names)}
    return {"reference": reference,
            "same_sign_test_meaningful": reference == "recorded",
            "per_channel_uv": per, "channel_mean_uv": float(np.mean(list(per.values()))),
            "all_channels_same_sign": bool(all(v > 0 for v in per.values())
                                           or all(v < 0 for v in per.values()))}


def by_block(epochs, roi: list[str], rng, p3_window: tuple[float, float]) -> dict:
    """Does the effect replicate in each block?"""

    present = [c for c in roi if c in epochs.ch_names]
    times = epochs.times
    window = (times >= p3_window[0]) & (times <= p3_window[1])
    out = {}
    for block in sorted(epochs.metadata["block"].unique()):
        sel = epochs[epochs.metadata["block"] == block]
        if not len(sel["go"]) or not len(sel["nogo"]):
            continue
        go, nogo = em.roi_trials(sel["go"], present), em.roi_trials(sel["nogo"], present)
        diff, lo, hi, _ = em.boot_diff_ci(go, nogo, rng)
        out[f"block{block}"] = {"n_go": int(len(go)), "n_nogo": int(len(nogo)),
                                "go_minus_nogo_uv": float(diff[window].mean()),
                                "ci95_uv": [float(lo[window].mean()), float(hi[window].mean())]}
    return out


# ---------------------------------------------------------------------------
# Figure


def figure(result: dict, rois: dict, topo: dict, clusters: dict, path: Path) -> None:
    n = len(rois)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 7.5), squeeze=False)
    axes = np.atleast_2d(axes)

    for col, (name, res) in enumerate(rois.items()):
        t = 1000 * res["_times"]
        ax = axes[0, col]
        for key, label in (("_go", "Go"), ("_nogo", "NoGo")):
            mean, lo, hi = res[key]
            ax.plot(t, mean, color=COLORS[label.lower()], lw=1.6, label=f"{label} (n={res['n_' + label.lower()]})")
            ax.fill_between(t, lo, hi, color=COLORS[label.lower()], alpha=0.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_title(f"{name}: {' '.join(res['roi'])}", fontsize=10)
        ax.set_xlabel("ms after the sound")
        ax.set_ylabel("µV" if col == 0 else "")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        diff, lo, hi = res["_diff"]
        ax.plot(t, diff, color=COLORS["diff"], lw=1.8)
        ax.fill_between(t, lo, hi, color=COLORS["diff"], alpha=0.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.6", lw=0.8)
        window = result["p3_window_s"]
        ax.axvspan(1000 * window[0], 1000 * window[1], color="tab:green", alpha=0.08)
        for cluster in clusters["clusters"][:3]:
            if cluster["p"] < 0.05:
                ax.axvspan(1000 * cluster["t_start"], 1000 * cluster["t_end"],
                           color="tab:orange", alpha=0.10, lw=0)
        ax.set_title(f"Go − NoGo: {res['go_minus_nogo_uv']:.2f} µV "
                     f"[{res['ci95_uv'][0]:.2f}, {res['ci95_uv'][1]:.2f}]", fontsize=10)
        ax.set_xlabel("ms after the sound")
        ax.set_ylabel("µV" if col == 0 else "")

    best = clusters["clusters"][0] if clusters["clusters"] else None
    note = (f"largest cluster {1000 * best['t_start']:.0f}–{1000 * best['t_end']:.0f} ms, "
            f"{best['sign']}, p={best['p']:.3f}" if best else "no cluster")
    sign = (f", all same sign: {topo['all_channels_same_sign']}"
            if topo["same_sign_test_meaningful"] else "")
    fig.suptitle(
        f"{result['device'].upper()} Go/NoGo, locked to the sound (events shifted "
        f"{1000 * result['timing']['chain_latency_s']:.1f} ms for the EEG chain, "
        f"{topo['reference']} reference).  {note}.  Channel mean at the "
        f"{result['primary_roi']} peak {topo['channel_mean_uv']:+.2f} µV{sign}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default=XDF)
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--chain-latency", type=float, default=None,
                        help="override the device's measured chain latency (s)")
    parser.add_argument("--p3-window", default=None, metavar="START,END",
                        help="summary window in seconds after the stimulus; the default "
                             "is this recording's entry in P3_WINDOW")
    parser.add_argument("--heog", default=None, metavar="LEFT,RIGHT",
                        help="left/right frontal pair for the saccade proxy, for a Flex "
                             "montage without the profile's default pair")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(em.SEED)
    stem = Path(args.xdf).stem
    p3_window = (tuple(float(v) for v in args.p3_window.split(","))
                 if args.p3_window else P3_WINDOW.get(stem, DEFAULT_P3_WINDOW))

    run = em.load(args.xdf)
    events, behaviour = task_events(run, args.csv)
    cleaned = em.clean(run, task_mask=task_mask(run, events),
                       manual_bads=MANUAL_BADS.get(stem, {}),
                       heog=tuple(args.heog.split(",")) if args.heog else None)
    epochs, timing = em.erp_epochs(run, cleaned, events, codes=CODES,
                                   chain_latency_s=args.chain_latency)

    picks = device_rois(run, epochs)
    primary = next(iter(picks))
    rois = {name: contrast(epochs, roi, rng, p3_window) for name, roi in picks.items()}
    topo = topography(epochs, rois[primary]["_peak_i"], run.dev.processing.reference)
    clusters = em.cluster_test(epochs["go"], epochs["nogo"], tmin=0.0, tmax=1.0)
    n1 = em.n1_p2(epochs["nogo"], picks["frontal"], picks["inferior"], rng)

    result = {
        "recording": args.xdf,
        "device": run.device,
        "primary_roi": primary,
        "p3_window_s": list(p3_window),
        "integrity": em.integrity(run, cleaned.raw),
        "preprocessing": cleaned.summary(),
        "timing": timing,
        "behaviour": behaviour,
        "rois": {k: em.to_jsonable(v) for k, v in rois.items()},
        "topography_at_primary_peak": topo,
        "clusters": clusters["clusters"],
        "nogo_n1_p2": em.to_jsonable(n1),
        "blocks_primary": by_block(epochs, picks[primary], rng, p3_window),
    }
    (args.out / "results.json").write_text(json.dumps(em.to_jsonable(result), indent=2), encoding="utf-8")
    events.to_csv(args.out / "events.csv", index=False)
    figure(result, rois, topo, clusters, args.out / "erp.png")

    best_roi = rois[primary]
    print(f"{args.xdf} [{run.device}]: {run.duration_s:.0f} s, "
          f"{cleaned.summary()['task_minutes']:.1f} min of task")
    rails = cleaned.rails
    if len(rails):
        print(f"  rail fraction: median {rails.median():.2f}%, worst "
              f"{rails.idxmax()} {rails.max():.2f}% (cut {run.dev.processing.rail_pct}%)")
    handling = {"drop": "dropped", "interpolate": "interpolated"}[run.dev.processing.bad_channels]
    print(f"  {handling} {cleaned.bads}; ICA removed {cleaned.ica_exclude} "
          f"(|r| {', '.join(f'{k}={np.abs(v).max():.2f}' for k, v in cleaned.ica_corr.items())})")
    print(f"  events shifted {timing['shift_samples']} samples "
          f"({1000 * timing['chain_latency_s']:.1f} ms); kept {timing['n_by_condition']}")
    print(f"  hits {behaviour['hits']}/{behaviour['go_trials']}, "
          f"false alarms {behaviour['false_alarms']}/{behaviour['nogo_trials']}, "
          f"RT median {1000 * behaviour["rt_median_s"]:.0f} ms from the marker")
    for name, res in rois.items():
        mark = "*" if name == primary else " "
        print(f" {mark}{name:>9} ({' '.join(res['roi'])}) Go−NoGo "
              f"{res['go_minus_nogo_uv']:.2f} µV "
              f"[{res['ci95_uv'][0]:.2f}, {res['ci95_uv'][1]:.2f}] over "
              f"{1000 * p3_window[0]:.0f}–{1000 * p3_window[1]:.0f} ms, "
              f"peak {1000 * res['peak_latency_s']:.0f} ms")
    print(f"  {primary} baseline RMS {best_roi['baseline_rms_uv']:.2f} µV vs ± noise "
          f"{best_roi['plus_minus_rms_uv']:.2f} µV")
    same_sign = (f", all one sign: {topo['all_channels_same_sign']}"
                 if topo["same_sign_test_meaningful"] else
                 " (same-sign test is vacuous under an average reference)")
    print(f"  channel mean at the {primary} peak {topo['channel_mean_uv']:+.2f} µV"
          f"{same_sign}")
    for cluster in clusters["clusters"][:2]:
        print(f"  cluster {1000 * cluster['t_start']:.0f}–{1000 * cluster['t_end']:.0f} ms "
              f"{cluster['sign']} p={cluster['p']:.4f} ({len(cluster['channels'])} channels)")
    print(f"  NoGo N1 {1000 * n1['n1_latency_s']:.0f} ms, P2 {1000 * n1['p2_latency_s']:.0f} ms "
          f"(after the sound)")
    print(f"  wrote {args.out}/results.json, events.csv, erp.png")


if __name__ == "__main__":
    main()
