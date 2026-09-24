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

Epochs are locked to the physical sound: ``emotiv.load`` shifts every marker by
the headset's fixed measured chain latency, and ``emotiv.erp_epochs`` applies only
an optional session-specific audio-output residual.  A latency read off these
waveforms is therefore a physical stimulus latency and needs no chain correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import emotiv as em

XDF = "data/GnG/gng.xdf"
CSV = "data/GnG/gng.csv"
OUT = Path("data/GnG/gng_erp")

# Known-bad channels are a property of one recording, not of the headset, so they
# are keyed by the file.  Everything else about a channel is decided from the data.
# Kept as a record of what was seen by eye; the pipeline no longer depends on it,
# since F4 is now recovered by the bad-epoch share test (11.0% of epochs, cut 10%)
# even though its window share, 4.2%, sits under preprocess.CHANNEL_BAD_SHARE.
MANUAL_BADS = {
    # F4's saline pad was failing during this recording: a ~200 ms ramp, a sharp
    # spike, then a ~50 uV offset recovering over ~2 s, recurring every 30-45 s.
    "gng": {"F4": "old saline pad: recurring electrode pops"},
}

# ROIs are reported in this order; one the device leaves empty is skipped.  The
# first one present is the primary contrast quoted in the summary and the title.
ROI_ORDER = ("central", "frontal", "posterior", "inferior")

CODES = {"go": 5, "nogo": 6}
CONDITIONS = ("go", "nogo")  # the contrast is Go minus NoGo

# Per-session audio delay, added after the loader's fixed hardware marker shift,
# so that t=0 is the sound in every session and the latencies printed for one
# recording mean the same thing as the latencies printed for another.
#
# The loader applies the EEG chain to every marker because that is a property of
# the headset.  A session whose audio path put the sound *after* its marker needs
# its own term on top:
#
# * ``gng``, the EPOC X session of 2026-09-15, played through a custom
#   sounddevice stream on the pre-WASAPI PsychoPy path, and neither the output
#   latency nor the device delay was measured at the time.  The offset below is
#   an inference, not a measurement: three independent comparisons with the Flex
#   session of the next day (same task, same subject, WASAPI) agree on it -- the
#   keypress RT from the marker is 128 ms longer, the NoGo N1 is 133 ms later and
#   the P2 is 133 ms later.  Without it that session's N1 reads 211 ms.
# * ``gng_flex`` and ``epoc_gng_timingfixed`` played through the headphone jack on
#   WASAPI exclusive mode, whose output latency is 0.2 ms [-0.6, 0.9]
#   (analysis/headphone_timing.py).  They need no audio term.
AUDIO_OFFSET_S = {"gng": 0.133}

# The window the Go-NoGo positivity is summarised over.  It is descriptive -- it
# was chosen after looking at the waveform -- and the cluster test is the
# inference.  With the audio offset applied the component peaks at 265-340 ms
# after the sound in all three sessions, so 200-400 ms brackets it; the canonical
# 300-500 ms catches only its falling edge and was what made the Flex P3 look
# absent.  It is no longer per recording: every session is now on the same time
# base, so one window fits all of them.
P3_WINDOW = (0.20, 0.40)

# ---------------------------------------------------------------------------
# The task


def task_events(run: em.Run, csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Go/NoGo events, keeping only the trials the participant got right.

    Rare Go = 1500 Hz ("high"), frequent NoGo = 1000 Hz ("low").  The markers
    carry the stimulus; whether a key was pressed only exists in the PsychoPy
    CSV, so the two are matched in order and checked.

    A session may mix response blocks with silent-counting blocks, in which the
    rare tones are tallied and no key is pressed.  Those are detected per block
    (no press anywhere in the block), kept in full, and flagged in a
    ``responding`` column: a Go-NoGo difference measured in a counting block
    contains no motor activity at all.
    """

    csv = pd.read_csv(csv_path)
    rows = csv[csv["stim"].isin(["high", "low"])].reset_index(drop=True)
    events = em.events_from_markers(run, em.split_condition({"GnG": "gng"}, ("high", "low")))
    block_intervals, block_markers = em.paired_blocks(run)
    if len(events) != len(rows):
        raise RuntimeError(f"{len(events)} GnG markers but {len(rows)} CSV task rows")
    marker_stim = events["condition"].str.split("/").str[-1].to_numpy()
    if not (marker_stim == rows["stim"].to_numpy()).all():
        raise RuntimeError("GnG markers and CSV rows are out of step")
    # A complete block is the unit used by task-time QC. Keep the same rule for
    # epochs: an isolated tone outside every complete pair is an aborted or
    # malformed tail, not task data. Keep the diagnostic count rather than
    # silently making the CSV/marker mismatch look like a participant omission.
    if not block_intervals.empty:
        in_block = np.zeros(len(events), bool)
        times = events["time"].to_numpy(float)
        for interval in block_intervals.itertuples(index=False):
            in_block |= (times >= interval.start) & (times <= interval.end)
        outside = int((~in_block).sum())
        if outside:
            block_markers["events_outside_complete_blocks"] = outside
            events = events.loc[in_block].reset_index(drop=True)
            rows = rows.loc[in_block].reset_index(drop=True)

    pressed = rows["choice_resp.keys"].apply(lambda v: isinstance(v, str)).to_numpy()
    is_go = rows["stim"].to_numpy() == "high"
    block = rows["blocks.thisN"].astype(int).to_numpy()
    # A block in which nothing was ever pressed is a silent-counting block: the
    # participant tallies the rare tones instead of responding.  There is no
    # per-trial accuracy to score there, so every trial is kept, and the block is
    # marked so the ERP can be split by whether a motor response was made.
    responding = np.array([pressed[block == b].any() for b in block])
    # choice_resp's clock starts at the routine's first flip and the tone is
    # scheduled `soa` after that, so rt - soa is the RT from the *marker*, 0-1
    # frame short.  It is not the RT from the sound: that would need this
    # session's audio-output latency, which was never measured, and it is not the
    # chain latency -- that delays the EEG, not the keypress.
    rt = np.where(pressed, rows["choice_resp.rt"].to_numpy(float) - rows["soa"].to_numpy(float), np.nan)
    correct = np.where(responding, pressed == is_go, True)
    events = events.assign(condition=np.where(is_go, "go", "nogo"), pressed=pressed,
                           correct=correct, rt_from_marker_s=rt, block=block,
                           responding=responding)
    hit_rt = rt[is_go & pressed]
    resp = responding  # accuracy is only defined where a response was required
    behaviour = {
        "go_trials": int(is_go.sum()), "nogo_trials": int((~is_go).sum()),
        # Only blocks that actually contain rare trials.  Incomplete marker tails
        # have already been removed above and are retained only in diagnostics.
        "counting_blocks": sorted({int(b) for b in block[~responding & is_go]}),
        "responding_blocks": sorted({int(b) for b in block[responding & is_go]}),
        "go_trials_counting": int((is_go & ~resp).sum()),
        "go_trials_responding": int((is_go & resp).sum()),
        "hits": int((is_go & pressed).sum()), "false_alarms": int((~is_go & pressed).sum()),
        "hit_rate": float(pressed[is_go & resp].mean()) if (is_go & resp).any() else None,
        "false_alarm_rate": float(pressed[~is_go & resp].mean()) if (~is_go & resp).any() else None,
        "rt_reference": "from the tone marker; audio-output latency for this session is unmeasured",
        "rt_median_s": float(np.nanmedian(hit_rt)) if np.isfinite(hit_rt).any() else None,
        "rt_iqr_s": ([float(v) for v in np.nanpercentile(hit_rt, [25, 75])]
                     if np.isfinite(hit_rt).any() else None),
        "block_marker_diagnostics": block_markers,
    }
    return events[events["correct"]].copy(), behaviour


def task_mask(run: em.Run, events: pd.DataFrame, pad_s: float = 2.0) -> np.ndarray:
    """Union of the marked task blocks, so idle breaks do not count towards QC."""

    mask, _ = em.block_mask(run, pad_s=pad_s,
                            fallback_times=events["time"].to_numpy(float))
    return mask


# ---------------------------------------------------------------------------
# Measurement


def device_rois(run: em.Run, epochs) -> dict[str, list[str]]:
    """The ROIs this headset defines, in report order, real electrodes only.

    ``roi_picks`` drops channels that are missing and channels the profile
    interpolated, so a reported ROI is the electrodes it was actually measured
    from.
    """

    out = {}
    for name in ROI_ORDER:
        roi = em.roi_picks(epochs, getattr(run.dev, f"{name}_roi", ()))
        if roi:
            out[name] = roi
    if not out:
        raise RuntimeError(f"none of {run.device}'s ROI channels survived cleaning")
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default=XDF)
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--chain-latency", type=float, default=None,
                        help="replace the profile hardware marker shift (s)")
    parser.add_argument("--audio-offset", type=float, default=None, metavar="SECONDS",
                        help="additional delay from the shifted marker to the sound for "
                             "this session; the default is this recording's entry in "
                             "AUDIO_OFFSET_S (0 for normal sessions)")
    parser.add_argument("--p3-window", default=None, metavar="START,END",
                        help=f"summary window in seconds after the sound (default "
                             f"{P3_WINDOW[0]}-{P3_WINDOW[1]})")
    parser.add_argument("--reference", default=None, metavar="POLICY",
                        help='override the device reference: "recorded", "average", or a '
                             'comma-separated channel list to link (e.g. PO9,PO10)')
    parser.add_argument("--heog", default=None, metavar="LEFT,RIGHT",
                        help="left/right frontal pair for the saccade proxy, for a Flex "
                             "montage without the profile's default pair")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(em.SEED)
    stem = Path(args.xdf).stem
    p3_window = (tuple(float(v) for v in args.p3_window.split(","))
                 if args.p3_window else P3_WINDOW)
    audio_offset = (args.audio_offset if args.audio_offset is not None
                    else AUDIO_OFFSET_S.get(stem, 0.0))

    run = em.load(args.xdf)
    chain = args.chain_latency if args.chain_latency is not None else run.marker_shift_s
    if args.chain_latency is not None:
        # The loader has already applied the profile's fixed hardware delay.  An
        # explicit override replaces it for every marker, including block/rest
        # markers, rather than being applied only at epoching.
        run.shift_markers(chain - run.marker_shift_s)
    events, behaviour = task_events(run, args.csv)
    reference_arg = args.reference
    if reference_arg and "," in reference_arg:
        reference_arg = tuple(reference_arg.split(","))
    cleaned = em.clean(run, task_mask=task_mask(run, events),
                       manual_bads=MANUAL_BADS.get(stem, {}), reference=reference_arg,
                       heog=tuple(args.heog.split(",")) if args.heog else None)
    # Hardware latency has already shifted every marker during loading.  Only the
    # one recording-specific audio-path correction remains an epoch-level shift.
    epochs, timing = em.erp_epochs(run, cleaned, events, codes=CODES,
                                   chain_latency_s=audio_offset)
    timing["audio_offset_s"] = float(audio_offset)
    timing["hardware_marker_shift_s"] = float(run.marker_shift_s)
    timing["chain_latency_s"] = float(chain)
    timing["effective_marker_to_sound_s"] = float(run.marker_shift_s + audio_offset)

    picks = device_rois(run, epochs)
    primary = next(iter(picks))
    rois = {name: em.contrast(epochs, roi, rng, p3_window, CONDITIONS)
            for name, roi in picks.items()}
    reference = cleaned.parameters["reference"]
    topo = em.topography(epochs, rois[primary]["_peak_i"], reference, CONDITIONS)
    clusters = em.cluster_test(epochs["go"], epochs["nogo"], tmin=0.0, tmax=1.0,
                               bad_cells=run.dev.processing.bad_cells)
    # Interpolating a sporadic bad cell keeps the epoch, but the reader should be
    # able to see what the same test says when those epochs are simply dropped,
    # so the stricter answer is always computed and stored alongside it.
    strict = (em.cluster_test(epochs["go"], epochs["nogo"], tmin=0.0, tmax=1.0, bad_cells="complete")
              if run.dev.processing.bad_cells != "complete" else clusters)
    n1 = em.n1_p2(epochs["nogo"], picks["frontal"], picks["inferior"], rng)
    blocks = em.by_block(epochs, picks[primary], rng, p3_window, CONDITIONS,
                         bad_cells=run.dev.processing.bad_cells)

    result = {
        "recording": args.xdf,
        "device": run.device,
        "primary_roi": primary,
        "window_s": list(p3_window),
        "integrity": em.integrity(run, cleaned.raw),
        "preprocessing": cleaned.summary(),
        "timing": timing,
        "behaviour": behaviour,
        "rois": {k: em.to_jsonable(v) for k, v in rois.items()},
        "topography_at_primary_peak": topo,
        "clusters": clusters["clusters"],
        "cluster_n": [clusters["n_a"], clusters["n_b"]],
        "cluster_bad_cells": clusters["bad_cells"],
        "clusters_complete_cases": strict["clusters"],
        "cluster_n_complete_cases": [strict["n_a"], strict["n_b"]],
        "nogo_n1_p2": em.to_jsonable(n1),
        "blocks_primary": blocks,
    }
    (args.out / "results.json").write_text(json.dumps(em.to_jsonable(result), indent=2), encoding="utf-8")
    events.to_csv(args.out / "events.csv", index=False)
    # Deliberately short: the timing correction and the trial counts are printed
    # and stored, and repeating them on the figure is what made it unreadable.
    title = f"{run.device.upper()} Go/NoGo, {reference} reference, locked to the sound"
    em.contrast_figure(result, rois, args.out / "erp.png", title)
    # A session with one block is fully described by erp.png; with more than one,
    # the blocks may not be the same task (see block_figure).
    written = ["results.json", "events.csv", "erp.png"]
    if len(blocks) > 1:
        em.block_figure(result, blocks, args.out / "erp_blocks.png",
                        f"{run.device.upper()} Go/NoGo by block \u2014 {primary} ROI "
                        f"({' '.join(picks[primary])}), {reference} reference")
        written.append("erp_blocks.png")

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
    print(f"  reference: {reference}")
    print(f"  events shifted {timing['shift_samples']} additional samples "
          f"({1000 * timing['hardware_marker_shift_s']:.1f} ms hardware marker shift"
          + (f" + {1000 * timing['audio_offset_s']:.0f} ms audio, inferred"
             if timing.get("audio_offset_s") else "")
          + f"); kept {timing['n_by_condition']}")
    if timing.get("n_per_channel"):
        worst = sorted(timing["channel_epochs_excluded"].items(), key=lambda kv: -kv[1])[:5]
        print(f"  per-channel trials {timing['n_per_channel_min']}\u2013"
              f"{timing['n_per_channel_max']} of {timing['n_epochs']}; all channels usable in "
              f"{timing['n_epochs_all_channels_usable']} (the cluster test's n)"
              + ("; most excluded " + ", ".join(f"{c} \u2212{k}" for c, k in worst) if worst else ""))
        share = timing["bad_channels_from_epoch_share"]
        print(f"  guards: worst epoch lost {timing['bad_channels_per_epoch_max']} of "
              f"{len(timing['n_per_channel'])} channels, "
              f"{timing['epochs_dropped_bad_channel_share']} epochs dropped over the "
              f"{timing['epoch_max_bad_share']:.0%} cut; "
              + (", ".join(f"{c} marked bad ({v:.0%} of epochs)" for c, v in share.items())
                 if share else f"no channel over the {timing['channel_max_bad_epoch_share']:.0%} "
                               "bad-epoch cut"))
    rt_text = ("no responses" if behaviour["rt_median_s"] is None else
               f"RT median {1000 * behaviour['rt_median_s']:.0f} ms from the marker")
    counting = behaviour["counting_blocks"]
    print(f"  hits {behaviour['hits']}/{behaviour['go_trials_responding']}, "
          f"false alarms {behaviour['false_alarms']}, {rt_text}"
          + (f"; counting blocks {counting} ({behaviour['go_trials_counting']} Go, no response)"
             if counting else ""))
    for name, res in rois.items():
        mark = "*" if name == primary else " "
        print(f" {mark}{name:>9} ({' '.join(res['roi'])}) Go−NoGo "
              f"{res['difference_uv']:.2f} µV "
              f"[{res['ci95_uv'][0]:.2f}, {res['ci95_uv'][1]:.2f}] over "
              f"{1000 * p3_window[0]:.0f}–{1000 * p3_window[1]:.0f} ms, "
              f"peak {1000 * res['peak_latency_s']:.0f} ms "
              f"(FWHM {1000 * res['peak_fwhm_s']:.0f} ms)")
    print(f"  {primary} baseline RMS {best_roi['baseline_rms_uv']:.2f} µV vs ± noise "
          f"{best_roi['plus_minus_rms_uv']:.2f} µV")
    same_sign = (f", all one sign: {topo['all_channels_same_sign']}"
                 if topo["same_sign_test_meaningful"] else
                 " (same-sign test is vacuous under an average reference)")
    print(f"  channel mean at the {primary} peak {topo['channel_mean_uv']:+.2f} µV"
          f"{same_sign}")
    for name, b in blocks.items():
        kind = {True: "pressed", False: "counting, no response", None: "?"}[b["responding"]]
        best = next((c for c in b["clusters"] if c["p"] < 0.25), None)
        note = (f", cluster {1000 * best['t_start']:.0f}–{1000 * best['t_end']:.0f} ms "
                f"{best['sign']} p={best['p']:.4f}" if best else ", no cluster p<0.25")
        print(f"  {name} ({kind}): {b['n_go']} Go, Go−NoGo {b['difference_uv']:+.2f} µV "
              f"[{b['ci95_uv'][0]:+.2f}, {b['ci95_uv'][1]:+.2f}], peak {b['peak_uv']:+.2f} µV "
              f"at {1000 * b['peak_latency_s']:.0f} ms "
              f"(FWHM {1000 * b['peak_fwhm_s']:.0f} ms){note}")
    if strict is not clusters:
        best = strict["clusters"][0]["p"] if strict["clusters"] else float("nan")
        print(f"  cluster test on {clusters['n_a']}/{clusters['n_b']} trials "
              f"(bad cells {clusters['bad_cells']}d); dropping those epochs instead leaves "
              f"{strict['n_a']}/{strict['n_b']} and a best p of {best:.4f}")
    for cluster in clusters["clusters"][:2]:
        print(f"  cluster {1000 * cluster['t_start']:.0f}–{1000 * cluster['t_end']:.0f} ms "
              f"{cluster['sign']} p={cluster['p']:.4f} ({len(cluster['channels'])} channels)")
    print(f"  NoGo N1 {1000 * n1['n1_latency_s']:.0f} ms, P2 {1000 * n1['p2_latency_s']:.0f} ms "
          f"(after the sound)")
    print(f"  wrote {args.out}/" + ", ".join(written))


if __name__ == "__main__":
    main()
