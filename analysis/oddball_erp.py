#!/usr/bin/env python3
"""Auditory oddball ERPs (mismatch negativity) from an EPOC X or Flex 1.0 recording.

Everything device-shaped -- loading, the sample-grid rebuild, the reference, the
artifact thresholds, ICA, timing correction -- lives in the ``emotiv`` package and
comes from that headset's ``Processing`` profile in ``devices.py``.  The
measurement and the figures are the same two-condition machinery the Go/NoGo
analysis uses (``emotiv.contrast``, ``emotiv.figures``), so a number here means
what the same number means there.  What is specific to this task is only which
markers are standards and which are deviants::

    python analysis/oddball_erp.py --xdf data/Oddball/oddball_flex.xdf \
        --out data/Oddball/oddball_flex_erp

There is no PsychoPy CSV for this paradigm: the markers carry the stimulus and
nothing else is needed, so the run is read from the XDF alone.

Two things differ from a Go/NoGo contrast and both are deliberate.

**The standard average excludes post-deviant standards.**  A standard heard
immediately after a deviant is itself locally rare, and its response carries part
of the deviance effect.  Including it shrinks the MMN.  ``--keep-post-deviant``
turns the exclusion off to show what it costs.

**Bad channels are dropped, not interpolated.**  The MMN is read partly from its
polarity: frontocentral negative, reversing at the inferior ring.  Interpolating
the inferior-most electrodes from superior neighbours pulls the inferior ROI
towards the superior one, which is the very contrast being read.  The Flex
profile interpolates by default, and this script overrides it; ``--bad-channels
interpolate`` restores the profile behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import emotiv as em

XDF = "data/Oddball/oddball_flex.xdf"
OUT = Path("data/Oddball/oddball_flex_erp")

# Known-bad channels are a property of one recording, not of the headset.
MANUAL_BADS: dict[str, dict[str, str]] = {}

CODES = {"standard": 1, "deviant": 2}
CONDITIONS = ("deviant", "standard")  # the contrast is deviant minus standard

# ROIs are reported in this order; one the device leaves empty is skipped.  The
# MMN is frontocentral, so that ROI is primary, and ``inferior`` is kept in the
# report because the component should reverse sign there.
ROI_ORDER = ("frontal", "central", "inferior", "posterior")

# An oddball produces two deviance responses and they have opposite signs, so one
# window cannot measure both: averaging across them cancels them.  Measuring
# 100-250 ms on oddball_flex.xdf returned +1.08 uV and looked like a null, because
# the window ran from the -2.3 uV MMN trough at 172 ms into the rising edge of a
# +6 uV P3.  Each component therefore gets its own a-priori window and its own
# expected polarity, and both are reported.
#
# The MMN follows the N1 and overlaps the P2; 100-200 ms is the canonical window
# and it stops before the P3 rises (~190 ms in this recording).  The deviant P3
# is measured over 250-450 ms, matching the Go/NoGo P3 window offset for its later
# onset in a passive task.
COMPONENTS = {
    "mmn": {"window": (0.10, 0.20), "sign": -1, "label": "MMN"},
    "deviant_p3": {"window": (0.25, 0.45), "sign": +1, "label": "deviant P3"},
}
PRIMARY_COMPONENT = "mmn"

# Marker naming, from data/Oddball/oddball.py: the deviant is pushed as
# "Oddball-oddball" and reported here as "deviant", because "oddball" is the
# paradigm and is not a condition name anyone can read in a table.
STIM_OF = {"standard": "standard", "oddball": "deviant"}


# ---------------------------------------------------------------------------
# The task


def task_events(run: em.Run, keep_post_deviant: bool = False) -> tuple[pd.DataFrame, dict]:
    """Standard and deviant events, with block numbers and the post-deviant flag.

    Blocks come from the ``BlockStart-n`` / ``BlockEnd-n`` markers that bracket
    the tones; a tone outside every block would be a marker-stream fault and
    raises rather than being silently assigned to one.
    """

    def condition_of(value: str) -> str | None:
        prefix, _, stim = str(value).partition("-")
        return STIM_OF.get(stim) if prefix == "Oddball" else None

    events = em.events_from_markers(run, condition_of)

    blocks, block_markers = em.paired_blocks(run)
    if blocks.empty:
        raise RuntimeError("no complete BlockStart/BlockEnd interval found")
    block = np.full(len(events), -1)
    for i, row in enumerate(blocks.itertuples(index=False)):
        block[(events["time"].to_numpy() >= row.start) &
              (events["time"].to_numpy() <= row.end)] = i
    if (block < 0).any():
        raise RuntimeError(f"{int((block < 0).sum())} tones fall outside every block")

    is_deviant = (events["condition"] == "deviant").to_numpy()
    # A standard heard straight after a deviant is itself locally rare, so its
    # response carries part of the deviance effect; the convention is to leave it
    # out of the standard average.  The first tone of a block has no predecessor
    # within the block and is treated the same way.
    post_deviant = np.r_[True, is_deviant[:-1]] | np.r_[True, np.diff(block) != 0]
    events = events.assign(block=block, deviant=is_deviant, post_deviant=post_deviant)

    counts = events.groupby(["block", "condition"]).size().unstack(fill_value=0)
    soa_values = [np.diff(np.sort(events.loc[events["block"] == b, "time"].to_numpy()))
                  for b in sorted(set(block))]
    soa_values = (np.concatenate([v for v in soa_values if len(v)])
                  if any(len(v) for v in soa_values) else np.array([]))
    soa = ({"min": float(soa_values.min()), "median": float(np.median(soa_values)),
            "max": float(soa_values.max())}
           if len(soa_values) else {"min": None, "median": None, "max": None})
    behaviour = {
        "n_standard": int((~is_deviant).sum()), "n_deviant": int(is_deviant.sum()),
        "deviant_rate": float(is_deviant.mean()),
        "post_deviant_standards_excluded": bool(not keep_post_deviant),
        "n_post_deviant_standards": int((post_deviant & ~is_deviant).sum()),
        "blocks": {int(b): {k: int(v) for k, v in row.items()} for b, row in counts.iterrows()},
        "block_marker_diagnostics": block_markers,
        # Within blocks only: the gap across a block boundary is a rest break,
        # not an SOA, and it is minutes long.
        "soa_s": soa,
    }
    if not keep_post_deviant:
        events = events[is_deviant | ~post_deviant]
    return events.reset_index(drop=True), behaviour


def task_mask(run: em.Run, events: pd.DataFrame, pad_s: float = 2.0) -> np.ndarray:
    """Union of the marked tone blocks, excluding between-block rest from QC."""

    mask, _ = em.block_mask(run, pad_s=pad_s,
                            fallback_times=events["time"].to_numpy(float))
    return mask


# ---------------------------------------------------------------------------
# Measurement


def device_rois(run: em.Run, epochs) -> dict[str, list[str]]:
    """The device's ROIs this montage actually has, in ROI_ORDER.

    ``roi_picks`` drops channels that are missing and channels the profile
    interpolated, so a reported ROI is the electrodes it was measured from.
    """

    out = {}
    for name in ROI_ORDER:
        roi = em.roi_picks(epochs, getattr(run.dev, f"{name}_roi", ()))
        if roi:
            out[name] = roi
    if not out:
        raise RuntimeError(f"none of {ROI_ORDER} are defined for {run.device}")
    return out


def polarity_check(rois: dict, primary: str) -> dict:
    """Does the effect reverse sign between the primary ROI and the inferior ring?

    A frontocentral negativity that is generated in auditory cortex inverts below
    the Sylvian fissure in a reference near that plane.  It is a supporting
    observation, not a test: the Flex records against TP9, which is itself close
    to the inversion, so the inferior sites can sit near the null rather than
    clearly reversed, and the size of the reversal says more about the reference
    than about the generator.
    """

    if "inferior" not in rois or primary not in rois:
        return {"available": False}
    a, b = rois[primary]["difference_uv"], rois["inferior"]["difference_uv"]
    return {"available": True, "primary_uv": a, "inferior_uv": b,
            "reverses": bool(np.sign(a) != np.sign(b) and abs(b) > 0.3),
            "inferior_ci95_uv": rois["inferior"]["ci95_uv"]}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default=XDF)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--chain-latency", type=float, default=None,
                        help="replace the profile hardware marker shift (s)")
    parser.add_argument("--audio-offset", type=float, default=0.0, metavar="SECONDS",
                        help="additional delay from the shifted marker to the sound for this "
                             "session; 0 for normal recordings")
    parser.add_argument("--mmn-window", default=None, metavar="START,END",
                        help=f"MMN window in seconds after the sound (default "
                             f"{COMPONENTS['mmn']['window'][0]}-{COMPONENTS['mmn']['window'][1]})")
    parser.add_argument("--p3-window", default=None, metavar="START,END",
                        help=f"deviant-P3 window in seconds after the sound (default "
                             f"{COMPONENTS['deviant_p3']['window'][0]}-"
                             f"{COMPONENTS['deviant_p3']['window'][1]})")
    parser.add_argument("--reference", default=None, metavar="POLICY",
                        help='override the device reference: "recorded", "average", or a '
                             'comma-separated channel list to link (e.g. FT9,FT10)')
    parser.add_argument("--bad-channels", default="drop", choices=("drop", "interpolate"),
                        help="what to do with channels flagged bad (default drop; see the "
                             "module docstring for why this overrides the Flex profile)")
    parser.add_argument("--keep-post-deviant", action="store_true",
                        help="keep standards that followed a deviant in the standard average")
    parser.add_argument("--heog", default=None, metavar="LEFT,RIGHT",
                        help="left/right frontal pair for the saccade proxy, for a Flex "
                             "montage without the profile's default pair")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(em.SEED)
    stem = Path(args.xdf).stem
    components = {k: dict(v) for k, v in COMPONENTS.items()}
    if args.mmn_window:
        components["mmn"]["window"] = tuple(float(v) for v in args.mmn_window.split(","))
    if args.p3_window:
        components["deviant_p3"]["window"] = tuple(float(v) for v in args.p3_window.split(","))
    window = components[PRIMARY_COMPONENT]["window"]

    run = em.load(args.xdf)
    chain = args.chain_latency if args.chain_latency is not None else run.marker_shift_s
    if args.chain_latency is not None:
        run.shift_markers(chain - run.marker_shift_s)
    events, behaviour = task_events(run, keep_post_deviant=args.keep_post_deviant)
    reference_arg = args.reference
    if reference_arg and "," in reference_arg:
        reference_arg = tuple(reference_arg.split(","))
    cleaned = em.clean(run, task_mask=task_mask(run, events),
                       manual_bads=MANUAL_BADS.get(stem, {}), reference=reference_arg,
                       heog=tuple(args.heog.split(",")) if args.heog else None)
    epochs, timing = em.erp_epochs(run, cleaned, events, codes=CODES,
                                   chain_latency_s=args.audio_offset,
                                   bad_channels=args.bad_channels)
    timing["audio_offset_s"] = float(args.audio_offset)
    timing["hardware_marker_shift_s"] = float(run.marker_shift_s)
    timing["chain_latency_s"] = float(chain)
    timing["effective_marker_to_sound_s"] = float(run.marker_shift_s + args.audio_offset)

    picks = device_rois(run, epochs)
    primary = next(iter(picks))
    measured = {
        comp: {name: em.contrast(epochs, roi, rng, spec["window"], CONDITIONS, spec["sign"])
               for name, roi in picks.items()}
        for comp, spec in components.items()}
    rois = measured[PRIMARY_COMPONENT]
    reference = cleaned.parameters["reference"]
    topo = {comp: em.topography(epochs, r[primary]["_peak_i"], reference, CONDITIONS)
            for comp, r in measured.items()}
    clusters = em.cluster_test(epochs["deviant"], epochs["standard"], tmin=0.0, tmax=1.0,
                               bad_cells=run.dev.processing.bad_cells)
    # Interpolating a sporadic bad cell keeps the epoch, but the reader should be
    # able to see what the same test says when those epochs are simply dropped,
    # so the stricter answer is always computed and stored alongside it.
    strict = (em.cluster_test(epochs["deviant"], epochs["standard"], tmin=0.0, tmax=1.0, bad_cells="complete")
              if run.dev.processing.bad_cells != "complete" else clusters)
    # The standard's own N1/P2, as an independent check that this session's time
    # base matches the other recordings from the same headset.
    n1 = em.n1_p2(epochs["standard"], picks[primary],
                  picks.get("inferior", picks[primary]), rng)
    blocks = em.by_block(epochs, picks[primary], rng, window, CONDITIONS,
                         components[PRIMARY_COMPONENT]["sign"],
                         bad_cells=run.dev.processing.bad_cells)

    result = {
        "recording": args.xdf,
        "device": run.device,
        "primary_roi": primary,
        "primary_component": PRIMARY_COMPONENT,
        "window_s": list(window),
        "windows_s": [list(spec["window"]) for spec in components.values()],
        "integrity": em.integrity(run, cleaned.raw),
        "preprocessing": cleaned.summary(),
        "timing": timing,
        "behaviour": behaviour,
        "components": {comp: {"window_s": list(components[comp]["window"]),
                              "sign": components[comp]["sign"],
                              "rois": {k: em.to_jsonable(v) for k, v in r.items()},
                              "topography_at_primary_peak": topo[comp],
                              "polarity": polarity_check(r, primary)}
                       for comp, r in measured.items()},
        "rois": {k: em.to_jsonable(v) for k, v in rois.items()},
        "topography_at_primary_peak": topo[PRIMARY_COMPONENT],
        "polarity": polarity_check(rois, primary),
        "clusters": clusters["clusters"],
        "cluster_n": [clusters["n_a"], clusters["n_b"]],
        "cluster_bad_cells": clusters["bad_cells"],
        "clusters_complete_cases": strict["clusters"],
        "cluster_n_complete_cases": [strict["n_a"], strict["n_b"]],
        "standard_n1_p2": em.to_jsonable(n1),
        "blocks_primary": blocks,
    }
    (args.out / "results.json").write_text(json.dumps(em.to_jsonable(result), indent=2),
                                           encoding="utf-8")
    events.to_csv(args.out / "events.csv", index=False)
    title = f"{run.device.upper()} auditory oddball, {reference} reference, locked to the sound"
    em.contrast_figure(result, rois, args.out / "erp.png", title)
    written = ["results.json", "events.csv", "erp.png"]
    if len(blocks) > 1:
        em.block_figure(result, blocks, args.out / "erp_blocks.png",
                        f"{run.device.upper()} oddball by block — {primary} ROI "
                        f"({' '.join(picks[primary])}), {reference} reference")
        written.append("erp_blocks.png")

    print(f"{args.xdf} [{run.device}]: {run.duration_s:.0f} s, "
          f"{cleaned.summary()['task_minutes']:.1f} min of tones")
    rails = cleaned.rails
    if len(rails):
        print(f"  rail fraction: median {rails.median():.2f}%, worst "
              f"{rails.idxmax()} {rails.max():.2f}% (cut {run.dev.processing.rail_pct}%)")
    print(f"  {args.bad_channels}ped {cleaned.bads}; ICA removed {cleaned.ica_exclude} "
          f"({', '.join(f'|r| {k}={np.abs(v).max():.2f}' for k, v in cleaned.ica_corr.items())})"
          .replace("dropped", "dropped").replace("interpolateped", "interpolated"))
    print(f"  reference: {reference}")
    print(f"  {behaviour['n_deviant']} deviants / {behaviour['n_standard']} standards "
          f"({100 * behaviour['deviant_rate']:.0f}% deviant), SOA "
          f"{behaviour['soa_s']['min']:.2f}-{behaviour['soa_s']['max']:.2f} s"
          + (f"; {behaviour['n_post_deviant_standards']} post-deviant standards excluded"
             if not args.keep_post_deviant else ""))
    print(f"  events shifted {timing['shift_samples']} additional samples "
          f"({1000 * timing['hardware_marker_shift_s']:.1f} ms hardware marker shift"
          + (f" + {1000 * timing['audio_offset_s']:.0f} ms audio"
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
    for comp, spec in components.items():
        r = measured[comp]
        w = spec["window"]
        print(f"  [{spec['label']}] {1000 * w[0]:.0f}\u2013{1000 * w[1]:.0f} ms, "
              f"expected {'negative' if spec['sign'] < 0 else 'positive'}")
        for name, res in r.items():
            mark = "*" if name == primary else " "
            extreme = "trough" if spec["sign"] < 0 else "peak"
            print(f"   {mark}{name:>9} ({' '.join(res['roi'])}) deviant\u2212standard "
                  f"{res['difference_uv']:+.2f} \u00b5V "
                  f"[{res['ci95_uv'][0]:+.2f}, {res['ci95_uv'][1]:+.2f}], {extreme} "
                  f"{res['peak_uv']:+.2f} \u00b5V at {1000 * res['peak_latency_s']:.0f} ms "
                  f"(FWHM {1000 * res['peak_fwhm_s']:.0f} ms)")
        pol = polarity_check(r, primary)
        if pol["available"]:
            print(f"    polarity: {primary} {pol['primary_uv']:+.2f} \u00b5V vs inferior "
                  f"{pol['inferior_uv']:+.2f} \u00b5V "
                  f"[{pol['inferior_ci95_uv'][0]:+.2f}, {pol['inferior_ci95_uv'][1]:+.2f}] \u2014 "
                  + ("reverses" if pol["reverses"] else "no clear reversal"))
        t = topo[comp]
        print(f"    channel mean at the {primary} {'trough' if spec['sign'] < 0 else 'peak'} "
              f"{t['channel_mean_uv']:+.2f} \u00b5V"
              + (f", all one sign: {t['all_channels_same_sign']}"
                 if t["same_sign_test_meaningful"] else ""))
    best = rois[primary]
    print(f"  {primary} baseline RMS {best['baseline_rms_uv']:.2f} \u00b5V vs \u00b1 noise "
          f"{best['plus_minus_rms_uv']:.2f} \u00b5V")
    for name, b in blocks.items():
        cl = next((c for c in b["clusters"] if c["p"] < 0.25), None)
        note = (f", cluster {1000 * cl['t_start']:.0f}–{1000 * cl['t_end']:.0f} ms "
                f"{cl['sign']} p={cl['p']:.4f}" if cl else ", no cluster p<0.25")
        print(f"  {name}: {b['n_test']} deviants, deviant−standard "
              f"{b['difference_uv']:+.2f} µV [{b['ci95_uv'][0]:+.2f}, {b['ci95_uv'][1]:+.2f}], "
              f"trough {b['peak_uv']:+.2f} µV at {1000 * b['peak_latency_s']:.0f} ms{note}")
    if strict is not clusters:
        best = strict["clusters"][0]["p"] if strict["clusters"] else float("nan")
        print(f"  cluster test on {clusters['n_a']}/{clusters['n_b']} trials "
              f"(bad cells {clusters['bad_cells']}d); dropping those epochs instead leaves "
              f"{strict['n_a']}/{strict['n_b']} and a best p of {best:.4f}")
    for cluster in clusters["clusters"][:3]:
        print(f"  cluster {1000 * cluster['t_start']:.0f}–{1000 * cluster['t_end']:.0f} ms "
              f"{cluster['sign']} p={cluster['p']:.4f} ({len(cluster['channels'])} channels)")
    print(f"  standard N1 {1000 * n1['n1_latency_s']:.0f} ms, "
          f"P2 {1000 * n1['p2_latency_s']:.0f} ms (after the sound)")
    print(f"  wrote {args.out}/" + ", ".join(written))


if __name__ == "__main__":
    main()
