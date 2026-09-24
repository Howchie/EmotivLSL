#!/usr/bin/env python3
"""What the cluster test costs, and what restricting it to the tested sites buys.

``emotiv.cluster_test`` needs a rectangular trials x times x channels block, so
an electrode that was unusable in one epoch forces a choice with no good option:
drop that epoch for every channel (``bad_cells="complete"``) or fill the cell
with a spline (``"interpolate"``).  Neither is right when the electrode that
failed could not have contributed to the effect being tested.  Run it and see::

    python analysis/cluster_sets.py

Four tests per session, all on the same epochs and the same window:

* **all / device**  -- the current headline, using that device's ``bad_cells``.
* **all / complete** -- the strict version, dropping any epoch with a hole.
* **core / complete** -- the same strict rule applied only to the electrodes the
  hypothesis is about.  On the Flex that is the inner cap: the rim (Fp, F7/F8,
  FT9/10, T7/8, P7/8, PO9/10, O) is where this cap seats worst and is not where
  a P3 is read.  The EPOC X has no inner cap -- all 14 of its sensors are on the
  rim -- so its core set is all of them and this row equals the one above.
* **ROI / time only** -- ``emotiv.roi_cluster_test`` on one named site or a small
  ROI.  This never needs the rectangular block at all: ``roi_trials`` averages
  each trial over the ROI channels usable in that trial, so nothing is dropped
  for an outside electrode and nothing is interpolated.

The channel set and the ROI are hypotheses and must be fixed before looking at
the topography; they are listed in the output so a report can state them.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import emotiv as em
import gng_erp
import oddball_erp

OUT = Path("data/device_quality")

SESSIONS = [
    {"key": "gng_epocx", "xdf": "data/GnG/gng.xdf", "csv": "data/GnG/gng.csv", "task": "gng"},
    {"key": "gng_epocx_2", "xdf": "data/GnG/epoc_gng_timingfixed.xdf",
     "csv": "data/GnG/epoc_gng_timingfixed.csv", "task": "gng"},
    {"key": "gng_flex", "xdf": "data/GnG/gng_flex.xdf", "csv": "data/GnG/gng_flex.csv",
     "task": "gng"},
    {"key": "oddball_flex", "xdf": "data/Oddball/oddball_flex.xdf", "csv": None,
     "task": "oddball"},
]

# The Flex rim, excluded from the "core" set.  Chosen on where the cap seats --
# the outermost ring of electrodes, which is also where every session's bad
# channels have come from -- and not on where any cluster landed.
FLEX_RIM = ("Fp1", "Fp2", "F7", "F8", "FT9", "FT10", "T7", "T8",
            "P7", "P8", "PO9", "PO10", "O1", "Oz", "O2")

# Small, a-priori ROIs.  One row per hypothesis worth stating separately; a
# session that lacks the channels simply skips the row.
ROIS = {
    "Fz": ("Fz",), "Cz": ("Cz",), "Pz": ("Pz",),
    "F3+F4": ("F3", "F4"),
    "FCz-ish (FC1+FC2)": ("FC1", "FC2"),
    "central ROI": em.FLEX.central_roi,
    "EPOC X frontal ROI": em.EPOCX.frontal_roi,
}

WINDOW = (0.0, 1.0)
# The a-priori P3 test, run alongside the exploratory 0-1 s one.  Restricting the
# window and the tail is only legitimate because both were fixed in advance:
# gng_erp.P3_WINDOW is the summary window every session already uses, widened to
# cover the range the component has peaked in (250-340 ms), and the P3 is
# positive by definition.  The 0-1 s two-tailed test is reported next to it so
# nothing is hidden by the restriction.
P3_WINDOW = (0.15, 0.60)
REPORT_P = 0.15  # list every cluster at least this significant


def load_session(spec: dict):
    run = em.load(spec["xdf"], key=spec["key"])
    stem = Path(spec["xdf"]).stem
    if spec["task"] == "gng":
        events, _ = gng_erp.task_events(run, spec["csv"])
        task = gng_erp.task_mask(run, events)
        codes, conditions = gng_erp.CODES, gng_erp.CONDITIONS
        offset = gng_erp.AUDIO_OFFSET_S.get(stem, 0.0)
        bad_channels = None  # the device's own policy
    else:
        events, _ = oddball_erp.task_events(run)
        task = oddball_erp.task_mask(run, events)
        codes, conditions = oddball_erp.CODES, oddball_erp.CONDITIONS
        offset = 0.0
        bad_channels = "drop"  # oddball_erp overrides the profile; match it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = em.clean(run, task_mask=task,
                           manual_bads=gng_erp.MANUAL_BADS.get(stem, {}))
        epochs, timing = em.erp_epochs(
            run, cleaned, events, codes=codes, bad_channels=bad_channels,
            chain_latency_s=offset)
    return run, epochs, timing, conditions


def best(clusters: list[dict]) -> dict:
    """The strongest cluster, or a null row when the test found none.

    Reported next to the full list rather than instead of it: on a 0-1 s
    two-tailed ROI test the smallest p is often a narrow late deflection that is
    not the component anyone is asking about.
    """

    if not clusters:
        return {"p": float("nan"), "sign": "-", "t_start": float("nan"),
                "t_end": float("nan"), "n_channels": 0}
    c = clusters[0]
    return {"p": c["p"], "sign": c["sign"], "t_start": c["t_start"], "t_end": c["t_end"],
            "n_channels": len(c.get("channels", []))}


def run_session(spec: dict) -> list[dict]:
    run, epochs, timing, conditions = load_session(spec)
    test, ref = conditions
    a, b = epochs[test], epochs[ref]
    policy = run.dev.processing.bad_cells
    core = ([c for c in epochs.ch_names if c not in FLEX_RIM] if run.device == "flex"
            else list(epochs.ch_names))

    rows = []
    variants = [("all / device", None, policy), ("all / complete", None, "complete"),
                ("core / complete", core, "complete")]
    for label, picks, cells in variants:
        res = em.cluster_test(a, b, tmin=WINDOW[0], tmax=WINDOW[1],
                              bad_cells=cells, picks=picks)
        rows.append({"session": spec["key"], "device": run.device, "test": label,
                     "channels": len(res["picks"]), "n_test": res["n_a"], "n_ref": res["n_b"],
                     **best(res["clusters"]),
                     "picks": " ".join(res["picks"])})

    for label, roi in ROIS.items():
        picks = em.roi_picks(epochs, roi)
        if not picks:
            continue
        res = em.roi_cluster_test(a, b, picks, tmin=WINDOW[0], tmax=WINDOW[1])
        rows.append({"session": spec["key"], "device": run.device,
                     "test": f"ROI {label}", "channels": len(picks),
                     "n_test": res["n_a"], "n_ref": res["n_b"], **best(res["clusters"]),
                     "picks": " ".join(picks),
                     "all_clusters": [c for c in res["clusters"] if c["p"] <= REPORT_P]})
        # The a-priori version: P3 window, positive tail.  Both contrasts are
        # written test-minus-reference with the positive component as the test
        # condition (Go-NoGo, deviant-standard), so +1 is right for both tasks.
        pri = em.roi_cluster_test(a, b, picks, tmin=P3_WINDOW[0], tmax=P3_WINDOW[1],
                                  tail=1)
        rows.append({"session": spec["key"], "device": run.device,
                     "test": f"ROI {label} [P3 window, +tail]", "channels": len(picks),
                     "n_test": pri["n_a"], "n_ref": pri["n_b"], **best(pri["clusters"]),
                     "picks": " ".join(picks),
                     "all_clusters": [c for c in pri["clusters"] if c["p"] <= REPORT_P]})

    for r in rows:
        r["epochs_available"] = [int(len(a)), int(len(b))]
        r["all_channels_usable"] = timing["n_epochs_all_channels_usable"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = SESSIONS
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in SESSIONS if s["key"] in wanted]

    rows = []
    for spec in specs:
        print(f"testing {spec['key']} ...", flush=True)
        rows += run_session(spec)
    frame = pd.DataFrame(rows)
    frame.drop(columns=["all_clusters"], errors="ignore").to_csv(
        args.out / "cluster_sets.csv", index=False)
    (args.out / "cluster_sets.json").write_text(
        json.dumps(em.to_jsonable(rows), indent=2), encoding="utf-8")

    pd.set_option("display.width", 220, "display.max_rows", 200,
                  "display.float_format", lambda v: f"{v:.3f}")
    show = ["session", "device", "test", "channels", "n_test", "n_ref",
            "p", "sign", "t_start", "t_end", "n_channels"]
    print()
    print(frame[show].to_string(index=False))
    print(f"\nevery cluster with p <= {REPORT_P} (the table above shows only the smallest):")
    for row in rows:
        found = row.get("all_clusters")
        if not found:
            continue
        detail = "; ".join(f"{1000 * c['t_start']:.0f}-{1000 * c['t_end']:.0f} ms "
                           f"{c['sign'][:3]} p={c['p']:.3f} ({c['mean_diff_uv']:+.2f} uV)"
                           for c in found)
        print(f"  {row['session']:<13} {row['test']:<38} {detail}")
    print(f"\nwrote {args.out}/cluster_sets.csv, cluster_sets.json")


if __name__ == "__main__":
    main()
