#!/usr/bin/env python3
"""What the two headsets get for the same Go/NoGo effect, measured the same way.

``device_quality.py`` says how much noise each headset carries per electrode.
This says what that costs where it matters: the same ROI, the same task, the
same contrast, measured with ``emotiv.contrast`` on both caps::

    python analysis/device_roi_snr.py

Three things are held fixed so the comparison is about the hardware:

* **The ROI.**  The EPOC X's frontal ROI (F3 F4 FC5 FC6) exists on both caps, so
  it is measured on both.  Each device's own primary ROI is reported underneath
  it, which is where the Flex gets to use the midline sites the EPOC X does not
  have -- that is the density's case, and it should be read against, not
  instead of, the shared row.
* **The trials.**  Only blocks in which the participant actually responded.
  ``epoc_gng_timingfixed.xdf`` ran its first block as silent counting, and a
  block with no motor activity does not produce the same Go-NoGo difference, so
  pooling it dilutes that session against the two that pressed throughout.
* **The time base.**  ``gng.xdf`` played through the pre-WASAPI audio path and
  needs ``gng_erp.AUDIO_OFFSET_S``; the other two need nothing.  Both come from
  the task script rather than being restated here.

The number to compare is ``snr``: the window-mean Go-NoGo difference over the
+/- average RMS, which is the noise that remains at that trial count.  It is a
ratio, so it is unaffected by the reference each headset is analysed in.
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

OUT = Path("data/device_quality")

SESSIONS = [
    {"key": "gng_epocx", "xdf": "data/GnG/gng.xdf", "csv": "data/GnG/gng.csv"},
    {"key": "gng_epocx_2", "xdf": "data/GnG/epoc_gng_timingfixed.xdf",
     "csv": "data/GnG/epoc_gng_timingfixed.csv"},
    {"key": "gng_flex", "xdf": "data/GnG/gng_flex.xdf", "csv": "data/GnG/gng_flex.csv"},
]

# The ROI both caps have, named for the device that only has this one.
SHARED_ROI = em.EPOCX.frontal_roi


def measure(spec: dict, rng) -> list[dict]:
    run = em.load(spec["xdf"], key=spec["key"])
    events, _ = gng_erp.task_events(run, spec["csv"])
    stem = Path(spec["xdf"]).stem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = em.clean(run, task_mask=gng_erp.task_mask(run, events),
                           manual_bads=gng_erp.MANUAL_BADS.get(stem, {}))
        epochs, timing = em.erp_epochs(
            run, cleaned, events, codes=gng_erp.CODES,
            chain_latency_s=gng_erp.AUDIO_OFFSET_S.get(stem, 0.0))

    # Responding blocks only; a silent-counting block has no motor activity.
    if "responding" in epochs.metadata:
        epochs = epochs[epochs.metadata["responding"].to_numpy()]

    rois = {"shared frontal": SHARED_ROI}
    primary = next((n for n in gng_erp.ROI_ORDER
                    if em.roi_picks(epochs, getattr(run.dev, f"{n}_roi", ()))), None)
    if primary:
        rois[f"own primary ({primary})"] = getattr(run.dev, f"{primary}_roi")

    rows = []
    for name, roi in rois.items():
        picks = em.roi_picks(epochs, roi)
        if not picks:
            continue
        res = em.contrast(epochs, picks, rng, gng_erp.P3_WINDOW, gng_erp.CONDITIONS)
        rows.append({
            "session": spec["key"], "device": run.device, "roi": name,
            "channels": " ".join(picks),
            "n_go": res["n_test"], "n_nogo": res["n_ref"],
            "difference_uv": res["difference_uv"],
            "ci_lo": res["ci95_uv"][0], "ci_hi": res["ci95_uv"][1],
            "peak_uv": res["peak_uv"], "peak_ms": 1000 * res["peak_latency_s"],
            "fwhm_ms": 1000 * res["peak_fwhm_s"],
            "baseline_rms_uv": res["baseline_rms_uv"],
            "pm_noise_uv": res["plus_minus_rms_uv"],
            "snr": res["difference_uv"] / res["plus_minus_rms_uv"],
            "bad_channels": " ".join(timing["bad_channels"]) or "-",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(em.SEED)

    rows = []
    for spec in SESSIONS:
        print(f"measuring {spec['key']} ...", flush=True)
        rows += measure(spec, rng)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "roi_snr.csv", index=False)
    (args.out / "roi_snr.json").write_text(
        json.dumps(em.to_jsonable(rows), indent=2), encoding="utf-8")

    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:.2f}")
    print(f"\nGo-NoGo over {1000 * gng_erp.P3_WINDOW[0]:.0f}-"
          f"{1000 * gng_erp.P3_WINDOW[1]:.0f} ms, responding blocks only\n")
    print(frame.to_string(index=False))
    print(f"\nwrote {args.out}/roi_snr.csv, roi_snr.json")


if __name__ == "__main__":
    main()
