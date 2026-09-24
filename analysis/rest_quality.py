#!/usr/bin/env python3
"""Contact quality during the eyes-closed / eyes-open rest block, and its alpha.

The rest block is the one part of a session where both headsets were doing the
same thing under the same instruction -- sitting still, eyes shut or open -- so
it is the cleanest condition available for comparing them, cleaner than task time
(different tasks) and far cleaner than the gaps between blocks (uncontrolled)::

    python analysis/rest_quality.py

**Read the coverage before reading the numbers.** The rest block is four
closed/open pairs and only some were actually held:

* ``gng_flex`` and ``oddball_flex``: both pairs held, 39-71 s each.
* ``gng``: one pair held -- 32 s closed, 21 s open -- after three false starts,
  which is why the marker stream has 52 eye markers rather than 16.
* ``epoc_gng_timingfixed``: one 47 s eyes-closed and no open period. Usable for a
  noise and contact measurement, **not** for a closed-minus-open contrast.

So the cross-device comparison is one EPOC X pair and one extra EPOC X closed
period against four Flex pairs.  The EPOC X side is thin and its one pair is the
shortest of the set; that is a limit of the recordings, not of the analysis, and
the fix is to hold the rest block properly on the next EPOC X session.

Quality is measured exactly as ``device_quality.py`` measures it -- median
reference across channels, one common 120 uV threshold for both headsets -- so
the rows here can be put beside the task-time rows directly.
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
import device_quality as dq  # noqa: E402
import gng_erp  # noqa: E402
import oddball_erp  # noqa: E402

OUT = Path("data/device_quality")

# Seconds trimmed from each end of a self-timed interval: the participant is
# still settling at the start and anticipating the keypress at the end.  Same
# values qc.alpha_spectra uses, so the two agree about which samples are rest.
TRIM = em.qc.EYE_TRIM


def rest_channel_table(run: em.Run, cleaned: em.Cleaned, intervals: list[dict]) -> pd.DataFrame:
    """Per channel, per rest interval: the device_quality metrics on those seconds."""

    x, labels = dq.median_reference(cleaned)
    sfreq = float(run.sfreq_hz)
    p2p, starts = em.sliding_p2p(x, sfreq=sfreq)
    t_win = run.t[np.clip(starts, 0, len(run.t) - 1)]
    rails = cleaned.rails

    rows = []
    for iv in intervals:
        a, b = iv["start"] + TRIM[0], iv["end"] - TRIM[1]
        sel = np.flatnonzero((t_win >= a) & (t_win < b))
        if len(sel) < 20:
            continue
        block = p2p[:, sel]
        seg = x[:, int(a * sfreq):int(b * sfreq)]
        hf = em.preprocess.sosfiltfilt(
            em.preprocess.butter(4, [20, 40], btype="band", fs=sfreq, output="sos"),
            seg, axis=1)
        for j, ch in enumerate(labels):
            rows.append({
                "session": run.key, "device": run.device, "channel": ch,
                "region": dq.REGION_OF.get(ch, "other"),
                "state": iv["state"], "pair": iv["pair"], "duration_s": b - a,
                "hf_20_40_uv_rms": float(np.sqrt(np.mean(hf[j] ** 2))),
                "p2p_median_uv": float(np.median(block[j])),
                "p2p_p99_over_median": float(np.percentile(block[j], 99) / np.median(block[j])),
                "pct_windows_over_common": float(100 * (block[j] > dq.COMMON_P2P_UV).mean()),
                "rail_pct": float(rails[ch]) if len(rails) else float("nan"),
                "marked_bad": ch in cleaned.bads,
            })
    return pd.DataFrame(rows)


def alpha_table(run: em.Run, cleaned: em.Cleaned) -> pd.DataFrame:
    """Alpha power per channel per interval, and the closed-minus-open difference.

    ``qc.alpha_spectra`` applies the headset's own reference policy and rejects
    artifact windows, so this is the same measurement the alpha analysis would
    make; it simply had no intervals to run on until ``find_eye_intervals`` was
    fixed.
    """

    raw = em.make_raw(run)
    frame, spectra = em.alpha_spectra(run, raw, bads=cleaned.bads)
    if frame.empty:
        return pd.DataFrame()
    freqs = spectra["freqs"]
    rows = []
    for (state, pair), psd in spectra["spectra"].items():
        scores = em.alpha_scores(freqs, psd)
        for j, ch in enumerate(run.labels):
            if ch in cleaned.bads:
                continue
            rows.append({"session": run.key, "device": run.device, "channel": ch,
                         "region": dq.REGION_OF.get(ch, "other"), "state": state,
                         "pair": pair,
                         "alpha_db": float(scores["alpha_db"][j]),
                         "alpha_rel_db": float(scores["alpha_rel_db"][j]),
                         "hf_db": float(scores["hf_db"][j])})
    wide = pd.DataFrame(rows)
    if wide.empty:
        return wide
    piv = wide.pivot_table(index=["session", "device", "channel", "region", "pair"],
                           columns="state", values=["alpha_db", "alpha_rel_db"])
    if "closed" not in piv["alpha_db"] or "open" not in piv["alpha_db"]:
        return wide.assign(closed_minus_open_db=np.nan)
    out = piv.reset_index()
    out.columns = ["_".join(c for c in col if c).strip("_") for col in out.columns]
    out["closed_minus_open_db"] = out["alpha_db_closed"] - out["alpha_db_open"]
    out["rel_closed_minus_open_db"] = out["alpha_rel_db_closed"] - out["alpha_rel_db_open"]
    return out


def measure(spec: dict) -> dict | None:
    run = em.load(spec["xdf"], key=spec["key"])
    # Keep both halves where they exist, and a lone long interval where they do
    # not; the closed-only EPOC X session is still worth a noise measurement.
    intervals = em.find_eye_intervals(run, require_pair=False)
    if not intervals:
        print(f"  {spec['key']}: no usable rest interval, skipped")
        return None
    stem = Path(spec["xdf"]).stem
    if spec["task"] == "gng":
        events, _ = gng_erp.task_events(run, spec["csv"])
        task = gng_erp.task_mask(run, events)
    else:
        events, _ = oddball_erp.task_events(run)
        task = oddball_erp.task_mask(run, events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Cleaned exactly as the task analysis cleans it, so the bad-channel list
        # and the ICA are the ones the ERP used; only the measurement window moves.
        cleaned = em.clean(run, task_mask=task,
                           manual_bads=gng_erp.MANUAL_BADS.get(stem, {}))
        rest = rest_channel_table(run, cleaned, intervals)
        alpha = alpha_table(run, cleaned)
    return {"spec": spec, "run": run, "cleaned": cleaned, "intervals": intervals,
            "rest": rest, "alpha": alpha}


def figure(rest: pd.DataFrame, path: Path) -> None:
    states = ["closed", "open"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    xpos = {r: i for i, r in enumerate(dq.REGION_ORDER)}
    marker = {"epocx": "o", "flex": "s"}
    colour = {}
    for ax, (col, label) in zip(axes, [("hf_20_40_uv_rms", "20–40 Hz noise (µV rms)"),
                                       ("pct_windows_over_common",
                                        f"% of rest seconds over {dq.COMMON_P2P_UV:.0f} µV p2p")]):
        for (session, state), sub in rest.groupby(["session", "state"]):
            if state not in states:
                continue
            colour.setdefault(session, f"C{len(colour)}")
            agg = sub.groupby("region", observed=True)[col].median()
            agg = agg.reindex([r for r in dq.REGION_ORDER if r in agg.index])
            ax.plot([xpos[r] for r in agg.index], agg.values,
                    marker=marker[sub["device"].iloc[0]],
                    ls="-" if state == "closed" else "--",
                    color=colour[session], lw=1.5, ms=5,
                    label=f"{session} {state}")
        ax.set_xticks(range(len(dq.REGION_ORDER)))
        ax.set_xticklabels(dq.REGION_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Signal quality during the eyes-closed / eyes-open rest block", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in dq.SESSIONS:
        print(f"measuring {spec['key']} ...", flush=True)
        res = measure(spec)
        if res:
            results.append(res)
    if not results:
        print("no session has a usable rest block")
        return

    rest = pd.concat([r["rest"] for r in results], ignore_index=True)
    alpha = pd.concat([r["alpha"] for r in results if len(r["alpha"])], ignore_index=True)
    rest["region"] = pd.Categorical(rest["region"], dq.REGION_ORDER + ["other"], ordered=True)
    rest.to_csv(args.out / "rest_channels.csv", index=False)
    if len(alpha):
        alpha.to_csv(args.out / "rest_alpha.csv", index=False)
    figure(rest, args.out / "rest_quality.png")

    pd.set_option("display.width", 220, "display.max_rows", 300,
                  "display.float_format", lambda v: f"{v:.2f}")
    print("\n=== rest-block coverage ===")
    for r in results:
        iv = ", ".join(f"{v['state']}{v['pair']} {v['end'] - v['start']:.0f}s" for v in r["intervals"])
        print(f"  {r['run'].key:<14} {r['run'].device:<6} {iv}")

    print("\n=== per region, per state (medians over channels and intervals) ===")
    agg = (rest.groupby(["session", "device", "state", "region"], observed=True)
           .agg(n_ch=("channel", "size"), hf_uv=("hf_20_40_uv_rms", "median"),
                p2p_uv=("p2p_median_uv", "median"),
                ratio=("p2p_p99_over_median", "median"),
                pct_over=("pct_windows_over_common", "median")).reset_index())
    print(agg.to_string(index=False))

    print("\n=== whole cap, per state ===")
    whole = (rest.groupby(["session", "device", "state"])
             .agg(hf_med=("hf_20_40_uv_rms", "median"), hf_max=("hf_20_40_uv_rms", "max"),
                  ratio_med=("p2p_p99_over_median", "median"),
                  pct_over_med=("pct_windows_over_common", "median"),
                  pct_over_max=("pct_windows_over_common", "max")).reset_index())
    print(whole.to_string(index=False))

    if len(alpha) and "closed_minus_open_db" in alpha:
        print("\n=== alpha, closed minus open (dB), by region ===")
        a = alpha.copy()
        a["region"] = pd.Categorical(a["region"], dq.REGION_ORDER + ["other"], ordered=True)
        print(a.groupby(["session", "region"], observed=True)
              .agg(n=("channel", "size"),
                   alpha_db=("closed_minus_open_db", "median"),
                   rel_db=("rel_closed_minus_open_db", "median")).reset_index().to_string(index=False))
        post = a[a["region"] == "occipital"]
        if len(post):
            print(f"\n  occipital closed-minus-open: "
                  f"{post['closed_minus_open_db'].median():+.2f} dB "
                  f"(n={len(post)} channel-pairs, "
                  f"{100 * (post['closed_minus_open_db'] > 0).mean():.0f}% positive)")

    (args.out / "rest_quality.json").write_text(json.dumps(em.to_jsonable({
        "coverage": [{"session": r["run"].key, "device": r["run"].device,
                      "intervals": r["intervals"]} for r in results],
        "per_region": agg.to_dict("records"), "whole_cap": whole.to_dict("records"),
    }), indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}/rest_channels.csv, rest_alpha.csv, rest_quality.json, "
          "rest_quality.png")


if __name__ == "__main__":
    main()
