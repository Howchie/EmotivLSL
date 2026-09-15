#!/usr/bin/env python3
"""Compare EPOC X alpha/theta power across eyes and task blocks.

The existing EPOC X sanity analysis estimates posterior alpha separately for
the four eyes-closed/open pairs.  This companion analysis puts those intervals
on the same 2-s Welch scale as the passive oddball and active Go/NoGo blocks.
It is descriptive: there is only one passive block and one active block, so
window-level confidence intervals should not be read as independent-subject
inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

import epocx_sanity as ex
import flex_sanity as fx

mne.set_log_level("ERROR")

XDF = "data/Oddball/data/epochx_0740.xdf"
CSV = "data/Oddball/data/epochx_0740.csv"
FS = fx.FS
WINDOW_S = 2.0
STEP_S = 1.0
ALPHA = (8.0, 13.0)
THETA = (4.0, 7.0)
FLANK = ((5.0, 7.0), (14.0, 17.0))
POSTERIOR = ex.POSTERIOR_ROI
METRICS = ("alpha_db", "theta_db", "alpha_theta_db", "alpha_flank_db")


def interval_definitions(run) -> list[dict]:
    """Return comparable time intervals for eyes and the two task blocks."""

    intervals = []
    for iv in run.eye_intervals:
        intervals.append({
            "state": f"eyes_{iv['state']}",
            "unit": f"eyes_{iv['state']}_pair{iv['pair']}",
            "unit_type": "eyes_pair",
            "pair": int(iv["pair"]),
            "start_s": float(iv["start"] + fx.EYE_TRIM[0]),
            "end_s": float(iv["end"] - fx.EYE_TRIM[1]),
        })

    for block, label in [("passive", "passive"), ("gng", "active")]:
        ev = run.tone_events[run.tone_events["block"] == block]
        # There are no explicit task-start markers in this XDF.  Include one
        # second around the first/last tone, while keeping blocks separate.
        intervals.append({
            "state": label,
            "unit": label,
            "unit_type": "task_block",
            "pair": "",
            "start_s": float(ev["time"].min() - 1.0),
            "end_s": float(ev["time"].max() + 1.0),
        })
    return intervals


def band_metrics(freqs: np.ndarray, psd: np.ndarray, roi_idx: list[int]) -> dict[str, float]:
    """ROI-mean band power in dB and alpha/theta ratios in dB."""

    roi_psd = psd[roi_idx]

    def band_power(bounds):
        sel = (freqs >= bounds[0]) & (freqs <= bounds[1])
        return float(roi_psd[:, sel].mean())

    alpha = band_power(ALPHA)
    theta = band_power(THETA)
    flank = np.mean([band_power(b) for b in FLANK])
    return {
        "alpha_db": 10.0 * np.log10(alpha),
        "theta_db": 10.0 * np.log10(theta),
        "alpha_theta_db": 10.0 * np.log10(alpha / theta),
        "alpha_flank_db": 10.0 * np.log10(alpha / flank),
    }


def analyze_intervals(run, raw, bads: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Compute 2-s Welch metrics and per-unit average spectra."""

    # Keep filtering continuous, as required for EEG spectral estimates.
    filtered = raw.copy().filter(1.0, 40.0)
    filtered.info["bads"] = list(bads)
    filtered.set_eeg_reference("average")
    x = filtered.get_data() * 1e6
    ok = fx.good_mask(filtered)
    good_idx = [j for j, ch in enumerate(run.labels) if ch not in bads]
    roi_idx = [run.labels.index(ch) for ch in POSTERIOR if ch not in bads]
    n = int(round(WINDOW_S * FS))
    step = int(round(STEP_S * FS))
    rows, unit_rows, unit_psds = [], [], {}
    freqs = None

    for iv in interval_definitions(run):
        a = max(0, int(round(iv["start_s"] * FS)))
        b = min(len(run.t), int(round(iv["end_s"] * FS)))
        values, psds = [], []
        rejected_fill = rejected_amp = 0
        for s in range(a, b - n + 1, step):
            seg = x[:, s:s + n]
            if not ok[s:s + n].all():
                rejected_fill += 1
                continue
            if np.ptp(seg[good_idx], axis=1).max() > 200.0:
                rejected_amp += 1
                continue
            freqs, p = welch(seg, fs=FS, nperseg=n, axis=1)
            psds.append(p)
            values.append(band_metrics(freqs, p, roi_idx))
            rows.append({
                **{k: iv[k] for k in ("state", "unit", "unit_type", "pair")},
                "window_start_s": float(run.t[s]),
                "window_end_s": float(run.t[s + n - 1]),
                **values[-1],
            })
        if not psds:
            raise RuntimeError(f"No usable spectral windows for {iv['unit']}")
        mean_psd = np.mean(psds, axis=0)
        unit_psds[iv["unit"]] = {
            "state": iv["state"],
            "pair": iv["pair"],
            "psd": mean_psd,
            "windows": len(psds),
            "rejected_fill": rejected_fill,
            "rejected_amplitude": rejected_amp,
        }
        unit_rows.append({
            **{k: iv[k] for k in ("state", "unit", "unit_type", "pair")},
            "windows": len(psds),
            "rejected_fill": rejected_fill,
            "rejected_amplitude": rejected_amp,
            **band_metrics(freqs, mean_psd, roi_idx),
        })

    frame = pd.DataFrame(rows)
    unit_frame = pd.DataFrame(unit_rows)
    return frame, unit_frame, {"freqs": freqs, "unit_psds": unit_psds, "roi": [run.labels[i] for i in roi_idx]}, {
        "filtered_band_hz": [1.0, 40.0],
        "welch_window_s": WINDOW_S,
        "welch_step_s": STEP_S,
        "artifact_peak_to_peak_uv": 200.0,
        "bad_channels": bads,
        "posterior_roi": [run.labels[i] for i in roi_idx],
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def summarize(frame: pd.DataFrame, unit: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Summaries at window and interval/block-unit levels."""

    states = {}
    for state, group in frame.groupby("state", sort=False):
        units = unit[unit["state"] == state]
        states[state] = {
            "n_windows": int(len(group)),
            "n_units": int(len(units)),
            "window_metrics": {},
            "unit_metrics": {},
        }
        for metric in METRICS:
            vals = group[metric].to_numpy(float)
            unit_vals = units[metric].to_numpy(float)
            states[state]["window_metrics"][metric] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "sd": float(vals.std(ddof=1)),
                "bootstrap_ci95_mean": bootstrap_mean(vals, rng),
            }
            states[state]["unit_metrics"][metric] = {
                "mean": float(unit_vals.mean()),
                "values": unit_vals.tolist(),
            }

    paired = {}
    closed = unit[unit["state"] == "eyes_closed"].set_index("pair")
    opened = unit[unit["state"] == "eyes_open"].set_index("pair")
    for metric in METRICS:
        d = (closed[metric] - opened[metric]).dropna().to_numpy(float)
        t = stats.ttest_1samp(d, 0.0)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        paired[metric] = {
            "closed_minus_open_by_pair": d.tolist(),
            "mean_difference": float(d.mean()),
            "t": float(t.statistic),
            "df": int(len(d) - 1),
            "p_t_test": float(t.pvalue),
            "ci95_mean_difference": [float(ci[0]), float(ci[1])],
            "sign_test_p": float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue),
        }

    # The task blocks have one unit each, so these are descriptive contrasts,
    # not valid independent-sample hypothesis tests.
    task_contrasts = {}
    eye_open_mean = unit[unit["state"] == "eyes_open"][list(METRICS)].mean()
    eye_closed_mean = unit[unit["state"] == "eyes_closed"][list(METRICS)].mean()
    for state in ("passive", "active"):
        row = unit[unit["state"] == state].iloc[0]
        task_contrasts[state] = {
            "minus_eyes_open_unit_mean": {m: float(row[m] - eye_open_mean[m]) for m in METRICS},
            "minus_eyes_closed_unit_mean": {m: float(row[m] - eye_closed_mean[m]) for m in METRICS},
        }

    return {"states": states, "paired_eyes": paired, "task_contrasts": task_contrasts}


def make_figure(frame: pd.DataFrame, unit: pd.DataFrame, spectra: dict, out: Path) -> None:
    freqs = spectra["freqs"]
    state_order = ["eyes_closed", "eyes_open", "passive", "active"]
    colors = {"eyes_closed": "#2455a4", "eyes_open": "#e28e2c", "passive": "#555555", "active": "#8b3a8b"}
    labels = {"eyes_closed": "Eyes closed", "eyes_open": "Eyes open", "passive": "Passive", "active": "Active Go/NoGo"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    for state in state_order:
        psds = [v["psd"] for v in spectra["unit_psds"].values() if v["state"] == state]
        # Equal weight per eye interval/task block, rather than letting longer
        # states dominate the plotted spectrum.
        # The unit PSD is channels x frequency; recover posterior channels from
        # the labels stored in the analysis metadata via the same ROI indices.
        # (All units share the same channel order.)
        posterior = [i for i, ch in enumerate(ex.LABELS) if ch in spectra["roi"]]
        p = np.mean(np.mean(psds, axis=0)[posterior], axis=0)
        ax.plot(freqs, 10 * np.log10(p), color=colors[state], lw=2, label=labels[state])
    ax.axvspan(*THETA, color="0.7", alpha=0.2, label="theta 4–7 Hz")
    ax.axvspan(*ALPHA, color="tab:blue", alpha=0.1, label="alpha 8–13 Hz")
    ax.set(xlim=(2, 30), xlabel="Hz", ylabel="posterior PSD (dB µV²/Hz)", title="Equal-weight mean posterior spectrum")
    ax.legend(fontsize=8, ncol=2)

    metrics_plot = [("alpha_db", "Absolute alpha (dB)"), ("alpha_theta_db", "Alpha / theta (dB)"), ("alpha_flank_db", "Alpha / flanks (dB)")]
    for ax, (metric, title) in zip(axes.flat[1:], metrics_plot):
        positions = np.arange(len(state_order))
        for j, state in enumerate(state_order):
            vals = frame.loc[frame["state"] == state, metric].to_numpy(float)
            # Window points show within-block variation; the diamond is the
            # equal-weight unit mean used for the spectrum and table.
            ax.scatter(np.full(len(vals), j), vals, s=8, alpha=0.12, color=colors[state], rasterized=True)
            u = unit.loc[unit["state"] == state, metric].to_numpy(float)
            ax.scatter([j], [u.mean()], s=65, color=colors[state], edgecolor="black", zorder=3, marker="D")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(positions, [labels[s] for s in state_order], rotation=20, ha="right")
        ax.set_ylabel("dB")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EPOC X posterior band power across eyes and task blocks", fontsize=13)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    out = Path("data/Oddball/data/epocx_bandpower_blocks")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    run, _ = ex.load_epocx(XDF, CSV)
    raw = fx.make_raw(run)
    noise = fx.channel_noise(raw)
    bads = noise.loc[noise["bad"], "channel"].tolist()
    frame, unit, spectra, preprocessing = analyze_intervals(run, raw, bads)
    summary = summarize(frame, unit, rng)
    result = {
        "recording": XDF,
        "states": {
            "eyes_closed": "four closed-eye intervals, trimmed as in the existing sanity analysis",
            "eyes_open": "four open-eye intervals, trimmed as in the existing sanity analysis",
            "passive": "from one second before the first passive tone through one second after the last",
            "active": "from one second before the first Go/NoGo tone through one second after the last",
        },
        "preprocessing": preprocessing,
        "summary": summary,
        "unit_means": unit,
    }
    (out / "results.json").write_text(json.dumps(fx.to_jsonable(result), indent=2), encoding="utf-8")
    frame.to_csv(out / "window_metrics.csv", index=False)
    unit.to_csv(out / "unit_means.csv", index=False)
    make_figure(frame, unit, spectra, out / "bandpower_blocks.png")

    s = summary["states"]
    paired = summary["paired_eyes"]
    lines = [
        "# EPOC X alpha/theta power across eyes and task blocks",
        "",
        "Power was estimated from continuous, average-referenced EEG after 1–40 Hz filtering using 2-s Welch windows (1-s step). The posterior ROI was O1/O2/P7/P8. Windows exceeding 200 µV peak-to-peak or overlapping a filled sample were rejected.",
        "",
        "The eyes conditions have four repeated intervals each. Passive and active each have only one block, so their window distributions are descriptive; window-level bootstrap intervals do not represent independent-subject uncertainty.",
        "",
        "## Posterior ROI results",
        "",
        "| State | windows | units | alpha (dB) | alpha/theta (dB) | alpha/flanks (dB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state in ["eyes_closed", "eyes_open", "passive", "active"]:
        row = s[state]["unit_metrics"]
        lines.append(f"| {state.replace('_', ' ')} | {s[state]['n_windows']} | {s[state]['n_units']} | {row['alpha_db']['mean']:.2f} | {row['alpha_theta_db']['mean']:.2f} | {row['alpha_flank_db']['mean']:.2f} |")
    lines += [
        "",
        "## Closed versus open",
        "",
        f"Across the four matched eye pairs, closed-minus-open alpha was **{paired['alpha_db']['mean_difference']:.2f} dB** (t(3) = {paired['alpha_db']['t']:.2f}, p = {paired['alpha_db']['p_t_test']:.3f}; exact sign-test p = {paired['alpha_db']['sign_test_p']:.3f}). Closed-minus-open alpha/theta was **{paired['alpha_theta_db']['mean_difference']:.2f} dB** (t(3) = {paired['alpha_theta_db']['t']:.2f}, p = {paired['alpha_theta_db']['p_t_test']:.3f}; sign-test p = {paired['alpha_theta_db']['sign_test_p']:.3f}).",
        "",
        f"Interpretationally, the ratio effects are driven more by low-frequency power than by a large alpha increase: absolute alpha changes little between eyes closed and open (**{paired['alpha_db']['mean_difference']:.2f} dB** closed-minus-open), whereas alpha/theta is **{paired['alpha_theta_db']['mean_difference']:.2f} dB** higher with eyes closed. The extra theta in eyes-open and active periods may include eye movements, muscle, and task-related movement. The task blocks also occur later in the recording than all eyes conditions, so block contrasts combine task, vigilance, and time-on-task effects.",
        "",
        "The figure shows the individual 2-s windows faintly and the equal-weight interval/block means as diamonds. The task blocks should be interpreted as one recording segment each, not as replicated subjects.",
        "",
        "Files: `bandpower_blocks.png`, `window_metrics.csv`, `unit_means.csv`, and `results.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(fx.to_jsonable(result), indent=2))


if __name__ == "__main__":
    # Allow direct execution from the repository root without installing the
    # analysis directory as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
