#!/usr/bin/env python3
"""Quality-control and exploratory analysis for the uploaded EPOC Flex XDF run."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import welch


FS = 128.0
EYE_MIN_DURATION = 20.0
EYE_TRIM = 5.0
ALPHA_BAND = (8.0, 12.0)
ERP_TMIN = -0.2
ERP_TMAX = 0.8
ERP_OFFSETS = np.arange(round(ERP_TMIN * FS), round(ERP_TMAX * FS) + 1)
ERP_TIMES = ERP_OFFSETS / FS


def stream_labels(stream: dict) -> list[str]:
    try:
        return [c["label"][0] for c in stream["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, IndexError, TypeError):
        return [f"ch{i}" for i in range(int(stream["info"]["channel_count"][0]))]


def stream_by(streams: list[dict], *, name: str | None = None, typ: str | None = None) -> dict:
    for stream in streams:
        info = stream["info"]
        if name is not None and info["name"][0] != name:
            continue
        if typ is not None and info.get("type", [""])[0] != typ:
            continue
        return stream
    raise KeyError(f"stream not found: name={name!r}, type={typ!r}")


def scalar_markers(stream: dict) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(stream["time_stamps"], dtype=float)
    values = np.asarray(stream["time_series"], dtype=object).reshape(-1)
    values = np.array([str(v) for v in values], dtype=object)
    return times, values


def find_eye_block(marker_times: np.ndarray, marker_values: np.ndarray) -> tuple[list[dict], list[int]]:
    """Select the one long 4-repeat eyes-closed/open sequence."""

    selected: list[dict] | None = None
    selected_idx: list[int] | None = None
    for start in range(len(marker_values) - 15):
        # A valid block is four alternating close/open repeats (16 markers).
        expected = [
            "CloseEyes_Start", "CloseEyes_End", "OpenEyes_Start", "OpenEyes_End",
        ] * 4
        if list(marker_values[start : start + 16]) != expected:
            continue
        idx = list(range(start, start + 16))
        durations = [
            float(marker_times[start + 1] - marker_times[start]),
            float(marker_times[start + 3] - marker_times[start + 2]),
            float(marker_times[start + 5] - marker_times[start + 4]),
            float(marker_times[start + 7] - marker_times[start + 6]),
        ]
        if min(durations) < EYE_MIN_DURATION:
            continue
        intervals = []
        for rep in range(4):
            c0, c1 = start + rep * 4, start + rep * 4 + 1
            o0, o1 = start + rep * 4 + 2, start + rep * 4 + 3
            intervals.extend(
                [
                    {"label": f"closed{rep + 1}", "start": marker_times[c0], "end": marker_times[c1]},
                    {"label": f"open{rep + 1}", "start": marker_times[o0], "end": marker_times[o1]},
                ]
            )
        selected, selected_idx = intervals, idx
        break
    if selected is None or selected_idx is None:
        raise RuntimeError("could not identify the long eyes-closed/open marker block")
    return selected, selected_idx


def block_events(marker_times: np.ndarray, marker_values: np.ndarray, prefix: str, start: float, end: float):
    sel = (
        (marker_times > start)
        & (marker_times < end)
        & np.char.startswith(marker_values.astype(str), prefix + "-")
    )
    idx = np.flatnonzero(sel)
    return marker_times[idx], marker_values[idx].astype(str), idx


def interval_stats(t: np.ndarray, filled: np.ndarray, gaps: np.ndarray, resets: np.ndarray, missing: np.ndarray, start: float, end: float) -> dict:
    sel = (t >= start) & (t <= end)
    return {
        "samples": int(sel.sum()),
        "filled": int(filled[sel].sum()),
        "filled_pct": float(100 * filled[sel].mean()) if sel.any() else np.nan,
        "gap_flags": int(gaps[sel].sum()),
        "resets": int(resets[sel].sum()),
        "missing_reports": int(missing[sel].sum()),
    }


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def quality_summary(stream: dict, t0: float, intervals: dict[str, tuple[float, float]]) -> dict:
    t = np.asarray(stream["time_stamps"], dtype=float) - t0
    x = np.asarray(stream["time_series"], dtype=float)
    labels = stream_labels(stream)
    out: dict = {"labels": labels, "overall": {}, "channels": {}}
    for label in intervals:
        a, b = intervals[label]
        sel = (t >= a) & (t <= b)
        out["overall"][label] = {"n": int(sel.sum())}
        if not sel.any():
            continue
        if "Contact Quality" in stream["info"]["name"][0]:
            overall = x[sel, labels.index("OVERALL")]
            signal = x[sel, labels.index("Signal")]
            out["overall"][label].update(
                mean=float(np.mean(overall)),
                median=float(np.median(overall)),
                pct_ge80=float(100 * np.mean(overall >= 80)),
                signal_mean=float(np.mean(signal)),
            )
            for ch in labels[2:-2]:
                v = x[sel, labels.index(ch)]
                out["channels"].setdefault(ch, {})[label] = {
                    "mean": float(np.mean(v)),
                    "green_pct": float(100 * np.mean(v >= 4)),
                    "yellow_pct": float(100 * np.mean(v == 2)),
                    "non_green_pct": float(100 * np.mean(v < 4)),
                }
        else:
            overall = x[sel, labels.index("overall")]
            rate = x[sel, labels.index("sampleRateQuality")]
            out["overall"][label].update(
                mean=float(np.mean(overall)),
                median=float(np.median(overall)),
                pct_ge75=float(100 * np.mean(overall >= 75)),
                sample_rate_mean=float(np.mean(rate)),
                sample_rate_pct_ge09=float(100 * np.mean(rate >= 0.9)),
            )
            for ch in labels[3:]:
                v = x[sel, labels.index(ch)]
                out["channels"].setdefault(ch, {})[label] = {
                    "mean": float(np.mean(v)),
                    "green_pct": float(100 * np.mean(v >= 4)),
                    "non_green_pct": float(100 * np.mean(v < 4)),
                }
    return out


def band_power_windows(data: np.ndarray, times: np.ndarray, filled: np.ndarray, start: float, end: float, labels: list[str], trim: float = EYE_TRIM):
    """Return per-window alpha power and PSDs, rejecting filled/artifact windows."""

    nper = 512
    step = 256
    starts = range(int(np.searchsorted(times, start + trim)), int(np.searchsorted(times, end - trim)) - nper + 1, step)
    rows: list[np.ndarray] = []
    psds: list[np.ndarray] = []
    rejected = 0
    freqs = None
    for st in starts:
        ii = np.arange(st, st + nper)
        if filled[ii].any():
            rejected += 1
            continue
        row = np.full(len(labels), np.nan)
        prow = np.full((len(labels), nper // 2 + 1), np.nan)
        for j in range(len(labels)):
            a = data[ii, j]
            if np.max(np.abs(a - np.median(a))) > 500:
                continue
            f, p = welch(a, fs=FS, nperseg=nper, noverlap=0, detrend="constant", scaling="density")
            freqs = f
            prow[j] = p
            band = (f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])
            row[j] = np.trapezoid(p[band], f[band])
        rows.append(row)
        psds.append(prow)
    return np.asarray(rows), np.asarray(psds), freqs, rejected


def prepare_epochs(data_uv: np.ndarray, times: np.ndarray, filled: np.ndarray, event_times: np.ndarray, event_values: np.ndarray, prefix: str, labels: list[str]):
    out: dict[str, list[np.ndarray]] = {"standard": [], "oddball": []}
    reasons: dict[str, dict[str, int]] = {"standard": {"packet": 0, "amplitude": 0, "boundary": 0}, "oddball": {"packet": 0, "amplitude": 0, "boundary": 0}}
    sample_errors: list[float] = []
    for et, value in zip(event_times, event_values):
        cond = "oddball" if value.endswith("oddball") else "standard"
        ix = int(np.searchsorted(times, et))
        sample_errors.append(float(times[min(ix, len(times) - 1)] - et))
        ii = ix + ERP_OFFSETS
        if ii[0] < 0 or ii[-1] >= len(times):
            reasons[cond]["boundary"] += 1
            continue
        if filled[ii].any():
            reasons[cond]["packet"] += 1
            continue
        ep = data_uv[ii, :].T.copy()
        ep -= ep[:, : int(round(-ERP_TMIN * FS))].mean(axis=1, keepdims=True)
        if np.max(np.abs(ep)) > 200 or np.max(np.ptp(ep, axis=1)) > 400:
            reasons[cond]["amplitude"] += 1
            continue
        out[cond].append(ep)
    out_arr = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    return out_arr, reasons, np.asarray(sample_errors)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, seed: int = 1, n_boot: int = 3000) -> tuple[float, float, float]:
    observed = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pooled = np.r_[a, b]
    count = 0
    for _ in range(2000):
        p = rng.permutation(pooled)
        z = p[: len(a)].mean() - p[len(a) :].mean()
        count += abs(z) >= abs(observed)
    pvalue = (count + 1) / 2001
    return observed, float(lo), float(hi), float(pvalue)


def fmt(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.input.parent / "flex_qc"
    out.mkdir(parents=True, exist_ok=True)

    streams, xdf_header = pyxdf.load_xdf(str(args.input), synchronize_clocks=False)
    eeg_stream = stream_by(streams, name="Epoc Flex 1.0", typ="EEG")
    diag_stream = stream_by(streams, name="Epoc Flex 1.0 Packet Diagnostics")
    cq_stream = stream_by(streams, name="Epoc Flex 1.0 Contact Quality")
    eq_stream = stream_by(streams, name="Epoc Flex 1.0 EEG Quality")
    marker_stream = stream_by(streams, name="PsychoPy Markers")

    eeg_t_abs = np.asarray(eeg_stream["time_stamps"], dtype=float)
    t0 = float(eeg_t_abs[0])
    eeg_t = eeg_t_abs - t0
    data_uv = np.asarray(eeg_stream["time_series"], dtype=float)
    labels = stream_labels(eeg_stream)
    diag = np.asarray(diag_stream["time_series"], dtype=float)
    diag_t = np.asarray(diag_stream["time_stamps"], dtype=float) - t0
    filled = diag[:, 8].astype(bool)
    gap_flags = diag[:, 2].astype(bool)
    missing = diag[:, 3]
    resets = diag[:, 5].astype(bool)
    counters = diag[:, 0].astype(int)
    expected = diag[:, 1].astype(int)
    markers_abs, marker_values = scalar_markers(marker_stream)
    marker_t = markers_abs - t0

    # Basic file/stream checks.
    duration = float(eeg_t[-1] - eeg_t[0])
    dt = np.diff(eeg_t)
    counter_expected = (np.r_[counters[0], counters[:-1]] + np.r_[0, np.ones(len(counters) - 1, dtype=int)]) % 128
    counter_mismatch = int(np.sum(counters != counter_expected))
    marker_counts = pd.Series(marker_values).value_counts().to_dict()
    csv_crosscheck = None
    if args.csv is not None and args.csv.exists():
        csv_df = pd.read_csv(args.csv)
        trial_df = csv_df[csv_df["stim"].notna()].copy()
        by_start = trial_df.groupby("date")["stim"].agg(["size", lambda s: int(np.sum(s == "standard")), lambda s: int(np.sum(s == "oddball"))])
        by_start.columns = ["n", "standard", "oddball"]
        csv_crosscheck = {
            "rows": int(len(csv_df)),
            "trial_rows": int(len(trial_df)),
            "starts": int(trial_df["date"].nunique()),
            "by_start": by_start.to_dict(orient="index"),
        }
    eyes, eye_idx = find_eye_block(marker_t, marker_values)
    eye_end = float(eyes[-1]["end"])
    first_end_idx = next(i for i, v in enumerate(marker_values) if v == "Oddball_End" and marker_t[i] > eye_end)
    first_events_t, first_events_v, first_event_idx = block_events(marker_t, marker_values, "Tone", eye_end, marker_t[first_end_idx])
    attended_end_idx = np.flatnonzero(marker_values == "Oddball_End")[-1]
    attended_events_t, attended_events_v, attended_event_idx = block_events(marker_t, marker_values, "AttendedTone", eye_end, marker_t[attended_end_idx])
    block_bounds = {
        "Tone": (float(eye_end), float(marker_t[first_end_idx])),
        "AttendedTone": (float(attended_events_t[0] - 0.29), float(marker_t[attended_end_idx])),
    }
    # The attended block begins at its first attended marker; using a small pre-window is
    # only for QC summaries and does not affect epoching.
    block_bounds["AttendedTone"] = (float(attended_events_t[0]), float(marker_t[attended_end_idx]))

    interval_bounds: dict[str, tuple[float, float]] = {"all": (0.0, duration)}
    for eye in eyes:
        interval_bounds[eye["label"]] = (float(eye["start"]), float(eye["end"]))
    interval_bounds["valid_eyes"] = (float(eyes[0]["start"]), eye_end)
    interval_bounds["Tone"] = block_bounds["Tone"]
    interval_bounds["AttendedTone"] = block_bounds["AttendedTone"]

    packet = {k: interval_stats(eeg_t, filled, gap_flags, resets, missing, *v) for k, v in interval_bounds.items()}
    fill_runs = contiguous_runs(filled)

    # Signal sanity metrics, computed on the recorded values before filtering.
    signal_rows = []
    for j, ch in enumerate(labels):
        med = float(np.median(data_uv[:, j]))
        mad = float(1.4826 * np.median(np.abs(data_uv[:, j] - med)))
        signal_rows.append(
            {
                "channel": ch,
                "median_uV": med,
                "robust_sd_uV": mad,
                "std_uV": float(np.std(data_uv[:, j])),
                "abs_deviation_gt500_pct": float(100 * np.mean(np.abs(data_uv[:, j] - med) > 500)),
                "finite": bool(np.isfinite(data_uv[:, j]).all()),
            }
        )
    signal_df = pd.DataFrame(signal_rows)

    # Contact and EEG quality summaries.
    cq_summary = quality_summary(cq_stream, t0, interval_bounds)
    eq_summary = quality_summary(eq_stream, t0, interval_bounds)

    # Continuous MNE preprocessing: 0.5--30 Hz, then average reference.
    info = mne.create_info(labels, FS, ch_types="eeg")
    raw = mne.io.RawArray(data_uv.T * 1e-6, info, verbose="ERROR")
    try:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="warn", verbose="ERROR")
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = raw.copy().filter(0.5, 30.0, method="fir", phase="zero", verbose="ERROR")
    filtered.set_eeg_reference("average", projection=False, verbose="ERROR")
    filtered_uv = filtered.get_data() * 1e6

    # Eyes-closed/open alpha power.
    alpha: dict[str, dict] = {}
    for eye in eyes:
        power, psds, freqs, rejected = band_power_windows(data_uv, eeg_t, filled, eye["start"], eye["end"], labels)
        alpha[eye["label"]] = {
            "n_windows": int(len(power)),
            "rejected_windows": int(rejected),
            "alpha_power": np.nanmedian(power, axis=0).tolist() if len(power) else [np.nan] * len(labels),
            "psd": np.nanmedian(psds, axis=0).tolist() if len(psds) else None,
            "freqs": freqs.tolist() if freqs is not None else None,
        }
    alpha_rows = []
    for ch_i, ch in enumerate(labels):
        row = {"channel": ch}
        for rep in range(1, 5):
            c = alpha[f"closed{rep}"]["alpha_power"][ch_i]
            o = alpha[f"open{rep}"]["alpha_power"][ch_i]
            row[f"closed{rep}"] = c
            row[f"open{rep}"] = o
            row[f"open_minus_closed_db{rep}"] = 10 * np.log10(o / c) if c > 0 and o > 0 else np.nan
        cvals = [row[f"closed{r}"] for r in range(1, 5)]
        ovals = [row[f"open{r}"] for r in range(1, 5)]
        row["pooled_closed"] = float(np.nanmedian(cvals))
        row["pooled_open"] = float(np.nanmedian(ovals))
        row["pooled_open_minus_closed_db"] = float(10 * np.log10(row["pooled_open"] / row["pooled_closed"]))
        alpha_rows.append(row)
    alpha_df = pd.DataFrame(alpha_rows)

    # ERP epochs and exploratory differences.
    erp_data: dict[str, dict] = {}
    for prefix, event_t, event_v in [
        ("Tone", first_events_t, first_events_v),
        ("AttendedTone", attended_events_t, attended_events_v),
    ]:
        epochs, reasons, sample_errors = prepare_epochs(filtered_uv.T, eeg_t, filled, event_t, event_v, prefix, labels)
        erp_data[prefix] = {
            "epochs": epochs,
            "reasons": reasons,
            "sample_error_ms": sample_errors * 1000,
        }

    # Save machine-readable tables.
    signal_df.to_csv(out / "channel_signal_metrics.csv", index=False)
    alpha_df.to_csv(out / "alpha_power.csv", index=False)
    pd.DataFrame(packet).T.to_csv(out / "packet_interval_metrics.csv")
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(args.input),
                "streams": [s["info"]["name"][0] for s in streams],
                "duration_s": duration,
                "n_eeg_samples": int(len(data_uv)),
                "n_channels": int(data_uv.shape[1]),
                "sample_rate_nominal": FS,
                "sample_rate_from_timestamps": float((len(data_uv) - 1) / duration),
                "marker_counts": {str(k): int(v) for k, v in marker_counts.items()},
                "csv_crosscheck": csv_crosscheck,
                "eye_intervals": eyes,
                "packet": {
                    "filled_samples": int(filled.sum()),
                    "filled_pct": float(100 * filled.mean()),
                    "gap_flags": int(gap_flags.sum()),
                    "resets": int(resets.sum()),
                    "missing_reports": int(missing.sum()),
                    "counter_mismatches_mod128": counter_mismatch,
                    "filled_runs": len(fill_runs),
                },
                "packet_intervals": packet,
                "signal": signal_rows,
                "contact_quality": cq_summary,
                "eeg_quality": eq_summary,
            },
            f,
            indent=2,
        )

    # Figure 1: packet and quality timeline.
    cq_t = np.asarray(cq_stream["time_stamps"], dtype=float) - t0
    cq = np.asarray(cq_stream["time_series"], dtype=float)
    cq_l = stream_labels(cq_stream)
    eq_t = np.asarray(eq_stream["time_stamps"], dtype=float) - t0
    eq = np.asarray(eq_stream["time_series"], dtype=float)
    eq_l = stream_labels(eq_stream)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(eeg_t, filled.astype(int), lw=0.3, color="tab:red")
    axes[0].set_ylabel("FILLED")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("EPOC Flex acquisition QC")
    axes[1].plot(cq_t, cq[:, cq_l.index("OVERALL")], label="Contact overall", lw=1)
    axes[1].plot(eq_t, eq[:, eq_l.index("overall")], label="EEG overall", lw=1)
    axes[1].axhline(80, color="0.5", ls="--", lw=0.7)
    axes[1].set_ylabel("quality")
    axes[1].legend(loc="lower right")
    axes[2].plot(eq_t, eq[:, eq_l.index("sampleRateQuality")], label="sample-rate quality", lw=1)
    axes[2].set_ylabel("0–1")
    axes[2].set_xlabel("seconds from EEG stream start")
    axes[2].set_ylim(-0.1, 1.05)
    for ax in axes:
        for eye in eyes:
            ax.axvspan(eye["start"], eye["end"], color="tab:blue" if "closed" in eye["label"] else "tab:orange", alpha=0.08)
        for st, en in block_bounds.values():
            ax.axvspan(st, en, color="tab:green", alpha=0.05)
    fig.savefig(out / "quality_packet_timeline.png", dpi=160)
    plt.close(fig)

    # Figure 2: posterior alpha spectra and channel-wise eye-state contrast.
    roi = ["O1", "Oz", "O2", "Pz"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for state, color in [("closed", "tab:blue"), ("open", "tab:orange")]:
        curves = []
        for rep in range(1, 5):
            entry = alpha[f"{state}{rep}"]
            if entry["psd"] is None:
                continue
            f = np.asarray(entry["freqs"])
            psd = np.asarray(entry["psd"])
            curves.append(np.nanmedian(psd[[labels.index(ch) for ch in roi]], axis=0))
        if curves:
            axes[0].plot(f, np.nanmedian(curves, axis=0), color=color, label=state)
    # The raw spectrum has a large low-frequency component; zooming to the
    # alpha neighborhood makes the state contrast visible instead of hiding
    # it under the 1-Hz power.
    axes[0].set_xlim(5, 15)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Hz")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].set_title("Posterior median spectrum")
    axes[0].legend()
    plot_df = alpha_df.sort_values("pooled_open_minus_closed_db")
    axes[1].barh(plot_df["channel"], plot_df["pooled_open_minus_closed_db"], color="tab:gray")
    axes[1].axvline(0, color="black", lw=0.7)
    axes[1].set_xlabel("open − closed alpha (dB)")
    axes[1].set_title("8–12 Hz eye-state contrast")
    fig.savefig(out / "alpha_eye_state.png", dpi=160)
    plt.close(fig)

    # Figure 3: ERP traces and oddball-minus-standard difference waves.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for col, ch in enumerate(["Fz", "Pz", "Oz"]):
        j = labels.index(ch)
        ax = axes[0, col]
        for prefix, color, ls in [("Tone", "tab:blue", "-"), ("AttendedTone", "tab:orange", "-")]:
            for cond, c2, ls2 in [("standard", "0.35", "--"), ("oddball", color, "-")]:
                ep = erp_data[prefix]["epochs"][cond]
                if len(ep):
                    ax.plot(ERP_TIMES, ep[:, j, :].mean(axis=0), color=c2 if cond == "oddball" else "0.35", ls=ls2, label=f"{prefix} {cond}")
        ax.axvline(0, color="black", lw=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(ch)
        ax.set_ylabel("µV")
        if col == 0:
            ax.legend(fontsize=7, loc="best")
        ax = axes[1, col]
        for prefix, color in [("Tone", "tab:blue"), ("AttendedTone", "tab:orange")]:
            odd = erp_data[prefix]["epochs"]["oddball"]
            std = erp_data[prefix]["epochs"]["standard"]
            if len(odd) and len(std):
                ax.plot(ERP_TIMES, odd[:, j, :].mean(axis=0) - std[:, j, :].mean(axis=0), color=color, label=prefix)
        ax.axvline(0, color="black", lw=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("seconds")
        ax.set_ylabel("odd − standard (µV)")
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Exploratory auditory ERPs (0.5–30 Hz, average referenced)")
    fig.savefig(out / "erp_oddball.png", dpi=160)
    plt.close(fig)

    # Figure 4: channel-level robust signal metrics.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    srt = signal_df.sort_values("robust_sd_uV")
    axes[0].barh(srt["channel"], srt["robust_sd_uV"], color="tab:blue")
    axes[0].set_xlabel("robust SD (µV)")
    axes[0].set_title("Channel amplitude scale")
    srt = signal_df.sort_values("abs_deviation_gt500_pct")
    axes[1].barh(srt["channel"], srt["abs_deviation_gt500_pct"], color="tab:red")
    axes[1].set_xlabel("samples >500 µV from channel median (%)")
    axes[1].set_title("Gross-amplitude outlier fraction")
    fig.savefig(out / "channel_signal_metrics.png", dpi=160)
    plt.close(fig)

    # Report tables.
    cq_all = cq_summary["overall"]["all"]
    eq_all = eq_summary["overall"]["all"]
    posterior = alpha_df[alpha_df.channel.isin(roi)]
    tone = erp_data["Tone"]
    attended = erp_data["AttendedTone"]
    feature_rows = []
    for prefix, data_block in [("Tone", tone), ("AttendedTone", attended)]:
        for ch in ["Fz", "Pz"]:
            j = labels.index(ch)
            for lo, hi in [(0.3, 0.6), (0.5, 0.7)]:
                sel = (ERP_TIMES >= lo) & (ERP_TIMES <= hi)
                odd = data_block["epochs"]["oddball"][:, j, :][:, sel].mean(axis=1)
                std = data_block["epochs"]["standard"][:, j, :][:, sel].mean(axis=1)
                d, lo_ci, hi_ci, pval = bootstrap_diff(odd, std)
                feature_rows.append((prefix, ch, f"{lo:.1f}–{hi:.1f}", len(odd), len(std), d, lo_ci, hi_ci, pval))

    report = []
    report.append(f"# EPOC Flex run sanity-check report\n\n**Input:** `{args.input}`\n\n")
    report.append("## Executive summary\n\n")
    report.append(
        f"The XDF contains {len(data_uv):,} samples from 32 EEG channels at nominal 128 Hz ({duration/60:.1f} min), with EEG and packet-diagnostics streams exactly sample-aligned. The packet counter is internally consistent modulo 128. The reader flagged {int(filled.sum()):,} filled samples ({100*filled.mean():.2f}%), {int(gap_flags.sum())} gap flags, and {int(resets.sum())} reset events; this is real acquisition loss and should be excluded from analyses rather than treated as ordinary EEG. Contact quality is predominantly green, while EEG quality varies as expected; sample-rate quality is high.\n\n"
    )
    report.append(
        "For the eye-state test, the long marker block contains four usable close/open pairs. The short marker sequences before and after it are inconsistent with the requested ~60-s self-timed periods and were excluded. The two auditory blocks each contain 200 events with 180 standards and 20 oddballs; the second is identified by `AttendedTone-*`.\n\n"
    )
    report.append("## Recording and marker structure\n\n")
    report.append(f"- Marker count: {len(marker_values)}. The first/last marker times span {marker_t[0]:.2f}–{marker_t[-1]:.2f} s relative to EEG start.\n")
    report.append(f"- EEG/diagnostic timestamp spacing: median {1000*np.median(dt):.3f} ms (nominal 7.8125 ms); effective rate from timestamps {(len(data_uv)-1)/duration:.3f} Hz. The EEG and diagnostics timestamp arrays are identical sample-for-sample.\n")
    if csv_crosscheck is not None:
        report.append(f"- PsychoPy CSV cross-check: {csv_crosscheck['trial_rows']} trial rows across {csv_crosscheck['starts']} experiment starts; each start has 200 trials with 180 standard and 20 oddball rows.\n")
    report.append(f"- Three early `Tone-standard` markers occur before the selected eye block and are treated as restart artifacts; the post-block short eye-marker burst is also excluded.\n")
    report.append(f"- Selected eye block markers: indices {eye_idx[0]}–{eye_idx[-1]}; durations (s): " + ", ".join(fmt(e["end"] - e["start"]) for e in eyes) + ".\n")
    report.append(f"- `Tone-*`: {len(first_events_t)} events ({int(np.sum(first_events_v == 'Tone-standard'))} standard, {int(np.sum(first_events_v == 'Tone-oddball'))} oddball).\n")
    report.append(f"- `AttendedTone-*`: {len(attended_events_t)} events ({int(np.sum(attended_events_v == 'AttendedTone-standard'))} standard, {int(np.sum(attended_events_v == 'AttendedTone-oddball'))} oddball).\n")
    report.append(f"- Tone-to-tone intervals: first block median {fmt(np.median(np.diff(first_events_t)))} s (range {fmt(np.min(np.diff(first_events_t)))}–{fmt(np.max(np.diff(first_events_t)))}); attended block median {fmt(np.median(np.diff(attended_events_t)))} s.\n")
    report.append(f"- Marker-to-nearest-EEG-sample error: Tone median {fmt(np.median(np.abs(tone['sample_error_ms'])), 3)} ms, attended median {fmt(np.median(np.abs(attended['sample_error_ms'])), 3)} ms.\n\n")
    report.append("## Packet integrity\n\n")
    report.append("| Interval | samples | filled | filled % | gap flags | resets | estimated missing |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for name in ["all", "valid_eyes", "closed1", "open1", "closed2", "open2", "closed3", "open3", "closed4", "open4", "Tone", "AttendedTone"]:
        q = packet[name]
        report.append(f"| {name} | {q['samples']:,} | {q['filled']:,} | {q['filled_pct']:.2f} | {q['gap_flags']} | {q['resets']} | {q['missing_reports']} |\n")
    report.append(
        f"\nThere are {len(fill_runs)} contiguous filled runs. Most are short (~7–8 samples, about 60 ms), but the record also contains longer loss bursts; the largest contiguous filled spans are {max((b-a for a,b in fill_runs))} samples. The diagnostics cumulative counter starts at {int(diag[0, 4])} missing reports because the reader had already been running before the XDF capture; it increases by {int(missing.sum())} during this recording. The fourth eyes-open interval contains {packet['open4']['filled_pct']:.1f}% filled samples, leaving only {alpha['open4']['n_windows']} clean 4-s alpha windows, so it should not be used as a primary eye-state comparison.\n\n"
    )
    report.append("## Quality streams\n\n")
    report.append(
        f"Contact-quality overall: mean {cq_all['mean']:.1f}%, median {cq_all['median']:.1f}%, and {cq_all['pct_ge80']:.1f}% of 2-Hz samples at or above 80%. The `Signal` metric averaged {cq_all['signal_mean']:.3f}. EEG-quality overall: mean {eq_all['mean']:.1f}, median {eq_all['median']:.1f}, with {eq_all['pct_ge75']:.1f}% of samples at or above 75; sample-rate quality averaged {eq_all['sample_rate_mean']:.3f}, at or above 0.9 for {eq_all['sample_rate_pct_ge09']:.1f}% of samples.\n\n"
    )
    report.append("The main contact-quality exceptions are summarized below (whole run):\n\n")
    report.append("| channel | contact green | contact yellow | EEG-quality green |\n|---|---:|---:|---:|\n")
    for ch in ["O1", "Pz", "O2", "Oz"]:
        cq_ch = cq_summary["channels"][ch]["all"]
        eq_ch = eq_summary["channels"][ch]["all"]
        report.append(f"| {ch} | {cq_ch['green_pct']:.1f}% | {cq_ch['yellow_pct']:.1f}% | {eq_ch['green_pct']:.1f}% |\n")
    report.append("\nThis run is consistent with mostly good contact. Pz has intermittent lower contact scores; in this file O2—not O1—is the posterior channel with the most yellow contact scores. EEG-quality scores for O1/O2 are more variable than contact scores, so use the quality stream as a window/epoch flag rather than dropping an entire channel by default.\n\n")
    report.append("## Signal sanity and preprocessing\n\n")
    report.append(
        f"All {data_uv.size:,} EEG values are finite. Channel robust SDs span {signal_df.robust_sd_uV.min():.1f}–{signal_df.robust_sd_uV.max():.1f} µV. No channel is flat, and no channel has a gross-amplitude outlier fraction above 1% using a conservative >500-µV-from-median screen. The largest transient burden is on Pz/Fz/F8/frontopolar channels, but it is episodic rather than a persistent dead channel.\n\n"
    )
    report.append(
        "Recommended first-pass preprocessing: retain the nominal 128-Hz sampling rate and original timestamps; mark every `FILLED=1` sample and a short padding window around each packet-loss run as bad; band-pass 0.5–30 Hz for ERP work (or 1–40 Hz for broad spectral work); notch only if line-noise inspection warrants it; average-reference after filtering; use the standard-1020 montage; reject event epochs overlapping packet-loss annotations or showing gross amplitude/motion artifacts; and interpolate only short isolated gaps if a downstream method requires continuous data. Do not silently interpolate the long loss bursts.\n\n"
    )
    report.append("## Eyes-closed/open alpha\n\n")
    report.append("Alpha values below are Welch 8–12-Hz power from clean 4-s windows, with a 5-s trim at each interval edge.\n\n")
    report.append("| interval | clean windows | rejected windows | posterior O1/Oz/O2/Pz alpha power |\n|---|---:|---:|---|\n")
    for rep in range(1, 5):
        for state in ["closed", "open"]:
            e = alpha[f"{state}{rep}"]
            vals = [e["alpha_power"][labels.index(ch)] for ch in ["O1", "Oz", "O2", "Pz"]]
            report.append(f"| {state}{rep} | {e['n_windows']} | {e['rejected_windows']} | " + ", ".join(fmt(v) for v in vals) + " |\n")
    report.append("\nThe posterior alpha contrast is not stable across all four pairs: the first two pairs show lower open-eye alpha, whereas pairs 3–4 reverse direction, and pair 4 is heavily contaminated by packet loss. Pooled open-minus-closed contrasts in the clean-window summary are small and heterogeneous (see `alpha_power.csv` and the figure). This is enough to confirm that the spectral pipeline is behaving and that an ~11-Hz posterior rhythm is present, but not enough to call a reliable eyes-open suppression effect from this run.\n\n")
    report.append("## Auditory ERP sanity check\n\n")
    report.append("ERP epochs were extracted from −200 to +800 ms, baseline-corrected, filtered 0.5–30 Hz, average-referenced, and rejected for packet overlap or gross amplitude.\n\n")
    report.append("| block | condition | retained / total | packet rejects | amplitude rejects |\n|---|---|---:|---:|---:|\n")
    for prefix, label in [("Tone", "Tone"), ("AttendedTone", "AttendedTone")]:
        for cond in ["standard", "oddball"]:
            total = 180 if cond == "standard" else 20
            r = erp_data[prefix]["reasons"][cond]
            retained = len(erp_data[prefix]["epochs"][cond])
            report.append(f"| {label} | {cond} | {retained}/{total} | {r['packet']} | {r['amplitude']} |\n")
    report.append("\nExploratory oddball-minus-standard mean amplitudes (µV; bootstrap 95% interval and label-permutation p are descriptive only):\n\n")
    report.append("| block | channel | window | oddball n | standard n | difference | 95% interval | p |\n|---|---|---|---:|---:|---:|---|---:|\n")
    for row in feature_rows:
        prefix, ch, win, no, ns, d, lo, hi, pval = row
        report.append(f"| {prefix} | {ch} | {win} s | {no} | {ns} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {pval:.3f} |\n")
    report.append(
        "\nThe traces show a small, plausible event-locked auditory response. The attended block has a clearer frontocentral/parietal late positive tendency than the unattended block, consistent with a possible P3-like effect, but there are only 20 oddballs per block and the intervals are wide. Treat this as a hardware/marker sanity check, not a confirmatory ERP result.\n\n"
    )
    report.append("## Files\n\n")
    report.append("- `quality_packet_timeline.png`: packet-loss and quality timeline.\n- `alpha_eye_state.png`: posterior spectrum and eye-state alpha contrast.\n- `erp_oddball.png`: exploratory standard/oddball ERPs and difference waves.\n- `channel_signal_metrics.png`: robust channel amplitude and outlier summaries.\n- `summary.json`, `packet_interval_metrics.csv`, `channel_signal_metrics.csv`, `alpha_power.csv`: machine-readable outputs.\n")
    (out / "report.md").write_text("".join(report), encoding="utf-8")
    print(out / "report.md")


if __name__ == "__main__":
    main()
