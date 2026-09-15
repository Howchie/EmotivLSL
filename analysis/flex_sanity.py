#!/usr/bin/env python3
"""Hardware sanity check for EPOC Flex 1.0 XDF recordings.

Covers, per recording and pooled:

* acquisition integrity: decoder consistency, filled (lost) samples, dropped
  repeats, counter resets, timestamp linearity inside each task block;
* signal quality: spectra as recorded (CMS reference) and average-referenced,
  mains lines, per-channel noise, slew-limit saturation, data-driven bad
  channels, and whether the Cortex quality streams track any of it;
* eyes closed/open alpha, scored against the neighbouring frequencies so the
  broadband rise that comes with open eyes cannot mask it;
* auditory ERPs to standards (N1/P2 and its latency) and oddball-minus-standard
  effects for the passive and attended blocks (cluster permutation tests).

Figures, tables and ``results.json`` go to the output directory; the prose
report is written separately from those results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy import stats
from scipy.signal import butter, find_peaks, medfilt, sosfiltfilt, welch

mne.set_log_level("ERROR")

FS = 128.0
LSB_UV = 0.51
FILL_PAD = (0.25, 1.0)  # s before/after a filled run; under the 0.16 Hz leak the lost-delta step decays with tau ~1 s
EYE_TRIM = (3.0, 2.0)  # s dropped at the start/end of each self-timed eye interval
ERP_BAND = (0.1, 30.0)
# Epochs run to 1.0 s because the tone-locked response lands ~200 ms after the marker
# (see the N1 latency in results.json); the next tone's response starts after ~1.0 s.
ERP_WINDOW = (-0.2, 1.0)
REJECT_UV = 150.0
FC_ROI = ["Fz", "FC1", "FC2", "Cz"]
INFERIOR_ROI = ["FT9", "FT10", "PO9", "PO10"]
POSTERIOR_ROI = ["O1", "Oz", "O2", "PO9", "PO10", "P7", "P8", "P3", "P4", "Pz"]
SEED = 7

RUNS = {
    "run1": "data/Oddball/data/sub-flex_ses-S001_task-Default_run-001_eeg.xdf",
    "run2": "data/Oddball/data/sub-flex_oddball2.xdf",
}
EPOCX_PILOT = "data/Oddball/data/sub-epochx_0740_ses-S001_task-Default_run-001_eeg.xdf"
EYE_SEQUENCE = ["CloseEyes_Start", "CloseEyes_End", "OpenEyes_Start", "OpenEyes_End"] * 4
# Marker prefix -> block name.  "Oddball-" is the second recording's passive block.
BLOCKS = {"Tone": "passive", "Oddball": "passive", "AttendedTone": "attended"}


# ---------------------------------------------------------------------------
# Loading


def _labels(stream: dict) -> list[str]:
    return [c["label"][0] for c in stream["info"]["desc"][0]["channels"][0]["channel"]]


@dataclass
class Run:
    key: str
    t: np.ndarray  # EEG timestamps relative to the first sample
    x_uv: np.ndarray  # samples x channels, as published
    labels: list[str]
    diag: dict[str, np.ndarray]
    markers: pd.DataFrame  # time, value
    cq: pd.DataFrame | None
    eq: pd.DataFrame | None
    eye_intervals: list[dict] = field(default_factory=list)
    tone_events: pd.DataFrame | None = None
    # Flex decoder settings from the stream metadata; the defaults are what
    # recordings made before the zero fix used.
    dc_restore_hz: float = 0.16
    delta_zero: int = 64

    @property
    def filled(self) -> np.ndarray:
        return self.diag["FILLED"].astype(bool)

    @property
    def leak(self) -> float:
        return math.exp(-2 * math.pi * self.dc_restore_hz / FS)


def load_run(key: str, path: str) -> Run:
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    eeg = by_name("Epoc Flex 1.0")[0]
    diag_stream = by_name("Epoc Flex 1.0 Packet Diagnostics")[0]
    t_abs = np.asarray(eeg["time_stamps"], float)
    t0 = t_abs[0]
    diag_values = np.asarray(diag_stream["time_series"], float)
    diag = {name: diag_values[:, i] for i, name in enumerate(_labels(diag_stream))}

    # LabRecorder can pick the same marker outlet up twice; keep one copy of each marker.
    rows = set()
    for stream in by_name("PsychoPy Markers"):
        for ts, value in zip(stream["time_stamps"], stream["time_series"]):
            rows.add((round(float(ts) - t0, 4), str(value[0])))
    markers = pd.DataFrame(sorted(rows), columns=["time", "value"])

    quality = {}
    for name, short in [("Epoc Flex 1.0 Contact Quality", "cq"), ("Epoc Flex 1.0 EEG Quality", "eq")]:
        stream = by_name(name)
        if stream and len(stream[0]["time_stamps"]):
            frame = pd.DataFrame(np.asarray(stream[0]["time_series"], float), columns=_labels(stream[0]))
            frame.insert(0, "time", np.asarray(stream[0]["time_stamps"]) - t0)
            quality[short] = frame
        else:
            quality[short] = None

    run = Run(key, t_abs - t0, np.asarray(eeg["time_series"], float), _labels(eeg), diag, markers,
              quality["cq"], quality["eq"])
    cap = eeg["info"]["desc"][0].get("cap", [{}])[0]
    if "dc_restore_hz" in cap:
        run.dc_restore_hz = float(cap["dc_restore_hz"][0])
    if "delta_zero" in cap:
        run.delta_zero = int(cap["delta_zero"][0])
    run.eye_intervals = find_eye_intervals(markers)
    run.tone_events = find_tone_events(run)
    return run


def find_eye_intervals(markers: pd.DataFrame, min_duration: float = 20.0) -> list[dict]:
    """The one complete 4x closed/open sequence whose intervals all last >= min_duration."""

    values = markers["value"].tolist()
    times = markers["time"].to_numpy()
    for start in range(len(values) - len(EYE_SEQUENCE) + 1):
        if values[start:start + len(EYE_SEQUENCE)] != EYE_SEQUENCE:
            continue
        intervals = []
        for rep in range(4):
            base = start + 4 * rep
            intervals.append({"state": "closed", "pair": rep + 1, "start": times[base], "end": times[base + 1]})
            intervals.append({"state": "open", "pair": rep + 1, "start": times[base + 2], "end": times[base + 3]})
        if min(iv["end"] - iv["start"] for iv in intervals) >= min_duration:
            return intervals
    raise RuntimeError("no complete eyes closed/open sequence found")


def find_tone_events(run: Run) -> pd.DataFrame:
    """Tone markers inside each oddball block, mapped to the nearest EEG sample by timestamp."""

    eye_end = run.eye_intervals[-1]["end"]
    rows = []
    for time, value in run.markers.itertuples(index=False):
        prefix, _, stim = value.partition("-")
        if prefix not in BLOCKS or stim not in ("standard", "oddball"):
            continue
        # The three "Tone-" markers before the eye block are from the aborted PsychoPy start.
        if prefix == "Tone" and time < eye_end:
            continue
        sample = int(np.clip(np.searchsorted(run.t, time), 1, len(run.t) - 1))
        sample -= int(abs(run.t[sample - 1] - time) < abs(run.t[sample] - time))
        rows.append({"time": time, "prefix": prefix, "block": BLOCKS[prefix], "stim": stim,
                     "sample": sample, "sample_error_ms": 1000 * (run.t[sample] - time)})
    return pd.DataFrame(rows)


def contiguous(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ---------------------------------------------------------------------------
# Integrity


def integrity(run: Run) -> dict:
    d = run.diag
    filled = run.filled
    starts, stops = contiguous(filled)
    lengths = stops - starts
    reset = d["RESET_FLAG"].astype(bool)

    # Decoder consistency: undoing the leak must give whole-count deltas within the seven-bit range.
    counts = run.x_uv / LSB_UV
    if np.median(counts) > 4000:
        counts = counts - 8192
    deltas = counts[1:] - run.leak * counts[:-1]
    real = ~filled[1:]
    frac = np.abs(deltas[real] - np.round(deltas[real]))
    whole = np.round(deltas[real]).astype(int)
    saturated = (whole <= -run.delta_zero) | (whole >= 127 - run.delta_zero)
    # A lost 8-sample block drops 8 deltas, leaving a step equal to the signal's change over 8 samples.
    step8 = counts[8:] - run.leak ** 8 * counts[:-8]
    clean8 = np.convolve(filled.astype(int), np.ones(9, int), mode="valid") == 0
    step8_rms_uv = LSB_UV * np.sqrt(np.mean(step8[clean8] ** 2, axis=0))

    # Timestamp linearity: robust line through each stretch between counter resets.
    edges = [0, *np.flatnonzero(reset).tolist(), len(run.t)]
    residual_ms = np.full(len(run.t), np.nan)
    segments = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1000:
            continue
        idx = np.arange(a, b)
        keep = np.ones(len(idx), bool)
        for _ in range(5):
            fit = np.polyfit(idx[keep], run.t[a:b][keep], 1)
            res = run.t[a:b] - np.polyval(fit, idx)
            keep = np.abs(res - np.median(res[keep])) < 0.005
        residual_ms[a:b] = 1000 * res
        segments.append({"start_s": float(run.t[a]), "end_s": float(run.t[b - 1]), "rate_hz": float(1 / fit[0])})

    blocks = {}
    for (block, prefix), ev in run.tone_events.groupby(["block", "prefix"]):
        a, b = ev["sample"].min(), ev["sample"].max()
        span = slice(max(0, a - int(FS)), b + int(FS))
        near = np.array([filled[max(0, s - int(FILL_PAD[1] * FS)):s + int(ERP_WINDOW[1] * FS) + 1].any()
                         for s in ev["sample"]])
        blocks[prefix] = {
            "block": block,
            "tones": int(len(ev)),
            "oddballs": int((ev["stim"] == "oddball").sum()),
            "duration_s": float(ev["time"].max() - ev["time"].min()),
            "isi_median_s": float(np.median(np.diff(ev["time"]))),
            "isi_range_s": [float(np.min(np.diff(ev["time"]))), float(np.max(np.diff(ev["time"])))],
            "filled_samples": int(filled[span].sum()),
            "filled_runs": int(((starts >= span.start) & (starts < span.stop)).sum()),
            "tones_near_fill": int(near.sum()),
            "timestamp_residual_max_ms": float(np.nanmax(np.abs(residual_ms[span]))),
            "timestamp_residual_p99_ms": float(np.nanpercentile(np.abs(residual_ms[span]), 99)),
            "marker_to_sample_error_ms_max": float(ev["sample_error_ms"].abs().max()),
        }

    run_lengths, run_counts = np.unique(lengths, return_counts=True)
    return {
        "samples": int(len(run.t)),
        "duration_s": float(run.t[-1]),
        "filled_samples": int(filled.sum()),
        "filled_pct": float(100 * filled.mean()),
        "filled_runs": int(len(starts)),
        "filled_run_lengths": {int(k): int(v) for k, v in zip(run_lengths, run_counts)},
        "filled_runs_len_8_or_16": int(np.isin(lengths, [7, 8, 9, 16, 17]).sum()),
        "longest_fill_s": float(lengths.max() / FS) if len(lengths) else 0.0,
        "dropped_repeats": int(((d["GAP_FLAG"] > 0) & (d["MISSING_REPORTS"] == 0)).sum()),
        "counter_resets": int(reset.sum()),
        "counter_steps_not_one": int((np.diff(d["COUNTER"].astype(int)) % 128 != 1).sum()),
        "decoder_max_fraction": float(frac.max()),
        "decoder_delta_range": [int(whole.min()), int(whole.max())],
        "slew_saturated_pct": float(100 * saturated.mean()),
        "fill8_step_rms_uv_median_channel": float(np.median(step8_rms_uv)),
        "slew_saturated_pct_by_channel": {ch: float(100 * saturated[:, j].mean()) for j, ch in enumerate(run.labels)},
        "timestamp_segments": segments,
        "blocks": blocks,
        "_residual_ms": residual_ms,
    }


# ---------------------------------------------------------------------------
# Continuous data, bad spans and bad channels


def make_raw(run: Run) -> mne.io.RawArray:
    info = mne.create_info(run.labels, FS, "eeg")
    raw = mne.io.RawArray(run.x_uv.T * 1e-6, info)
    raw.set_montage("standard_1020")
    starts, stops = contiguous(run.filled)
    onset = np.maximum(0.0, starts / FS - FILL_PAD[0])
    duration = (stops - starts) / FS + sum(FILL_PAD)
    raw.set_annotations(mne.Annotations(onset, duration, ["BAD_fill"] * len(onset)))
    return raw


def good_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """Samples outside BAD annotations."""

    mask = np.ones(raw.n_times, bool)
    for ann in raw.annotations:
        if ann["description"].startswith("BAD"):
            a = int(ann["onset"] * FS)
            mask[a:a + int(math.ceil(ann["duration"] * FS))] = False
    return mask


def channel_noise(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Per-channel noise on clean spans (median reference).

    A channel is flagged when its 20-40 Hz power (above the EEG, where the
    Flex's own noise dominates) is more than 4x the median channel, or when it
    is flat.  Channel noise spans a ~20x range on this cap, so a z-score over
    channels is too lenient, and neighbour correlations are uninformative once
    the shared reference signal is removed.
    """

    x = raw.get_data() * 1e6
    x = x - np.median(x, axis=0, keepdims=True)
    ok = good_mask(raw)
    hf = sosfiltfilt(butter(4, [20, 40], btype="band", fs=FS, output="sos"), x, axis=1)[:, ok]
    lf = sosfiltfilt(butter(4, [1, 20], btype="band", fs=FS, output="sos"), x, axis=1)[:, ok]
    hf_power = np.mean(hf ** 2, axis=1)
    frame = pd.DataFrame({
        "channel": raw.ch_names,
        "hf_20_40_uv2": hf_power,
        "hf_ratio_to_median": hf_power / np.median(hf_power),
        "rms_1_20_uv": np.sqrt(np.mean(lf ** 2, axis=1)),
    })
    frame["bad"] = (frame["hf_ratio_to_median"] > 4) | (frame["rms_1_20_uv"] < 1)
    return frame


# ---------------------------------------------------------------------------
# Spectra and quality streams


def clean_psd(x: np.ndarray, ok: np.ndarray, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD (channels x freqs) pooled over clean stretches, weighted by length."""

    starts, stops = contiguous(ok)
    total, weight, freqs = None, 0, None
    for a, b in zip(starts, stops):
        if b - a < nperseg:
            continue
        freqs, p = welch(x[:, a:b], fs=FS, nperseg=nperseg, axis=1)
        total = p * (b - a) if total is None else total + p * (b - a)
        weight += b - a
    return freqs, total / weight


def line_check(freqs: np.ndarray, psd_median: np.ndarray) -> dict:
    baseline = np.exp(medfilt(np.log(psd_median), 41))
    ratio = psd_median / baseline
    out = {}
    # 50 Hz and where its harmonics alias to at 128 Hz.
    for f0 in (50, 22, 28, 6):
        near = np.abs(freqs - f0) <= 0.3
        out[f"{f0}Hz_peak_ratio"] = float(ratio[near].max())
    out["largest_narrowband_ratio_2_60Hz"] = float(ratio[(freqs > 2) & (freqs < 60)].max())
    return out


def quality_vs_noise(run: Run, raw: mne.io.BaseRaw) -> dict | None:
    """Do Cortex's per-channel quality numbers track the measured per-channel noise?"""

    if run.eq is None or run.cq is None:
        return None
    x = raw.get_data() * 1e6
    x = x - x.mean(axis=0, keepdims=True)
    hf = sosfiltfilt(butter(4, [1, 40], btype="band", fs=FS, output="sos"), x, axis=1)
    ok = good_mask(raw)
    out = {}
    for name, frame in [("contact", run.cq), ("eeg_quality", run.eq)]:
        noise, quality = [], []
        for _, row in frame.iterrows():
            idx = np.flatnonzero((run.t >= row["time"] - 1.0) & (run.t < row["time"]))
            if len(idx) < 100 or not ok[idx].all():
                continue
            noise.append(np.log(np.mean(hf[:, idx] ** 2, axis=1)))
            quality.append([row[ch] for ch in run.labels])
        noise = np.array(noise)
        noise -= noise.mean(axis=1, keepdims=True)  # channel-specific part only
        quality = np.array(quality)
        rho = {}
        for j, ch in enumerate(run.labels):
            rho[ch] = float(stats.spearmanr(quality[:, j], noise[:, j])[0]) if quality[:, j].std() > 0 else None
        out[name] = {
            "spearman_quality_vs_own_noise": rho,
            "pct_below_green": {ch: float(100 * np.mean(frame[ch] < 4)) for ch in run.labels},
            "overall_median": float(frame["OVERALL" if name == "contact" else "overall"].median()),
        }
    return out


# ---------------------------------------------------------------------------
# Eyes closed / open


def blink_times(raw: mne.io.BaseRaw, channels: tuple[str, ...] = ("Fp1", "Fp2")) -> np.ndarray:
    x = raw.get_data(picks=list(channels)) * 1e6
    fp = sosfiltfilt(butter(4, [0.5, 15], btype="band", fs=FS, output="sos"), x.mean(axis=0))
    peaks, _ = find_peaks(fp, height=100, prominence=100, distance=int(0.3 * FS))
    return peaks / FS


def alpha_analysis(run: Run, raw: mne.io.BaseRaw, bads: list[str],
                   blink_channels: tuple[str, ...] = ("Fp1", "Fp2")) -> tuple[pd.DataFrame, dict]:
    """Per-interval spectra on 1-40 Hz average-referenced data; 2-s windows, artifact-free only."""

    r = raw.copy().filter(1.0, 40.0)
    r.info["bads"] = list(bads)
    r.set_eeg_reference("average")
    x = r.get_data() * 1e6
    ok = good_mask(r)
    good = [j for j, ch in enumerate(run.labels) if ch not in bads]
    blinks = blink_times(raw, blink_channels)
    n = int(2 * FS)
    rows, spectra = [], {}
    for iv in run.eye_intervals:
        a = int((iv["start"] + EYE_TRIM[0]) * FS)
        b = int((iv["end"] - EYE_TRIM[1]) * FS)
        segments, rejected = [], 0
        for s in range(a, b - n + 1, n // 2):
            seg = x[:, s:s + n]
            if not ok[s:s + n].all() or np.ptp(seg[good], axis=1).max() > 200:
                rejected += 1
                continue
            segments.append(seg)
        freqs, p = welch(np.stack(segments), fs=FS, nperseg=n, axis=2)
        psd = p.mean(axis=0)
        spectra[(iv["state"], iv["pair"])] = psd
        n_blinks = int(((blinks >= iv["start"]) & (blinks < iv["end"])).sum())
        rows.append({
            "run": run.key, "state": iv["state"], "pair": iv["pair"],
            "start_s": iv["start"], "duration_s": iv["end"] - iv["start"],
            "windows": len(segments), "rejected_windows": rejected,
            "blinks_per_min": 60 * n_blinks / (iv["end"] - iv["start"]),
        })
    return pd.DataFrame(rows), {"freqs": freqs, "spectra": spectra}


def alpha_scores(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Absolute 8-13 Hz power (dB) and power relative to the 5-7/14-17 Hz flanks (dB)."""

    band = (freqs >= 8) & (freqs <= 13)
    flank = ((freqs >= 5) & (freqs <= 7)) | ((freqs >= 14) & (freqs <= 17))
    hf = (freqs >= 20) & (freqs <= 40)
    return {
        "alpha_db": 10 * np.log10(psd[:, band].mean(axis=1)),
        "alpha_rel_db": 10 * np.log10(psd[:, band].mean(axis=1) / psd[:, flank].mean(axis=1)),
        "hf_db": 10 * np.log10(psd[:, hf].mean(axis=1)),
    }


# ---------------------------------------------------------------------------
# ERPs


FLEX_EVENT_CODES = {"passive/standard": 1, "passive/oddball": 2, "attended/standard": 3, "attended/oddball": 4}


def erp_epochs(run: Run, raw: mne.io.BaseRaw, bads: list[str], eog_channels: tuple[str, ...] = ("Fp1", "Fp2"),
               codes: dict[str, int] = FLEX_EVENT_CODES) -> tuple[mne.Epochs, dict]:
    """0.1-30 Hz, ICA blink removal, average reference, bad channels interpolated.

    ``run.tone_events`` needs ``sample``, ``block`` and ``stim`` columns; ``codes``
    maps each ``block/stim`` to an event id, fixed so recordings can be concatenated.
    """

    fit_raw = raw.copy().filter(1.0, 30.0)
    fit_raw.info["bads"] = list(bads)
    fit_raw.set_eeg_reference("average")
    n_good = len(raw.ch_names) - len(bads)
    ica = mne.preprocessing.ICA(n_components=min(20, n_good - 1), method="fastica", random_state=SEED, max_iter=2000)
    ica.fit(fit_raw, reject_by_annotation=True)
    eog_idx, scores = ica.find_bads_eog(fit_raw, ch_name=list(eog_channels), threshold=3.0)
    # z-scoring over 20 components can also pick weakly related ones; keep blink-like ones only.
    scores = np.atleast_2d(scores)
    eog_idx = [i for i in eog_idx if np.abs(scores[:, i]).max() >= 0.5]

    erp_raw = raw.copy().filter(*ERP_BAND)
    erp_raw.info["bads"] = list(bads)
    erp_raw.set_eeg_reference("average")
    ica.apply(erp_raw, exclude=eog_idx)
    if bads:
        erp_raw.interpolate_bads(reset_bads=True)

    ev = run.tone_events
    events = np.column_stack([ev["sample"], np.zeros(len(ev), int),
                              [codes[f"{b}/{s}"] for b, s in zip(ev["block"], ev["stim"])]])
    present = {k: v for k, v in codes.items() if v in events[:, 2]}
    epochs = mne.Epochs(erp_raw, events, present, tmin=ERP_WINDOW[0], tmax=ERP_WINDOW[1],
                        baseline=(ERP_WINDOW[0], 0), reject=dict(eeg=REJECT_UV * 1e-6),
                        reject_by_annotation=True, preload=True, metadata=ev.assign(run=run.key))
    drops = {"fill": 0, "amplitude": 0}
    for log in epochs.drop_log:
        if "BAD_fill" in log:
            drops["fill"] += 1
        elif log:
            drops["amplitude"] += 1
    ica_info = {"excluded_components": [int(i) for i in eog_idx],
                "eog_scores_abs_max": [float(np.abs(scores[:, i]).max()) for i in eog_idx],
                "drops": drops}
    return epochs, ica_info


def roi_trials(epochs: mne.Epochs, roi: list[str]) -> np.ndarray:
    return epochs.get_data(picks=roi).mean(axis=1) * 1e6


def peak_latency(times: np.ndarray, wave: np.ndarray, window: tuple[float, float], sign: int) -> float:
    sel = (times >= window[0]) & (times <= window[1])
    return float(times[sel][np.argmax(sign * wave[sel])])


def bootstrap_latency(times, trials, window, sign, n_boot=2000, rng=None) -> list[float]:
    rng = rng or np.random.default_rng(SEED)
    lat = [peak_latency(times, trials[rng.integers(0, len(trials), len(trials))].mean(axis=0), window, sign)
           for _ in range(n_boot)]
    return [float(v) for v in np.percentile(lat, [2.5, 97.5])]


def cluster_test(a: mne.Epochs, b: mne.Epochs, tmin: float = 0.0, tmax: float = ERP_WINDOW[1], n_perm: int = 2000) -> dict:
    """Spatio-temporal cluster test, oddball vs standard trials (Welch t)."""

    adjacency, _ = mne.channels.find_ch_adjacency(a.info, "eeg")
    sel = (a.times >= tmin) & (a.times <= tmax)
    xa = a.get_data()[:, :, sel].transpose(0, 2, 1) * 1e6
    xb = b.get_data()[:, :, sel].transpose(0, 2, 1) * 1e6
    df = len(xa) + len(xb) - 2
    threshold = stats.t.ppf(1 - 0.01 / 2, df)
    stat_fun = lambda x, y: mne.stats.ttest_ind_no_p(x, y, equal_var=False)
    t_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_test(
        [xa, xb], adjacency=adjacency, threshold=threshold, n_permutations=n_perm, tail=0,
        stat_fun=stat_fun, seed=SEED, out_type="mask", buffer_size=None)
    times = a.times[sel]
    found = []
    for mask, p in zip(clusters, p_values):
        tt, cc = np.nonzero(mask)
        found.append({
            "p": float(p),
            "sign": "positive" if t_obs[mask].sum() > 0 else "negative",
            "t_start": float(times[tt.min()]), "t_end": float(times[tt.max()]),
            "channels": sorted({a.ch_names[c] for c in cc}),
            "mass": float(np.abs(t_obs[mask]).sum()),
        })
    found.sort(key=lambda c: c["p"])
    return {"n_oddball": len(xa), "n_standard": len(xb), "t_obs": t_obs, "times": times,
            "clusters": found, "masks": clusters, "p_values": p_values}


# ---------------------------------------------------------------------------
# Figures


def fig_integrity(runs, integ, raws, out: Path) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(14, 3.4 * len(runs)), constrained_layout=True)
    for ax, run in zip(np.atleast_1d(axes), runs):
        res = integ[run.key]["_residual_ms"]
        ax.plot(run.t, np.clip(res, -60, 60), lw=0.6, color="tab:blue", label="timestamp − linear fit (ms, clipped ±60)")
        starts, stops = contiguous(run.filled)
        for a, b in zip(starts, stops):
            ax.axvspan(run.t[a], run.t[min(b, len(run.t) - 1)] + 0.3, color="tab:red", alpha=0.35, lw=0)
        for s in np.flatnonzero(run.diag["RESET_FLAG"] > 0):
            ax.axvline(run.t[s], color="tab:red", lw=1.2)
        for iv in run.eye_intervals:
            ax.axvspan(iv["start"], iv["end"], ymin=0.92, ymax=1.0,
                       color="navy" if iv["state"] == "closed" else "gold", lw=0)
        for prefix, ev in run.tone_events.groupby("prefix"):
            ax.axvspan(ev["time"].min(), ev["time"].max(), ymin=0.0, ymax=0.08,
                       color="tab:green" if BLOCKS[prefix] == "passive" else "tab:purple", lw=0)
            ax.text(ev["time"].mean(), -55, f"{prefix} ({BLOCKS[prefix]})", ha="center", fontsize=8)
        ax.set_ylim(-62, 62)
        ax.set_xlim(0, run.t[-1])
        ax.set_ylabel("ms")
        i = integ[run.key]
        ax.set_title(f"{run.key}: {i['filled_pct']:.2f}% samples filled ({i['filled_runs']} runs, red), "
                     f"{i['counter_resets']} counter resets (red lines), {i['dropped_repeats']} dropped repeats; "
                     f"top bar navy=eyes closed, gold=open; bottom bar = oddball blocks", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("seconds from first EEG sample")
    fig.savefig(out / "01_integrity_timeline.png", dpi=130)
    plt.close(fig)


def fig_fill_example(run: Run, out: Path) -> None:
    """As-published signal around an isolated 8-sample loss during eyes closed."""

    starts, stops = contiguous(run.filled)
    closed = [(iv["start"], iv["end"]) for iv in run.eye_intervals if iv["state"] == "closed"]
    pick = None
    for i, (a, b) in enumerate(zip(starts, stops)):
        isolated = (i == 0 or a - stops[i - 1] > 4 * FS) and (i == len(starts) - 1 or starts[i + 1] - b > 4 * FS)
        if b - a == 8 and isolated and any(lo + 4 < run.t[a] < hi - 4 for lo, hi in closed):
            pick = (a, b)
            break
    if pick is None:
        return
    a, b = pick
    pre = int(1.5 * FS)
    span = slice(a - pre, b + pre)
    channels = ["Fz", "Cz", "Pz", "P4", "Oz", "F7", "F8", "T7", "T8"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    t = run.t[span] - run.t[a]
    # Average reference removes the large shared slow wave so the per-channel step is visible;
    # the lost deltas differ per channel, so the step survives re-referencing.
    x = run.x_uv[span] - run.x_uv[span].mean(axis=1, keepdims=True)
    spacing = 40
    for k, ch in enumerate(channels):
        y = x[:, run.labels.index(ch)]
        y = y - y[:pre].mean() - spacing * k
        ax.plot(t, y, lw=0.9, color="k")
        ax.text(t[0] - 0.03, y[0], ch, ha="right", va="center", fontsize=8)
    ax.axvspan(0, (b - a) / FS, color="tab:red", alpha=0.3, label="8 filled samples (zero delta)")
    ax.axvspan((b - a) / FS, FILL_PAD[1] + (b - a) / FS, color="tab:red", alpha=0.08, label="padding marked bad")
    ax.set_yticks([])
    ax.set_xlabel("s from start of the loss")
    ax.set_title(f"{run.key}: an isolated 8-sample loss at {run.t[a]:.1f} s (eyes closed), average reference, "
                 f"{spacing} µV between traces; each channel keeps an offset"
                 + (f" that fades with the reader's {run.dc_restore_hz:g} Hz leak" if run.dc_restore_hz else ""),
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "01b_fill_example.png", dpi=130)
    plt.close(fig)


def fig_spectra(spec, noise, epocx, out: Path) -> None:
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    colors = {"run1": "tab:blue", "run2": "tab:orange"}
    for key, s in spec.items():
        ax.semilogy(s["freqs"], np.median(s["as_recorded"], axis=0), color=colors[key], lw=1.2, ls="--",
                    label=f"Flex {key}: as recorded (CMS at TP9)")
        ax.semilogy(s["freqs"], np.median(s["avg_ref"], axis=0), color=colors[key], lw=1.8,
                    label=f"Flex {key}: average reference")
    if epocx is not None:
        ax.semilogy(epocx["freqs"], np.median(epocx["avg_ref"], axis=0), color="0.4", lw=1.2,
                    label="EPOC X pilot: average reference (14 ch)")
    ax.axvline(50, color="k", ls=":", lw=0.8)
    ax.text(50.5, ax.get_ylim()[1] * 0.3 if ax.get_ylim()[1] > 0 else 1, "50 Hz", fontsize=8)
    ax.set_xlim(0, 64)
    ax.set_xlabel("Hz")
    ax.set_ylabel("µV²/Hz (median channel)")
    ax.set_title("Whole-record spectra, clean spans only")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    key = "run1"
    s = spec[key]
    sel = (s["hires_freqs"] > 44) & (s["hires_freqs"] < 56)
    ax.semilogy(s["hires_freqs"][sel], np.median(s["hires_avg"], axis=0)[sel], color="tab:blue", lw=0.8)
    ax.axvline(50, color="k", ls=":", lw=0.8)
    ax.set_title("run1, 44–56 Hz at 0.016 Hz resolution\n(no 50 Hz line)")
    ax.set_xlabel("Hz")

    for col, key in enumerate(noise):
        ax = fig.add_subplot(gs[1, col])
        frame = noise[key].sort_values("hf_20_40_uv2")
        ax.barh(frame["channel"], frame["hf_20_40_uv2"], color=np.where(frame["bad"], "tab:red", "tab:gray"))
        ax.set_xscale("log")
        ax.set_xlabel("20–40 Hz power (µV², median ref, clean spans)")
        ax.set_title(f"{key}: per-channel noise (red = flagged bad)")
        ax.tick_params(axis="y", labelsize=7)
    ax = fig.add_subplot(gs[1, 2])
    for key in noise:
        sat = pd.Series(spec[key]["saturation"]).reindex(noise[key]["channel"])
        ax.scatter(noise[key]["hf_20_40_uv2"], sat.values, label=key, s=18)
        for ch, xv, yv in zip(noise[key]["channel"], noise[key]["hf_20_40_uv2"], sat.values):
            if yv > 1.0:
                ax.annotate(ch, (xv, yv), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("20–40 Hz power (µV²)")
    ax.set_ylabel("% deltas at the ±64-count slew limit")
    ax.set_title("Slew-limit saturation vs channel noise")
    ax.legend(fontsize=8)
    fig.savefig(out / "02_spectra_noise.png", dpi=130)
    plt.close(fig)


def fig_alpha(alpha_frames, alpha_spec, scores, labels, bads_by_run, info, out: Path) -> None:
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    for col, key in enumerate(alpha_spec):
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        freqs = alpha_spec[key]["freqs"]
        roi = [labels.index(ch) for ch in POSTERIOR_ROI if ch not in bads_by_run[key]]
        for (state, pair), psd in alpha_spec[key]["spectra"].items():
            ax.semilogy(freqs, psd[roi].mean(axis=0), color="tab:blue" if state == "closed" else "tab:orange",
                        lw=0.7, alpha=0.5)
        for state, color in [("closed", "tab:blue"), ("open", "tab:orange")]:
            mean = np.mean([psd[roi].mean(axis=0) for (s, _), psd in alpha_spec[key]["spectra"].items() if s == state], axis=0)
            ax.semilogy(freqs, mean, color=color, lw=2.2, label=f"eyes {state} (mean of 4 intervals)")
        ax.axvspan(8, 13, color="0.9", zorder=0)
        ax.set_xlim(1, 40)
        in_band = (freqs >= 1) & (freqs <= 40)
        values = np.concatenate([psd[roi].mean(axis=0)[in_band] for psd in alpha_spec[key]["spectra"].values()])
        ax.set_ylim(values.min() * 0.7, values.max() * 1.4)
        ax.set_xlabel("Hz")
        ax.set_ylabel("µV²/Hz")
        roi_names = [ch for ch in POSTERIOR_ROI if ch not in bads_by_run[key]]
        ax.set_title(f"{key}: posterior ROI ({', '.join(roi_names)}), avg ref", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    frame = pd.concat(alpha_frames.values())
    for metric, offset, marker in [("roi_alpha_rel_db", -0.1, "o"), ("roi_alpha_db", 0.1, "s")]:
        pivot = frame.pivot_table(index=["run", "pair"], columns="state", values=metric)
        diff = pivot["closed"] - pivot["open"]
        ax.scatter(np.full(len(diff), 0 if metric == "roi_alpha_rel_db" else 1) + np.linspace(-0.15, 0.15, len(diff)),
                   diff, marker=marker, c=["tab:blue" if r == "run1" else "tab:orange" for r, _ in diff.index])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks([0, 1], ["alpha vs flanks", "absolute alpha"])
    ax.set_ylabel("closed − open (dB), posterior ROI")
    ax.set_title("Per closed/open pair\n(blue = run1, orange = run2)", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    pivot = frame.pivot_table(index=["run", "pair"], columns="state", values="blinks_per_min")
    for (r, p), row in pivot.iterrows():
        ax.plot([0, 1], [row["closed"], row["open"]], color="tab:blue" if r == "run1" else "tab:orange", marker="o")
    ax.set_xticks([0, 1], ["closed", "open"])
    ax.set_ylabel("Fp blinks / min")
    ax.set_title("Marker check: blinks by labelled state", fontsize=9)

    for col, (metric, title) in enumerate([("alpha_rel_db", "closed − open: alpha vs flanks (dB)"),
                                           ("hf_db", "closed − open: 20–40 Hz power (dB)")]):
        ax = fig.add_subplot(gs[1, 2 + col])
        values = scores[metric]
        keep = np.flatnonzero(np.isfinite(values))
        lim = max(np.abs(values[keep]).max(), 0.5)
        im, _ = mne.viz.plot_topomap(values[keep], mne.pick_info(info, keep), axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-lim, lim), names=[labels[k] for k in keep], sensors=True)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{title}\nboth runs, 8 pairs", fontsize=9)
    fig.savefig(out / "03_alpha_eyes.png", dpi=130)
    plt.close(fig)


def fig_standards(epochs_by_block, pooled_std, latency, split_half, out: Path) -> None:
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    times = pooled_std.times
    for col, (roi, title) in enumerate([(FC_ROI, "Fz/FC1/FC2/Cz"), (INFERIOR_ROI, "FT9/FT10/PO9/PO10")]):
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        for name, ep in epochs_by_block.items():
            y = roi_trials(ep, roi)
            ax.plot(times, y.mean(axis=0), lw=1.2, label=f"{name} (n={len(y)})")
        y = roi_trials(pooled_std, roi)
        se = y.std(axis=0, ddof=1) / np.sqrt(len(y))
        ax.plot(times, y.mean(axis=0), color="k", lw=2.2, label=f"pooled (n={len(y)}) ± SE")
        ax.fill_between(times, y.mean(axis=0) - se, y.mean(axis=0) + se, color="k", alpha=0.2)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(0, 0.1, color="0.85", zorder=0)
        ax.axvline(0.1, color="tab:red", ls=":", lw=1)
        ax.set_title(f"Standards, {title} (grey = 100 ms tone; red dotted = textbook N1 latency)", fontsize=9)
        ax.set_xlabel("s from tone marker")
        ax.set_ylabel("µV")
        ax.legend(fontsize=7)
    evoked = pooled_std.average()
    for col, (name, lat) in enumerate([("N1-like trough", latency["n1_latency_s"]), ("P2-like peak", latency["p2_latency_s"])]):
        ax = fig.add_subplot(gs[1, col])
        evoked.plot_topomap(times=[lat], axes=ax, show=False, colorbar=False, average=0.03, sensors=True)
        ax.set_title(f"{name} at {1000 * lat:.0f} ms", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(times, split_half["odd"], label="odd trials")
    ax.plot(times, split_half["even"], label="even trials")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Split half, Fz/FC1/FC2/Cz: r = {split_half['r']:.2f} (0–600 ms)", fontsize=9)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 3])
    y = roi_trials(pooled_std, FC_ROI)
    order = np.argsort(pooled_std.metadata["time"].to_numpy() + 1e5 * (pooled_std.metadata["run"] == "run2").to_numpy())
    smooth = np.apply_along_axis(lambda v: np.convolve(v, np.ones(15) / 15, mode="same"), 0, y[order])
    ax.imshow(smooth, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, extent=[times[0], times[-1], len(y), 0])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title("Trial image (15-trial moving average)", fontsize=9)
    ax.set_xlabel("s")
    ax.set_ylabel("trial (recording order)")
    fig.savefig(out / "04_erp_standards.png", dpi=130)
    plt.close(fig)


def fig_oddball(conditions, tests, out: Path) -> None:
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    for row, (name, (odd, std)) in enumerate(conditions.items()):
        times = odd.times
        for col, ch in enumerate(["Fz", "Cz", "Pz"]):
            ax = fig.add_subplot(gs[row, col])
            for ep, label, color in [(std, "standard", "0.4"), (odd, "oddball", "tab:red")]:
                y = ep.get_data(picks=[ch])[:, 0] * 1e6
                se = y.std(axis=0, ddof=1) / np.sqrt(len(y))
                ax.plot(times, y.mean(axis=0), color=color, lw=1.5, label=f"{label} (n={len(y)})")
                ax.fill_between(times, y.mean(axis=0) - se, y.mean(axis=0) + se, color=color, alpha=0.2)
            test = tests[name]
            j = odd.ch_names.index(ch)
            for mask, p in zip(test["masks"], test["p_values"]):
                if p < 0.05:
                    on = mask[:, j]
                    if on.any():
                        ax.axvspan(test["times"][on].min(), test["times"][on].max(), color="gold", alpha=0.3, zorder=0)
            ax.axvline(0, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{name}: {ch}", fontsize=9)
            if col == 0:
                ax.legend(fontsize=7)
                ax.set_ylabel("µV")
        diff = mne.combine_evoked([odd.average(), std.average()], weights=[1, -1])
        shown = [c for c in tests[name]["clusters"] if c["p"] < 0.05][:1]
        if not shown:
            best = tests[name]["clusters"][0] if tests[name]["clusters"] else None
            shown = [best] if best else []
        for k, cluster in enumerate(shown):
            ax = fig.add_subplot(gs[row, 3 + k])
            centre = (cluster["t_start"] + cluster["t_end"]) / 2
            width = max(cluster["t_end"] - cluster["t_start"], 1 / FS)
            diff.plot_topomap(times=[centre], average=width, axes=ax, show=False, colorbar=False, sensors=True)
            ax.set_title(f"odd − std, {1000 * cluster['t_start']:.0f}–{1000 * cluster['t_end']:.0f} ms\n"
                         f"{cluster['sign']} cluster p = {cluster['p']:.3f}", fontsize=8)
    fig.suptitle("Oddball vs standard, time from tone marker (gold = significant spatio-temporal cluster at this channel, p < .05; "
                 "topomaps show significant clusters, or the strongest if none)", fontsize=11)
    fig.savefig(out / "05_erp_oddball.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/Oddball/data/flex_sanity"))
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = [load_run(key, path) for key, path in RUNS.items()]
    labels = runs[0].labels
    results: dict = {"runs": {}}

    integ, raws, noise, spec = {}, {}, {}, {}
    for run in runs:
        print(f"[{run.key}] integrity and noise")
        integ[run.key] = integrity(run)
        raws[run.key] = raw = make_raw(run)
        noise[run.key] = channel_noise(raw)
        ok = good_mask(raw)
        x = raw.get_data() * 1e6
        x_hp = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), x, axis=1)
        bads = noise[run.key].loc[noise[run.key]["bad"], "channel"].tolist()
        good = [j for j, ch in enumerate(labels) if ch not in bads]
        avg = x_hp - x_hp[good].mean(axis=0, keepdims=True)
        freqs, p_rec = clean_psd(x_hp, ok)
        _, p_avg = clean_psd(avg, ok)
        hi_f, p_hi = clean_psd(avg, ok, nperseg=8192)
        spec[run.key] = {"freqs": freqs, "as_recorded": p_rec, "avg_ref": p_avg, "hires_freqs": hi_f, "hires_avg": p_hi,
                         "saturation": integ[run.key]["slew_saturated_pct_by_channel"]}
        band = (freqs >= 20) & (freqs <= 40)
        results["runs"][run.key] = {
            "integrity": integ[run.key],
            "bad_channels": bads,
            "channel_noise": noise[run.key],
            "psd_20_40_median_channel": {"as_recorded": float(np.median(p_rec[:, band])),
                                         "avg_ref": float(np.median(p_avg[:, band]))},
            "line_noise": line_check(hi_f, np.median(p_hi, axis=0)),
            "quality_streams": quality_vs_noise(run, raw),
        }

    epocx = None
    if Path(EPOCX_PILOT).exists():
        streams, _ = pyxdf.load_xdf(EPOCX_PILOT, synchronize_clocks=True, dejitter_timestamps=False)
        eeg = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
        names = _labels(eeg)
        sensors = [i for i, ch in enumerate(names) if ch in
                   ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4")]
        ex = np.asarray(eeg["time_series"], float)[:, sensors].T
        ex = sosfiltfilt(butter(2, 0.5, btype="high", fs=FS, output="sos"), ex, axis=1)
        ex -= ex.mean(axis=0, keepdims=True)
        f_e, p_e = welch(ex, fs=FS, nperseg=512, axis=1)
        epocx = {"freqs": f_e, "avg_ref": p_e}
        results["epocx_pilot_psd_20_40_avg_ref"] = float(np.median(p_e[:, (f_e >= 20) & (f_e <= 40)]))

    fig_integrity(runs, integ, raws, out)
    fig_fill_example(runs[0], out)
    fig_spectra(spec, noise, epocx, out)

    # Eyes closed / open.
    print("alpha")
    alpha_frames, alpha_spec, diffs = {}, {}, {m: [] for m in ("alpha_rel_db", "alpha_db", "hf_db")}
    for run in runs:
        bads = results["runs"][run.key]["bad_channels"]
        frame, sp = alpha_analysis(run, raws[run.key], bads)
        roi = [labels.index(ch) for ch in POSTERIOR_ROI if ch not in bads]
        for metric in ("alpha_rel_db", "alpha_db", "hf_db"):
            frame[f"roi_{metric}"] = [alpha_scores(sp["freqs"], sp["spectra"][(s, p)])[metric][roi].mean()
                                      for s, p in zip(frame["state"], frame["pair"])]
        for pair in range(1, 5):
            c = alpha_scores(sp["freqs"], sp["spectra"][("closed", pair)])
            o = alpha_scores(sp["freqs"], sp["spectra"][("open", pair)])
            for metric in diffs:
                d = c[metric] - o[metric]
                d[[labels.index(ch) for ch in bads]] = np.nan
                diffs[metric].append(d)
        roi_psd = {state: np.mean([psd[roi].mean(axis=0) for (s, _), psd in sp["spectra"].items() if s == state], axis=0)
                   for state in ("closed", "open")}
        alpha_sel = (sp["freqs"] >= 7) & (sp["freqs"] <= 14)
        frame_peak = float(sp["freqs"][alpha_sel][np.argmax(roi_psd["closed"][alpha_sel] / roi_psd["open"][alpha_sel])])
        alpha_frames[run.key], alpha_spec[run.key] = frame, sp
        results["runs"][run.key]["alpha_intervals"] = frame
        results["runs"][run.key]["alpha_closed_over_open_peak_hz"] = frame_peak
    pairs = pd.concat(alpha_frames.values())
    alpha_stats = {}
    for metric in ("roi_alpha_rel_db", "roi_alpha_db", "roi_hf_db", "blinks_per_min"):
        pivot = pairs.pivot_table(index=["run", "pair"], columns="state", values=metric)
        d = (pivot["closed"] - pivot["open"]).to_numpy()
        alpha_stats[metric] = {
            "closed_minus_open": d.tolist(),
            "mean": float(d.mean()),
            "ci95": [float(v) for v in stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))],
            "pairs_closed_greater": int((d > 0).sum()),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
        }
    results["alpha"] = alpha_stats
    # A channel counts only if it was good in both runs.
    topo = {m: np.mean(np.array(v), axis=0) for m, v in diffs.items()}
    results["alpha"]["topography_closed_minus_open"] = {
        m: {ch: (None if not np.isfinite(val) else round(float(val), 2)) for ch, val in zip(labels, v)} for m, v in topo.items()}
    info = raws["run1"].copy().pick(labels).info
    fig_alpha(alpha_frames, alpha_spec, topo, labels,
              {k: results["runs"][k]["bad_channels"] for k in RUNS}, info, out)

    # ERPs.
    all_epochs = []
    for run in runs:
        print(f"[{run.key}] ERP epochs")
        bads = results["runs"][run.key]["bad_channels"]
        epochs, ica_info = erp_epochs(run, raws[run.key], bads)
        results["runs"][run.key]["erp_preprocessing"] = ica_info
        all_epochs.append(epochs)
    epochs = mne.concatenate_epochs(all_epochs, add_offset=True)
    meta = epochs.metadata
    counts = {}
    for (r, prefix, stim), n in meta.groupby(["run", "prefix", "stim"]).size().items():
        total = int(((run_ev := next(x for x in runs if x.key == r).tone_events)["prefix"] == prefix).mul(run_ev["stim"] == stim).sum())
        counts[f"{r}/{prefix}/{stim}"] = {"kept": int(n), "total": total}
    results["erp_trial_counts"] = counts

    blocks = {f"{r} {p}": epochs[(meta["run"] == r).to_numpy() & (meta["prefix"] == p).to_numpy() & (meta["stim"] == "standard").to_numpy()]
              for r, p in [("run1", "Tone"), ("run1", "AttendedTone"), ("run2", "Oddball")]}
    pooled_std = epochs[(meta["stim"] == "standard").to_numpy()]
    times = pooled_std.times
    fc = roi_trials(pooled_std, FC_ROI)
    grand = fc.mean(axis=0)
    rng = np.random.default_rng(SEED)
    n1_window, p2_window = (0.05, 0.35), (0.2, 0.5)
    n1 = peak_latency(times, grand, n1_window, -1)
    p2 = peak_latency(times, grand, (n1 + 0.03, p2_window[1]), +1)
    latency = {
        "n1_latency_s": n1, "n1_ci95": bootstrap_latency(times, fc, n1_window, -1, rng=rng),
        "p2_latency_s": p2, "p2_ci95": bootstrap_latency(times, fc, (n1 + 0.03, p2_window[1]), +1, rng=rng),
        "n1_amp_uv": float(grand[times == n1][0]), "p2_amp_uv": float(grand[times == p2][0]),
        "implied_offset_vs_100ms_n1_s": n1 - 0.1,
    }
    per_block = {}
    for name, ep in blocks.items():
        y = roi_trials(ep, FC_ROI).mean(axis=0)
        per_block[name] = {"n1_latency_s": peak_latency(times, y, n1_window, -1),
                           "p2_latency_s": peak_latency(times, y, p2_window, +1)}
        # lag that best aligns this block's average to the pooled average (0-600 ms)
        sel = (times >= -0.1) & (times <= 0.6)
        lags = np.arange(-8, 9)
        xc = [np.corrcoef(np.roll(y, -k)[sel], grand[sel])[0, 1] for k in lags]
        per_block[name]["lag_vs_pooled_ms"] = float(1000 * lags[int(np.argmax(xc))] / FS)
        per_block[name]["corr_with_pooled"] = float(max(xc))
    latency["per_block"] = per_block
    # Deflection from baseline: trial-level cluster test at the frontocentral ROI.
    t_obs, clusters, p_vals, _ = mne.stats.permutation_cluster_1samp_test(
        fc[:, times >= 0], n_permutations=args.permutations, seed=SEED, tail=0)
    latency["baseline_clusters"] = [
        {"t_start": float(times[times >= 0][c[0]].min()), "t_end": float(times[times >= 0][c[0]].max()),
         "sign": "positive" if t_obs[c[0]].sum() > 0 else "negative", "p": float(p)}
        for c, p in zip(clusters, p_vals) if p < 0.05]
    sel = (times >= 0) & (times <= 0.6)
    odd_avg, even_avg = fc[0::2].mean(axis=0), fc[1::2].mean(axis=0)
    r = float(np.corrcoef(odd_avg[sel], even_avg[sel])[0, 1])
    split_half = {"odd": odd_avg, "even": even_avg, "r": r, "spearman_brown": 2 * r / (1 + r)}
    inferior = roi_trials(pooled_std, INFERIOR_ROI).mean(axis=0)
    latency["inferior_roi_at_n1_uv"] = float(inferior[times == n1][0])
    latency["inferior_roi_at_p2_uv"] = float(inferior[times == p2][0])
    results["standards"] = {**latency, "split_half_r": r, "split_half_spearman_brown": split_half["spearman_brown"]}
    fig_standards(blocks, pooled_std, latency, split_half, out)

    print("oddball cluster tests")
    passive = (meta["block"] == "passive").to_numpy()
    attended = (meta["block"] == "attended").to_numpy()
    odd = (meta["stim"] == "oddball").to_numpy()
    conditions = {
        "passive (run1 Tone + run2)": (epochs[passive & odd], epochs[passive & ~odd]),
        "attended (run1 AttendedTone)": (epochs[attended & odd], epochs[attended & ~odd]),
        "all blocks pooled": (epochs[odd], epochs[~odd]),
    }
    tests = {name: cluster_test(o, s, n_perm=args.permutations) for name, (o, s) in conditions.items()}
    results["oddball_tests"] = {name: {k: v for k, v in t.items() if k in ("n_oddball", "n_standard", "clusters")}
                                for name, t in tests.items()}
    for name, (o, s) in conditions.items():
        diff = o.average().data - s.average().data
        for ch in ["Fz", "Cz", "Pz"]:
            j = o.ch_names.index(ch)
            results["oddball_tests"][name].setdefault("difference_peaks", {})[ch] = {
                "min_uv": float(diff[j].min() * 1e6), "min_s": float(o.times[np.argmin(diff[j])]),
                "max_uv": float(diff[j].max() * 1e6), "max_s": float(o.times[np.argmax(diff[j])]),
            }
    fig_oddball(conditions, tests, out)

    (out / "results.json").write_text(json.dumps(to_jsonable(results), indent=2), encoding="utf-8")
    for key in RUNS:
        noise[key].to_csv(out / f"channel_noise_{key}.csv", index=False)
    pairs.to_csv(out / "alpha_intervals.csv", index=False)
    print(out / "results.json")


if __name__ == "__main__":
    main()
