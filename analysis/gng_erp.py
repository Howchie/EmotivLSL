#!/usr/bin/env python3
"""Go/NoGo ERPs from the EPOC X 80/20 recording ``data/GnG/gng.xdf``.

Reference implementation of ``analysis/EPOCX_ERP_PIPELINE.md``: the functions
under "Loading" and "Pipeline" are meant to be reused by later EPOC X ERP
analyses.  ERPs are in the recorded (CMS) reference only; the average
reference cancels broad components on this montage (see the pipeline notes).

Outputs go to ``data/GnG/gng_erp``:

* ``qc_artifacts.png``  artifact timeline, removed ICA components, blink and
                        saccade checks, baseline vs noise, electrode pops
* ``erp.png``           ROI ERPs with bootstrap bands, Go - NoGo butterfly with
                        cluster test, N1 timing check, block replication,
                        topographies
* ``results.json``, ``events.csv``, ``bad_segments.csv``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epocx_sanity as ex  # noqa: E402
import flex_sanity as fx  # noqa: E402

mne.set_log_level("ERROR")

XDF = "data/GnG/gng.xdf"
CSV = "data/GnG/gng.csv"
LOOPBACK = "data/DRT_Timing/loopback_timing/results.json"
AUDIO = "data/Oddball/data/audio_latency/results.json"
OUT = Path("data/GnG/gng_erp")
FS = 128.0
SEED = 7
FRAME_S = 1.0 / 60.146  # frameRate in the CSV

# --- Pipeline parameters (see EPOCX_ERP_PIPELINE.md) -------------------------
ICA_BAND = (1.0, 30.0)
ERP_BAND = (0.1, 20.0)
GROSS_UV = 300.0  # 1-30 Hz p2p in a 1-s window: excluded from the ICA fit
RESIDUAL_UV = 100.0  # 0.1-20 Hz p2p in a 1-s window after ICA: excluded from ERPs
WIN_S, STEP_S = 1.0, 0.5
PAD_S = (0.25, 0.5)  # before/after a flagged window
EOG_R = 0.5  # |r| between an IC and the 1-10 Hz VEOG/HEOG proxy
CHANNEL_BAD_SHARE = 0.10  # a channel exceeding RESIDUAL_UV in >10% of task windows is bad
POP_UV = 60.0  # channel-specific deviation (1-20 Hz, channel minus median of the others)
EPOCH = (-0.3, 1.0)
BASELINE = (-0.2, 0.0)
N_BOOT = 2000

# Channels dropped by hand for this recording, with the reason.  They stay in
# the QC diagnostics but are excluded from ICA, rejection and ROIs.
MANUAL_BADS = {"F4": "old saline pad: recurring electrode pops (sharp spike, then a ~50 µV offset recovering over ~2 s)"}

FRONTAL = ["F3", "F4", "FC5", "FC6"]
INFERIOR = ["T7", "T8", "P7", "P8"]
POSTERIOR = ["P7", "P8", "O1", "O2"]
TEMPORAL = ["T7", "T8"]
ROIS = {"frontal": FRONTAL, "posterior": POSTERIOR, "temporal": TEMPORAL}
GOOD: list[str] = []  # good channels, set in main(); ROI titles list only these
COLORS = {"go": "#b2182b", "nogo": "#2166ac", "diff": "black", "resp": "#1b7837"}


# =============================================================================
# Loading
# =============================================================================


def load_recording(xdf_path: str, csv_path: str) -> fx.Run:
    """EPOC X run on the counter-rebuilt sample grid, with Go/NoGo events and CSV bookkeeping."""

    streams, _ = pyxdf.load_xdf(xdf_path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    grid = ex.epocx_grid(by_name("Epoc X")[0], by_name("Epoc X Packet Diagnostics")[0])
    t0 = grid["t"][0]

    # LabRecorder can pick the same marker outlet up twice; keep one copy of each marker.
    rows = set()
    for stream in by_name("PsychoPy Markers"):
        for ts, value in zip(stream["time_stamps"], stream["time_series"]):
            rows.add((round(float(ts) - t0, 4), str(value[0])))
    markers = pd.DataFrame(sorted(rows), columns=["time", "value"])

    diag = {"FILLED": grid["missing"].astype(float), "RESET_FLAG": np.zeros(len(grid["t"]))}
    run = fx.Run("gng", grid["t"] - t0, grid["x"], fx._labels(by_name("Epoc X")[0]), diag, markers, None, None)
    csv = pd.read_csv(csv_path)
    run.tone_events = ex.epocx_events(run, csv)
    task = csv[csv["stim"].isin(["high", "low"])].reset_index(drop=True)
    if len(task) != len(run.tone_events):
        raise RuntimeError(f"CSV task rows ({len(task)}) != GnG markers ({len(run.tone_events)})")
    run.tone_events["block_idx"] = task["blocks.thisN"].astype(int).to_numpy()
    run.tone_events["freq_hz"] = task["freq"].astype(float).to_numpy()
    return run


# =============================================================================
# Pipeline
# =============================================================================


def sliding_p2p(x: np.ndarray, win_s: float = WIN_S, step_s: float = STEP_S) -> tuple[np.ndarray, np.ndarray]:
    """Peak-to-peak (channels x windows) of ``x`` in sliding windows; returns window starts too."""

    n, step = int(round(win_s * FS)), int(round(step_s * FS))
    view = np.lib.stride_tricks.sliding_window_view(x, n, axis=1)[:, ::step]
    return np.ptp(view, axis=2), np.arange(view.shape[1]) * step


def flag_windows(p2p: np.ndarray, starts: np.ndarray, threshold: float, n_times: int,
                 pad_s: tuple[float, float] = PAD_S, channels: list[int] | None = None) -> np.ndarray:
    """Boolean sample mask of windows whose p2p exceeds ``threshold`` in any (selected) channel."""

    sel = p2p if channels is None else p2p[channels]
    n = int(round(WIN_S * FS))
    bad = np.zeros(n_times, bool)
    for s in starts[(sel > threshold).any(axis=0)]:
        bad[max(0, s - int(pad_s[0] * FS)):min(n_times, s + n + int(pad_s[1] * FS))] = True
    return bad


def mask_to_annotations(mask: np.ndarray, label: str) -> mne.Annotations:
    starts, stops = fx.contiguous(mask)
    return mne.Annotations(starts / FS, (stops - starts) / FS, [label] * len(starts))


def eog_proxies(x_uv: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    """VEOG = mean(AF3, AF4) (no infra-orbital sensor) and HEOG = F7 - F8, in µV."""

    i = labels.index
    return {"VEOG": (x_uv[i("AF3")] + x_uv[i("AF4")]) / 2, "HEOG": x_uv[i("F7")] - x_uv[i("F8")]}


def preprocess(run: fx.Run, task_mask: np.ndarray) -> tuple[mne.io.RawArray, dict]:
    """CMS-referenced, ICA-cleaned 0.1-20 Hz data plus BAD annotations and QC numbers.

    1. bad channels: MANUAL_BADS plus data-driven (flat / 20-40 Hz noise, from
       flex_sanity); dropped, not interpolated, on a 14-channel montage
    2. gross-artifact windows (>GROSS_UV, 1-30 Hz) excluded from the ICA fit
    3. FastICA on 1-30 Hz CMS-referenced good channels; components whose
       sources correlate |r| >= EOG_R with the 1-10 Hz VEOG or HEOG proxy are
       removed
    4. ICA applied to the 0.1-20 Hz data (bad channels are left untouched for
       the diagnostics)
    5. residual windows (>RESIDUAL_UV on good channels) annotated
       BAD_residual; a channel tripping the threshold in >CHANNEL_BAD_SHARE of
       task windows is added to the bad channels
    6. diagnostics only: per-channel electrode-pop counts
    """

    raw = fx.make_raw(run)
    labels = raw.ch_names
    noise = fx.channel_noise(raw)
    bads = [c for c in labels if c in MANUAL_BADS]
    bads += [c for c in noise.loc[noise["bad"], "channel"] if c not in bads]

    fit = raw.copy().filter(*ICA_BAND)
    x_fit = fit.get_data() * 1e6
    p2p_fit, starts = sliding_p2p(x_fit)
    good_idx = [j for j, c in enumerate(labels) if c not in bads]
    gross = flag_windows(p2p_fit, starts, GROSS_UV, raw.n_times, channels=good_idx)
    fit.set_annotations(fit.annotations + mask_to_annotations(gross, "BAD_gross"))
    fit.info["bads"] = list(bads)
    n_comp = len(good_idx) - 1
    ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=SEED, max_iter=3000)
    ica.fit(fit, reject_by_annotation=True)

    sources = ica.get_sources(fit).get_data()
    proxies = eog_proxies(x_fit, labels)
    sos = butter(4, [1.0, 10.0], btype="band", fs=FS, output="sos")
    ok = task_mask & ~gross
    corr = {}
    for name, sig in proxies.items():
        sig = sosfiltfilt(sos, sig)
        corr[name] = np.array([np.corrcoef(sources[k, ok], sig[ok])[0, 1] for k in range(n_comp)])
    exclude = sorted({k for name in corr for k in np.flatnonzero(np.abs(corr[name]) >= EOG_R)})

    uncleaned = raw.copy().filter(*ERP_BAND)
    clean = uncleaned.copy()
    clean.info["bads"] = list(bads)
    ica.apply(clean, exclude=exclude)

    x = clean.get_data() * 1e6
    p2p, starts = sliding_p2p(x)
    task_win = task_mask[starts] & task_mask[np.minimum(starts + int(WIN_S * FS) - 1, len(task_mask) - 1)]
    share = pd.Series((p2p[:, task_win] > RESIDUAL_UV).mean(axis=1), index=labels)
    for c in labels:
        if c not in bads and share[c] > CHANNEL_BAD_SHARE:
            bads.append(c)
    clean.info["bads"] = list(bads)
    good_idx = [j for j, c in enumerate(labels) if c not in bads]
    residual = flag_windows(p2p, starts, RESIDUAL_UV, raw.n_times, channels=good_idx)
    clean.set_annotations(clean.annotations + mask_to_annotations(residual, "BAD_residual"))
    uncleaned.info["bads"] = list(bads)
    uncleaned.set_annotations(clean.annotations)

    qc = {
        "bad_channels": bads,
        "manual_bads": {c: MANUAL_BADS[c] for c in bads if c in MANUAL_BADS},
        "pops": electrode_pops(x, labels, task_mask, fx.blink_times(raw, channels=("AF3", "AF4"))),
        "channel_noise": noise,
        "gross_mask": gross,
        "residual_mask": residual,
        "ica": ica,
        "ica_corr": corr,
        "ica_exclude": [int(k) for k in exclude],
        "residual_share_by_channel": share,
        "raw": raw,
        "uncleaned": uncleaned,
    }
    return clean, qc


def electrode_pops(x_uv: np.ndarray, labels: list[str], task_mask: np.ndarray, blinks_s: np.ndarray) -> dict:
    """Channel-specific transients: peaks of |channel - median(other channels)| > POP_UV at 1-20 Hz.

    Run on the ICA-cleaned data, so blinks and saccades are already gone from
    the good channels.  Bad channels are not cleaned by ICA, so +/-0.5 s around
    each detected blink is masked for every channel.  A diagnostic for saline
    pads drying out.
    """

    hp = sosfiltfilt(butter(4, [1.0, 20.0], btype="band", fs=FS, output="sos"), x_uv, axis=1)
    usable = task_mask.copy()
    for b in blinks_s:
        usable[max(0, int((b - 0.5) * FS)):int((b + 0.5) * FS)] = False
    minutes = usable.sum() / FS / 60.0
    out = {"threshold_uv": POP_UV, "task_minutes": float(minutes), "times_s": {}, "per_min": {}}
    for j, c in enumerate(labels):
        d = hp[j] - np.median(np.delete(hp, j, axis=0), axis=0)
        d[~usable] = 0.0
        pk, _ = find_peaks(np.abs(d), height=POP_UV, distance=int(FS))
        out["times_s"][c] = (pk / FS).tolist()
        out["per_min"][c] = float(len(pk) / minutes)
    return out


def good_channels(inst) -> list[str]:
    return [c for c in inst.ch_names if c not in inst.info["bads"]]


def roi_label(name: str) -> str:
    return f"{name} ({' '.join(c for c in ROIS[name] if c in GOOD)})"


def event_table(run: fx.Run, shift_s: float, eeg_chain_s: float) -> pd.DataFrame:
    """Tone onsets on the EEG timeline (seconds from the first sample) and RT from the sound.

    Onset: marker + ``shift_s``.  RT from the sound: RT from the scheduled tone
    + half a frame (PsychoPy resets the keyboard clock on the routine's first
    flip, 0-1 frame after the tone was scheduled) + EEG chain latency - shift.
    """

    ev = run.tone_events.copy().reset_index(drop=True)
    ev["onset_eeg_s"] = ev["time"] + shift_s
    ev["onset_sample"] = nearest_sample(run.t, ev["onset_eeg_s"].to_numpy())
    ev["rt_from_sound_s"] = np.where(ev["pressed"], ev["rt_from_tone"] + FRAME_S / 2 + eeg_chain_s - shift_s, np.nan)
    ev["kind"] = np.select([(ev["stim"] == "go") & ev["pressed"], ev["stim"] == "go", ev["pressed"]],
                           ["go_hit", "go_miss", "nogo_fa"], "nogo_cr")
    return ev


def nearest_sample(t: np.ndarray, when: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t, when).clip(1, len(t) - 1)
    return (idx - (np.abs(t[idx - 1] - when) < np.abs(t[idx] - when))).astype(int)


def make_epochs(raw: mne.io.BaseRaw, samples: np.ndarray, meta: pd.DataFrame, tmin: float, tmax: float,
                baseline: tuple[float, float] | None) -> mne.Epochs:
    events = np.column_stack([samples, np.zeros(len(samples), int), np.ones(len(samples), int)])
    return mne.Epochs(raw, events, {"ev": 1}, tmin=tmin, tmax=tmax, baseline=baseline, reject=None,
                      reject_by_annotation=True, preload=True, metadata=meta.reset_index(drop=True),
                      picks=good_channels(raw), on_missing="ignore")


# =============================================================================
# Statistics helpers
# =============================================================================


def roi(data: np.ndarray, labels: list[str], chans: list[str]) -> np.ndarray:
    """Mean over the ROI channels present in ``labels`` (axis -2) of trials x channels x times."""

    return data[..., [labels.index(c) for c in chans if c in labels], :].mean(axis=-2)


def boot_mean_ci(trials: np.ndarray, rng, n_boot: int = N_BOOT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-bootstrap mean and 95% CI of trials x times."""

    idx = rng.integers(0, len(trials), (n_boot, len(trials)))
    return trials.mean(0), *np.percentile(trials[idx].mean(axis=1), [2.5, 97.5], axis=0)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = N_BOOT):
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    boots = a[ia].mean(1) - b[ib].mean(1)
    return a.mean(0) - b.mean(0), *np.percentile(boots, [2.5, 97.5], axis=0), boots


def plus_minus_rms(trials: np.ndarray, sel: np.ndarray, rng, n_perm: int = 200) -> float:
    """RMS of the +/- average (random half of trials inverted): the noise-only residual."""

    vals = []
    for _ in range(n_perm):
        sign = np.ones(len(trials))
        sign[rng.permutation(len(trials))[: len(trials) // 2]] = -1
        vals.append(np.sqrt(np.mean((trials * sign[:, None]).mean(0)[sel] ** 2)))
    return float(np.mean(vals))


def peak(times, wave, window, sign):
    sel = (times >= window[0]) & (times <= window[1])
    i = np.argmax(sign * wave[sel])
    return float(times[sel][i]), float(wave[sel][i])


def boot_peak_ci(times, boots, window, sign):
    sel = (times >= window[0]) & (times <= window[1])
    lat = times[sel][np.argmax(sign * boots[:, sel], axis=1)]
    return [float(v) for v in np.percentile(lat, [2.5, 97.5])]


# =============================================================================
# Analyses
# =============================================================================


def timing_shift() -> dict:
    # NOTE: this session used the 3.5-mm headphone jack through a sounddevice
    # stream (p3.psyexp; PTB was broken by a PsychoPy update), not the USB
    # speaker, and its N1 is ~60 ms later than the PTB headphone run.  The
    # shift below still adds the USB-speaker audio estimate; the intended
    # change is to shift by the verified EEG chain latency only and leave
    # audio latency unmodelled.
    loop = json.loads(Path(LOOPBACK).read_text())
    audio = json.loads(Path(AUDIO).read_text())
    eeg_chain = float(loop["drt_timing"]["eeg_chain_latency_ms"]["mean_of_corrected_edges"]) / 1000.0
    jack = float(loop["marker_timing"]["estimated_audio_latency_ms"]) / 1000.0
    speaker_minus_jack = float(audio["speaker_minus_headphones_audio_latency_s"]["from_waveform_alignment"])
    return {"eeg_chain_s": eeg_chain, "jack_audio_s": jack, "speaker_minus_jack_s": speaker_minus_jack,
            "usb_speaker_audio_s": jack + speaker_minus_jack, "shift_s": eeg_chain + jack + speaker_minus_jack,
            "session_audio_route": "3.5-mm headphone jack via sounddevice (shift still assumes PTB + USB speaker)",
            "oddball_ptb_n1_marker_latency_s": {
                "usb_speaker": audio["runs"]["USB speaker"]["n1_latency_s"],
                "usb_speaker_ci95": audio["runs"]["USB speaker"]["n1_ci95"],
                "headphone_jack": audio["runs"]["headphones"]["n1_latency_s"],
                "headphone_jack_ci95": audio["runs"]["headphones"]["n1_ci95"]}}


def n1_marker_check(clean: mne.io.BaseRaw, ev: pd.DataFrame, run: fx.Run, rng) -> dict:
    """Frontal-minus-inferior N1/P2 of NoGo tones, marker-locked, measured as in audio_latency.py."""

    nogo = ev[ev["kind"] == "nogo_cr"]
    ep = make_epochs(clean, nearest_sample(run.t, nogo["time"].to_numpy()), nogo, -0.2, 0.8, (-0.2, 0.0))
    present = lambda chans: [c for c in chans if c in ep.ch_names]
    lat = ex.n1_p2(ep, present(FRONTAL), present(INFERIOR), rng)
    return {k: v for k, v in lat.items() if not k.startswith("_")} | {"_wave": lat["_wave"], "_times": ep.times}


def conventional(clean: mne.io.BaseRaw, ev: pd.DataFrame, rng) -> dict:
    keep = ev[ev["kind"].isin(["go_hit", "nogo_cr"])]
    ep = make_epochs(clean, keep["onset_sample"].to_numpy(), keep, *EPOCH, BASELINE)
    labels, times = ep.ch_names, ep.times
    base = (times >= BASELINE[0]) & (times < BASELINE[1])
    data = ep.get_data() * 1e6
    go = (ep.metadata["stim"] == "go").to_numpy()
    out = {"epochs": ep, "data": data, "go": go, "times": times, "labels": labels, "roi": {}, "baseline_noise": {}}
    for name, chans in ROIS.items():
        tr = roi(data, labels, chans)
        d_m, d_lo, d_hi, d_boot = boot_diff_ci(tr[go], tr[~go], rng)
        out["roi"][name] = {"go": boot_mean_ci(tr[go], rng), "nogo": boot_mean_ci(tr[~go], rng),
                            "diff": (d_m, d_lo, d_hi), "diff_boot": d_boot,
                            "n_go": int(go.sum()), "n_nogo": int((~go).sum())}
        out["baseline_noise"][name] = {
            cond: {"baseline_rms_uv": float(np.sqrt(np.mean(tr[m].mean(0)[base] ** 2))),
                   "plus_minus_rms_uv": plus_minus_rms(tr[m], base, rng), "n": int(m.sum())}
            for cond, m in (("go", go), ("nogo", ~go))}
    return out


def block_replication(conv: dict, rng, window=(0.20, 0.35)) -> dict:
    """Go - NoGo mean amplitude in ``window`` per task block, frontal and posterior ROIs."""

    ep, data, times = conv["epochs"], conv["data"], conv["times"]
    sel = (times >= window[0]) & (times <= window[1])
    go = conv["go"]
    out = {"window_s": list(window), "blocks": {}}
    for name in ("frontal", "posterior"):
        tr = roi(data, conv["labels"], ROIS[name])
        for block in sorted(ep.metadata["block_idx"].unique()):
            m_block = (ep.metadata["block_idx"] == block).to_numpy()
            d_m, d_lo, d_hi, d_boot = boot_diff_ci(tr[m_block & go], tr[m_block & ~go], rng)
            amp = d_boot[:, sel].mean(1)
            out["blocks"][f"{name}/block{int(block) + 1}"] = {
                "go_minus_nogo_uv": float(d_m[sel].mean()), "ci95_uv": [float(v) for v in np.percentile(amp, [2.5, 97.5])],
                "n_go": int((m_block & go).sum()), "n_nogo": int((m_block & ~go).sum()),
                "_wave": (d_m, d_lo, d_hi)}
    return out


# =============================================================================
# Figures
# =============================================================================


def band(ax, t_ms, m, lo, hi, color, label, lw=1.6, ls="-"):
    ax.fill_between(t_ms, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(t_ms, m, color=color, lw=lw, ls=ls, label=label)


def deco_axes(ax, xlabel="ms from sound onset (corrected)", ylabel="µV"):
    ax.axvline(0, color="0.3", lw=0.7)
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def fig_qc(qc, clean, conv, task_mask, blocks, out: Path) -> None:
    raw, ica, labels = qc["raw"], qc["ica"], qc["raw"].ch_names
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.1, 1.1, 0.9])

    # Row 0: artifact timeline over the task blocks.
    ax = fig.add_subplot(gs[0, :])
    x = clean.get_data() * 1e6
    p2p, starts = sliding_p2p(x)
    tt = starts / FS
    im = ax.imshow(np.log10(np.maximum(p2p, 1)), aspect="auto", cmap="magma", vmin=1.2, vmax=2.5,
                   extent=[tt[0], tt[-1] + WIN_S, len(labels) - 0.5, -0.5], interpolation="nearest")
    for s, e in zip(*fx.contiguous(qc["residual_mask"])):
        ax.axvspan(s / FS, e / FS, ymin=0.0, ymax=0.04, color="cyan")
    for a, b in blocks:
        ax.axvline(a, color="white", lw=1.2, ls="--")
        ax.axvline(b, color="white", lw=1.2, ls="--")
    ax.set_yticks(range(len(labels)), [f"{c} (bad)" if c in qc["bad_channels"] else c for c in labels])
    ax.set_xlim(blocks[0][0] - 20, blocks[1][1] + 5)
    ax.set_xlabel("recording time (s)")
    kept = 100 * (1 - qc["residual_mask"][task_mask].mean())
    ax.set_title(f"Post-ICA 1-s peak-to-peak (log µV); cyan = BAD_residual (> {RESIDUAL_UV:.0f} µV on good channels, padded). "
                 f"Task data kept: {kept:.1f}%. Bad channels (not cleaned by ICA): {', '.join(qc['bad_channels']) or 'none'}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="log10 µV", pad=0.005)

    # Row 1: removed ICA components, correlations, blink-locked check.
    for j, k in enumerate(qc["ica_exclude"][:2]):
        ax = fig.add_subplot(gs[1, j])
        ica.plot_components(picks=[k], axes=ax, show=False, colorbar=False)
        ax.set_title(f"IC{k} removed\nr(VEOG)={qc['ica_corr']['VEOG'][k]:.2f}, r(HEOG)={qc['ica_corr']['HEOG'][k]:.2f}", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    idx = np.arange(len(qc["ica_corr"]["VEOG"]))
    ax.bar(idx - 0.2, np.abs(qc["ica_corr"]["VEOG"]), 0.4, label="|r| VEOG (AF3+AF4)")
    ax.bar(idx + 0.2, np.abs(qc["ica_corr"]["HEOG"]), 0.4, label="|r| HEOG (F7−F8)")
    ax.axhline(EOG_R, color="k", ls="--", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xlabel("independent component")
    ax.set_title("IC correlation with EOG proxies (1–10 Hz)", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    blinks = fx.blink_times(raw, channels=("AF3", "AF4"))
    before = qc["uncleaned"].get_data() * 1e6
    half = int(0.5 * FS)
    tb = np.arange(-half, half) / FS * 1000
    for arr, ls, name in [(before, "--", "before ICA"), (x, "-", "after ICA")]:
        segs = np.stack([arr[:, int(b * FS) - half:int(b * FS) + half] for b in blinks if half < b * FS < arr.shape[1] - half])
        m = segs.mean(0)
        m = m - m[:, :int(0.15 * FS)].mean(1, keepdims=True)
        for c, col in [("AF3", "tab:blue"), ("AF4", "tab:orange"), ("F8", "tab:green"), ("O1", "tab:gray")]:
            ax.plot(tb, m[labels.index(c)], ls=ls, color=col, lw=1.2, label=f"{c} {name}" if ls == "-" or c == "AF3" else None)
    ax.set_title(f"Blink-locked average (n={len(blinks)})", fontsize=10)
    deco_axes(ax, "ms from blink peak")
    ax.legend(fontsize=7)

    # Row 2: horizontal-EOG proxy before/after ICA on the kept epochs; baseline vs noise.
    ep = conv["epochs"]
    t_ms = ep.times * 1000
    before_ep = make_epochs(qc["uncleaned"], ep.events[:, 0], ep.metadata, *EPOCH, BASELINE)
    heog = {}
    for name, e in (("before ICA", before_ep), ("after ICA", ep)):
        d = e.get_data() * 1e6
        heog[name] = d[:, e.ch_names.index("F7")] - d[:, e.ch_names.index("F8")]
    go = conv["go"]
    ax = fig.add_subplot(gs[2, 0])
    for name, ls in (("before ICA", "--"), ("after ICA", "-")):
        for m, col, lab in [(go, COLORS["go"], "Go"), (~go, COLORS["nogo"], "NoGo")]:
            ax.plot(t_ms, heog[name][m].mean(0), color=col, ls=ls, lw=1.2 if ls == "--" else 1.8, label=f"{lab} {name}")
    ax.set_title("F7−F8 (horizontal-EOG proxy) epoch average", fontsize=10)
    deco_axes(ax)
    ax.legend(fontsize=8)
    for j, name in enumerate(("before ICA", "after ICA")):
        ax = fig.add_subplot(gs[2, 1 + j])
        ax.imshow(heog[name], aspect="auto", cmap="RdBu_r", vmin=-40, vmax=40, extent=[t_ms[0], t_ms[-1], len(heog[name]), 0])
        ax.set_title(f"F7−F8 single trials, {name} (±40 µV)", fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel("epoch")
    ax = fig.add_subplot(gs[2, 3])
    rows = [(f"{name} {cond}", v) for name, d in conv["baseline_noise"].items() for cond, v in d.items()]
    yy = np.arange(len(rows))
    ax.barh(yy + 0.2, [r[1]["baseline_rms_uv"] for r in rows], 0.4, label="baseline RMS (−200–0 ms)")
    ax.barh(yy - 0.2, [r[1]["plus_minus_rms_uv"] for r in rows], 0.4, label="± average RMS (noise only)")
    ax.set_yticks(yy, [f"{n} (n={r['n']})" for n, r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("µV")
    ax.set_title("Baseline wiggle vs expected noise", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    # Row 3: electrode-pop diagnostics.
    pops = qc["pops"]
    ax = fig.add_subplot(gs[3, :2])
    ax.bar(range(len(labels)), [pops["per_min"][c] for c in labels],
           color=["tab:red" if c in qc["bad_channels"] else "0.5" for c in labels])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("pops / min (task)")
    ax.set_title(f"Electrode pops: |channel − median of others| > {pops['threshold_uv']:.0f} µV (1–20 Hz), "
                 "blinks masked; red = bad", fontsize=10)
    ax = fig.add_subplot(gs[3, 2:])
    for c in labels:
        tt = np.asarray(pops["times_s"][c])
        if len(tt) >= 5:
            ax.plot(tt[1:], np.diff(tt), "o-", ms=3, lw=1, label=f"{c} (n={len(tt)})")
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel("interval since previous pop (s)")
    ax.set_title("Pop intervals (channels with ≥5 pops; long gaps can be pops hidden by blink masking)", fontsize=10)
    ax.grid(alpha=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.suptitle("Go/NoGo artifact handling and quality control", fontsize=14)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_erp(conv, n1, blocks_rep, ev, cluster, timing, rng, out: Path) -> None:
    t_ms = conv["times"] * 1000
    labels = conv["labels"]
    rt = ev.loc[ev["kind"] == "go_hit", "rt_from_sound_s"].to_numpy() * 1000
    fig = plt.figure(figsize=(19, 15), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 0.9, 0.75, 0.75])

    # Row 0: ROI waveforms and the Go - NoGo butterfly with cluster-test intervals.
    for c, name in enumerate(ROIS):
        ax = fig.add_subplot(gs[0, c])
        d = conv["roi"][name]
        band(ax, t_ms, *d["go"], COLORS["go"], f"Go hits ({d['n_go']})")
        band(ax, t_ms, *d["nogo"], COLORS["nogo"], f"NoGo ({d['n_nogo']})")
        band(ax, t_ms, *d["diff"], COLORS["diff"], "Go − NoGo", lw=1.2, ls=":")
        ax.axvspan(*np.percentile(rt, [25, 75]), color=COLORS["resp"], alpha=0.10, label="Go RT IQR")
        ax.axvline(np.median(rt), color=COLORS["resp"], lw=1, ls="--")
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.set_title(roi_label(name), fontsize=10)
        deco_axes(ax)
        if c == 0:
            ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[0, 3])
    diff = conv["data"][conv["go"]].mean(0) - conv["data"][~conv["go"]].mean(0)
    cmap = plt.get_cmap("tab20")
    for j, ch in enumerate(labels):
        ax.plot(t_ms, diff[j], lw=1, color=cmap(j), label=ch)
    for cl in cluster["clusters"]:
        if cl["p"] < 0.05:
            ax.axvspan(cl["t_start"] * 1000, cl["t_end"] * 1000, color="gold", alpha=0.25)
    ax.set_title("Go − NoGo, all channels; gold = cluster p < .05", fontsize=10)
    ax.set_xlim(t_ms[0], t_ms[-1])
    deco_axes(ax)
    ax.legend(fontsize=6, ncol=2)

    # Row 1: N1 timing check (marker-locked) and block replication.
    ax = fig.add_subplot(gs[1, :2])
    tn = n1["_times"] * 1000
    band(ax, tn, *boot_mean_ci(n1["_wave"], rng), "k", f"this session, NoGo (n={n1['n_trials']})")
    ref = timing["oddball_ptb_n1_marker_latency_s"]
    for key, col, lab in [("headphone_jack", "tab:purple", "oddball N1, PTB + headphone jack"),
                          ("usb_speaker", "tab:olive", "oddball N1, PTB + USB speaker")]:
        ax.axvline(1000 * ref[key], color=col, ls="--", lw=1.2, label=f"{lab}: {1000 * ref[key]:.0f} ms")
    ax.axvline(1000 * n1["n1_latency_s"], color="k", ls=":", lw=1.2, label=f"this session N1: {1000 * n1['n1_latency_s']:.0f} ms")
    ax.axvline(1000 * timing["eeg_chain_s"], color="tab:gray", lw=1, label=f"EEG chain latency: {1000 * timing['eeg_chain_s']:.0f} ms")
    ax.set_title("Timing check: frontal − inferior, locked to the marker (no latency correction)", fontsize=10)
    deco_axes(ax, "ms from marker")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    for key, col in [("frontal/block1", "tab:purple"), ("frontal/block2", "tab:brown")]:
        b = blocks_rep["blocks"][key]
        band(ax, t_ms, *b["_wave"], col, f"{key.split('/')[1]} ({b['n_go']}/{b['n_nogo']}): "
                                         f"{b['go_minus_nogo_uv']:.1f} µV [{b['ci95_uv'][0]:.1f}, {b['ci95_uv'][1]:.1f}]")
    ax.axvspan(*[1000 * v for v in blocks_rep["window_s"]], color="tab:green", alpha=0.08)
    ax.set_title(f"Go − NoGo by block: {roi_label('frontal')}", fontsize=10)
    deco_axes(ax)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(rt, bins=np.arange(np.floor(rt.min() / 10) * 10, rt.max() + 10, 10), color=COLORS["resp"], alpha=0.8)
    q = np.percentile(rt, [25, 50, 75])
    for v in q:
        ax.axvline(v, color="k", ls="--", lw=0.8)
    ax.set_title(f"Go RT from corrected onset\nmedian {q[1]:.0f} ms, IQR {q[0]:.0f}–{q[2]:.0f} ms", fontsize=10)
    ax.set_xlabel("ms")

    # Rows 2-3: topographies.
    ep = conv["epochs"]
    ev_go, ev_nogo = ep[conv["go"]].average(), ep[~conv["go"]].average()
    ev_diff = mne.combine_evoked([ev_go, ev_nogo], weights=[1, -1])
    topo_times = [0.08, 0.16, 0.25, 0.35, 0.45, 0.6]
    for r, (evk, name) in enumerate([(ev_nogo, "NoGo"), (ev_diff, "Go − NoGo")]):
        vmax = np.abs(evk.data[:, (evk.times >= 0.05) & (evk.times <= 0.7)]).max() * 1e6
        sub = gs[2 + r, :].subgridspec(1, len(topo_times))
        for c, tt in enumerate(topo_times):
            ax = fig.add_subplot(sub[0, c])
            evk.plot_topomap(times=[tt], average=0.04, axes=ax, show=False, colorbar=False, sensors=True, vlim=(-vmax, vmax))
            ax.set_title(f"{name} {1000 * tt:.0f} ms (±{vmax:.1f} µV)", fontsize=9)
    fig.suptitle(f"Go/NoGo ERPs, recorded (CMS) reference, 0.1–20 Hz, baseline −200–0 ms, 95% trial-bootstrap bands. "
                 f"Onsets shifted +{1000 * timing['shift_s']:.1f} ms (see timing note)", fontsize=12)
    fig.savefig(out, dpi=110)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    run = load_recording(XDF, CSV)
    timing = timing_shift()
    ev = event_table(run, timing["shift_s"], timing["eeg_chain_s"])
    blocks = [(float(ev.loc[ev.block_idx == b, "onset_eeg_s"].min() - 1.0),
               float(ev.loc[ev.block_idx == b, "onset_eeg_s"].max() + 1.5)) for b in sorted(ev.block_idx.unique())]
    task_mask = np.zeros(len(run.t), bool)
    for a, b in blocks:
        task_mask[int(a * FS):int(b * FS)] = True

    clean, qc = preprocess(run, task_mask)
    GOOD[:] = good_channels(clean)
    conv = conventional(clean, ev, rng)
    n1 = n1_marker_check(clean, ev, run, rng)
    blocks_rep = block_replication(conv, rng)
    ep = conv["epochs"]
    res = fx.cluster_test(ep[conv["go"]], ep[~conv["go"]], tmin=0.0, tmax=EPOCH[1], n_perm=args.permutations)
    cluster = {"clusters": res["clusters"], "n_go": res["n_oddball"], "n_nogo": res["n_standard"]}

    fig_qc(qc, clean, conv, task_mask, blocks, OUT / "qc_artifacts.png")
    fig_erp(conv, n1, blocks_rep, ev, cluster, timing, rng, OUT / "erp.png")

    results = summarise(ev, timing, qc, conv, n1, blocks_rep, cluster, task_mask, blocks, args)
    (OUT / "results.json").write_text(json.dumps(fx.to_jsonable(results), indent=2), encoding="utf-8")
    ev.to_csv(OUT / "events.csv", index=False)
    segs = [{"label": a["description"], "onset_s": a["onset"], "duration_s": a["duration"]} for a in clean.annotations]
    pd.DataFrame(segs).to_csv(OUT / "bad_segments.csv", index=False)
    print(json.dumps(fx.to_jsonable(results), indent=2))


def summarise(ev, timing, qc, conv, n1, blocks_rep, cluster, task_mask, blocks, args) -> dict:
    times = conv["times"]
    out = {
        "recording": XDF,
        "timing": timing | {"n1_marker_check": {k: v for k, v in n1.items() if not k.startswith("_")}},
        "events": {
            "go_hit": int((ev["kind"] == "go_hit").sum()), "go_miss": int((ev["kind"] == "go_miss").sum()),
            "nogo_cr": int((ev["kind"] == "nogo_cr").sum()), "nogo_fa": int((ev["kind"] == "nogo_fa").sum()),
            "rt_from_sound_ms": ev.loc[ev["kind"] == "go_hit", "rt_from_sound_s"].mul(1000).describe().to_dict(),
            "blocks_eeg_s": blocks,
        },
        "preprocessing": {
            "parameters": {"ica_band_hz": ICA_BAND, "erp_band_hz": ERP_BAND, "gross_uv": GROSS_UV,
                           "residual_uv": RESIDUAL_UV, "window_s": WIN_S, "step_s": STEP_S, "pad_s": PAD_S,
                           "eog_r": EOG_R, "channel_bad_share": CHANNEL_BAD_SHARE, "pop_uv": POP_UV,
                           "epoch_s": EPOCH, "baseline_s": BASELINE, "reference": "CMS (as recorded)",
                           "n_boot": N_BOOT, "cluster_permutations": args.permutations},
            "bad_channels": qc["bad_channels"],
            "manual_bad_channels": qc["manual_bads"],
            "roi_channels_used": {name: [c for c in chans if c in GOOD] for name, chans in ROIS.items()},
            "electrode_pops": {"threshold_uv": qc["pops"]["threshold_uv"], "usable_task_minutes": qc["pops"]["task_minutes"],
                               "per_min": qc["pops"]["per_min"], "times_s": qc["pops"]["times_s"]},
            "ica_components": [{"ic": k, "r_veog": float(qc["ica_corr"]["VEOG"][k]), "r_heog": float(qc["ica_corr"]["HEOG"][k]),
                                "removed": k in qc["ica_exclude"]} for k in range(len(qc["ica_corr"]["VEOG"]))],
            "gross_fraction_of_task": float(qc["gross_mask"][task_mask].mean()),
            "residual_bad_fraction_of_task": float(qc["residual_mask"][task_mask].mean()),
            "residual_window_share_by_channel": qc["residual_share_by_channel"].to_dict(),
            "epochs_kept": {"go_hit": conv["roi"]["frontal"]["n_go"], "nogo_cr": conv["roi"]["frontal"]["n_nogo"]},
        },
        "baseline_noise": conv["baseline_noise"],
        "peaks": {},
        "cluster_go_vs_nogo": cluster,
        "block_replication": {"window_s": blocks_rep["window_s"],
                              "blocks": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                         for k, v in blocks_rep["blocks"].items()}},
    }
    windows = {"N1": ((0.04, 0.14), -1), "P2": ((0.12, 0.24), 1), "P3": ((0.20, 0.50), 1)}
    for name in ROIS:
        d = conv["roi"][name]
        entry = {}
        for comp, (win, sign) in windows.items():
            for cond in ("go", "nogo"):
                lat, amp = peak(times, d[cond][0], win, sign)
                entry[f"{cond}_{comp}"] = {"latency_ms": 1000 * lat, "amp_uv": amp}
            lat, amp = peak(times, d["diff"][0], win, sign)
            entry[f"diff_{comp}"] = {"latency_ms": 1000 * lat, "amp_uv": amp,
                                     "latency_ci95_ms": [1000 * v for v in boot_peak_ci(times, d["diff_boot"], win, sign)]}
        out["peaks"][name] = entry
    # Why the average reference is not used: the Go - NoGo positivity is on every channel.
    p3_t = out["peaks"]["frontal"]["diff_P3"]["latency_ms"] / 1000
    i = int(np.argmin(np.abs(times - p3_t)))
    diff = conv["data"][conv["go"]].mean(0) - conv["data"][~conv["go"]].mean(0)
    out["go_minus_nogo_at_frontal_p3_peak"] = {"latency_ms": 1000 * float(times[i]),
                                               "by_channel_uv": dict(zip(conv["labels"], diff[:, i].tolist())),
                                               "channel_mean_uv": float(diff[:, i].mean())}
    return out


if __name__ == "__main__":
    main()
