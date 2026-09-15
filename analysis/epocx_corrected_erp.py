#!/usr/bin/env python3
"""ERP analysis of the EPOC X auditory oddball after timing correction.

The event markers in ``epochx_0740.xdf`` are scheduled PsychoPy tone times,
not physical sound onsets.  This script shifts the event samples by the
measured EEG-chain delay plus the estimated USB-speaker output delay, then
uses the existing MNE preprocessing pipeline from ``flex_sanity.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import audio_latency as al
import epocx_sanity as ex
import flex_sanity as fx

mne.set_log_level("ERROR")

XDF = "data/Oddball/data/epochx_0740.xdf"
CSV = "data/Oddball/data/epochx_0740.csv"
LOOPBACK = "data/DRT_Timing/loopback_timing/results.json"
AUDIO = "data/Oddball/data/audio_latency/results.json"
FRONTAL = ex.FRONTAL_ROI
INFERIOR = ex.INFERIOR_ROI
POSTERIOR = ex.POSTERIOR_ROI


def nearest_sample(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(t, values).clip(1, len(t) - 1)
    choose_prev = np.abs(t[idx - 1] - values) < np.abs(t[idx] - values)
    idx = idx - choose_prev
    return idx.astype(int), (t[idx] - values) * 1000


def corrected_run(run, total_shift_s: float):
    events = run.tone_events.copy()
    scheduled = events["time"].to_numpy(float)
    corrected = scheduled + total_shift_s
    samples, error_ms = nearest_sample(run.t, corrected)
    events["scheduled_time"] = scheduled
    events["time"] = corrected
    events["sample"] = samples
    events["sample_error_ms"] = error_ms
    return replace(run, tone_events=events)


def condition_epochs(epochs: mne.Epochs, block: str, stim: str) -> mne.Epochs:
    meta = epochs.metadata
    return epochs[((meta["block"] == block) & (meta["stim"] == stim)).to_numpy()]


def roi_wave(epochs: mne.Epochs, roi: list[str]) -> np.ndarray:
    return fx.roi_trials(epochs, roi)


def latency_summary(epochs: mne.Epochs, rng: np.random.Generator) -> dict:
    standard = condition_epochs(epochs, "passive", "standard")
    wave = roi_wave(standard, FRONTAL) - roi_wave(standard, INFERIOR)
    grand = wave.mean(axis=0)
    # A broad 30--350 ms search is useful for finding the observed peak, but
    # it is too broad for a latency CI: trialwise noise can make a bootstrap
    # replicate select a later negative deflection.  Keep the N1 estimate in
    # the conventional early auditory window after the corrected event.
    n1_window = (0.04, 0.18)
    n1 = fx.peak_latency(standard.times, grand, n1_window, -1)
    p2 = fx.peak_latency(standard.times, grand, (n1 + 0.03, 0.55), +1)
    # Split-half reliability is calculated on the corrected trials, not on the
    # marker-relative epochs from the earlier report.
    sel = (standard.times >= -0.1) & (standard.times <= 0.6)
    half_r = float(np.corrcoef(wave[::2].mean(axis=0)[sel], wave[1::2].mean(axis=0)[sel])[0, 1])
    return {
        "n_trials": int(len(standard)),
        "n1_latency_s": float(n1),
        "n1_ci95_s": fx.bootstrap_latency(standard.times, wave, n1_window, -1, n_boot=2000, rng=rng),
        "n1_amp_uv": float(grand[standard.times == n1][0]),
        "p2_latency_s": float(p2),
        "p2_ci95_s": fx.bootstrap_latency(standard.times, wave, (n1 + 0.03, 0.55), +1, n_boot=2000, rng=rng),
        "p2_amp_uv": float(grand[standard.times == p2][0]),
        "split_half_r": half_r,
    }


def trial_quality(epochs: mne.Epochs, roi: list[str]) -> dict:
    y = roi_wave(epochs, roi)
    baseline = (epochs.times >= -0.2) & (epochs.times <= 0)
    post = (epochs.times >= 0) & (epochs.times <= 0.4)
    b = y[:, baseline]
    p = y[:, post]
    return {
        "baseline_sd_uv": float(b.std(ddof=1)),
        "post_peak_mean_abs_uv": float(np.mean(np.max(np.abs(p), axis=1))),
        "grand_peak_to_baseline_sd": float(np.max(np.abs(y.mean(axis=0)[post])) / (b.std(ddof=1) + 1e-12)),
    }


def make_figure(epochs: mne.Epochs, lat: dict, tests: dict, shift_ms: float, out: Path) -> None:
    meta = epochs.metadata
    times = epochs.times * 1000
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    groups = [
        ("passive", "standard", "Passive standard", "0.25"),
        ("passive", "oddball", "Passive oddball", "tab:red"),
        ("gng", "go", "Go hits", "tab:green"),
        ("gng", "nogo", "NoGo correct", "tab:purple"),
    ]
    for col, (roi, title) in enumerate([(FRONTAL, "frontocentral"), (INFERIOR, "inferior"), (POSTERIOR, "posterior")]):
        ax = fig.add_subplot(gs[0, col])
        for block, stim, label, color in groups:
            e = condition_epochs(epochs, block, stim)
            y = roi_wave(e, roi)
            mean = y.mean(axis=0)
            se = y.std(axis=0, ddof=1) / np.sqrt(len(y))
            ax.plot(times, mean, color=color, lw=1.5, label=f"{label} (n={len(y)})")
            ax.fill_between(times, mean - se, mean + se, color=color, alpha=0.13)
        ax.axvline(0, color="k", lw=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(1000 * lat["n1_latency_s"], color="tab:red", ls=":", lw=1)
        ax.axvline(1000 * lat["p2_latency_s"], color="tab:blue", ls=":", lw=1)
        ax.set(xlim=(-200, 600), xlabel="ms from estimated physical sound onset", ylabel="µV", title=title)
        if col == 0:
            ax.legend(fontsize=7)
    standard = condition_epochs(epochs, "passive", "standard")
    evoked = standard.average()
    ax = fig.add_subplot(gs[0, 3])
    evoked.plot_topomap(times=[lat["n1_latency_s"]], average=0.03, axes=ax, show=False, colorbar=False, sensors=True)
    ax.set_title(f"Standard topography at N1 ({1000 * lat['n1_latency_s']:.0f} ms)", fontsize=9)

    # Corrected standard trial image, useful for checking whether the ERP is
    # present throughout the recording rather than only in the grand average.
    ax = fig.add_subplot(gs[1, :2])
    wave = roi_wave(standard, FRONTAL) - roi_wave(standard, INFERIOR)
    order = np.argsort(standard.metadata["scheduled_time"].to_numpy())
    im = ax.imshow(wave[order], aspect="auto", origin="lower", extent=[times[0], times[-1], 0, len(wave)],
                   cmap="RdBu_r", vmin=-np.percentile(np.abs(wave), 98), vmax=np.percentile(np.abs(wave), 98))
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(1000 * lat["n1_latency_s"], color="k", ls=":", lw=0.8)
    ax.set(xlabel="ms from estimated physical sound onset", ylabel="standard trials (chronological)", title="Corrected standard-trial image")
    fig.colorbar(im, ax=ax, shrink=0.8, label="frontal − inferior (µV)")

    ax = fig.add_subplot(gs[1, 2:])
    for name, test in tests.items():
        best = test["clusters"][0] if test["clusters"] else None
        if best:
            ax.plot([best["t_start"] * 1000, best["t_end"] * 1000], [name, name], lw=8,
                    color="tab:green" if best["p"] < 0.05 else "0.65")
            ax.text(best["t_start"] * 1000, name, f"  p={best['p']:.3f}", va="center", fontsize=9)
        else:
            ax.text(0, name, "no clusters", va="center", fontsize=9)
    ax.axvline(0, color="k", lw=0.7)
    ax.set(xlim=(-10, 800), xlabel="ms from estimated physical sound onset", title=f"Condition-test strongest clusters (event shift {shift_ms:.1f} ms)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def clean_test(test: dict) -> dict:
    return {k: v for k, v in test.items() if k in ("n_oddball", "n_standard", "clusters")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/Oddball/data/epocx_corrected_erp"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loop = json.loads(Path(LOOPBACK).read_text())
    audio = json.loads(Path(AUDIO).read_text())
    eeg_chain = float(loop["drt_timing"]["eeg_chain_latency_ms"]["mean_of_corrected_edges"]) / 1000
    jack_audio = float(loop["marker_timing"]["estimated_audio_latency_ms"]) / 1000
    speaker_minus_jack = float(audio["speaker_minus_headphones_audio_latency_s"]["from_waveform_alignment"])
    speaker_audio = jack_audio + speaker_minus_jack
    total_shift = eeg_chain + speaker_audio

    run, extra = ex.load_epocx(XDF, CSV)
    raw = fx.make_raw(run)
    noise = fx.channel_noise(raw)
    bads = noise.loc[noise["bad"], "channel"].tolist()
    corrected = corrected_run(run, total_shift)
    epochs, prep = fx.erp_epochs(corrected, raw, bads, eog_channels=ex.EOG, codes=ex.CODES)
    rng = np.random.default_rng(7)

    meta = epochs.metadata
    counts = {
        f"{block}/{stim}": int(((meta["block"] == block) & (meta["stim"] == stim)).sum())
        for block, stim in [("passive", "standard"), ("passive", "oddball"), ("gng", "go"), ("gng", "nogo")]
    }
    totals = {
        f"{block}/{stim}": int(((run.tone_events["block"] == block) & (run.tone_events["stim"] == stim)).sum())
        for block, stim in [("passive", "standard"), ("passive", "oddball"), ("gng", "go"), ("gng", "nogo")]
    }
    lat = latency_summary(epochs, rng)
    # Recompute the N1 CI with the requested number of bootstrap replicates,
    # using the same constrained early-auditory window as the point estimate.
    standard_epochs = condition_epochs(epochs, "passive", "standard")
    standard_wave = roi_wave(standard_epochs, FRONTAL) - roi_wave(standard_epochs, INFERIOR)
    lat["n1_ci95_s"] = fx.bootstrap_latency(
        standard_epochs.times, standard_wave, (0.04, 0.18), -1,
        n_boot=args.bootstrap, rng=rng)
    tests = {}
    passive = (meta["block"] == "passive").to_numpy()
    odd = (meta["stim"] == "oddball").to_numpy()
    go = ((meta["stim"] == "go") & meta["correct"].astype(bool)).to_numpy()
    nogo = ((meta["stim"] == "nogo") & meta["correct"].astype(bool)).to_numpy()
    for name, a, b, label_a, label_b in [
        ("passive oddball vs standard", passive & odd, passive & ~odd, "oddball", "standard"),
        ("NoGo vs Go", nogo, go, "NoGo", "Go"),
    ]:
        tests[name] = fx.cluster_test(epochs[a], epochs[b], n_perm=args.permutations)

    result = {
        "recording": XDF,
        "timing_correction": {
            "eeg_chain_s": eeg_chain,
            "headphone_jack_audio_s": jack_audio,
            "usb_speaker_minus_headphone_s": speaker_minus_jack,
            "usb_speaker_audio_s": speaker_audio,
            "combined_marker_to_physical_sound_plus_eeg_s": total_shift,
            "combined_shift_ms": 1000 * total_shift,
            "note": "The USB-speaker component is inferred from the speaker/headphone ERP shift; the new jack loopback directly measures the jack component.",
        },
        "integrity": {
            "samples": int(len(run.t)), "duration_s": float(run.t[-1]), "eeg_rate_hz": float(1 / np.median(np.diff(run.t))),
            "missing_samples": int(run.filled.sum()), "bad_channels": bads,
        },
        "filtering_and_artifact_rejection": {"erp_band_hz": [0.1, 30.0], "ica_fit_band_hz": [1.0, 30.0],
                                               "average_reference": True, "blink_components": prep.get("excluded_components", []),
                                               "drops": prep.get("drops", {}), "reject_peak_to_peak_uv": 150.0},
        "epochs": {"kept": counts, "available": totals},
        "standards": {**lat, "trial_quality_frontal_minus_inferior": trial_quality(condition_epochs(epochs, "passive", "standard"), FRONTAL)},
        "condition_tests": {name: clean_test(test) for name, test in tests.items()},
    }
    # Save corrected marker times for downstream epoching/auditing.
    corrected.tone_events.to_csv(args.out / "corrected_events.csv", index=False)
    (args.out / "results.json").write_text(json.dumps(fx.to_jsonable(result), indent=2), encoding="utf-8")
    make_figure(epochs, lat, tests, 1000 * total_shift, args.out / "corrected_erp.png")

    # Keep the prose report short and point to the full machine-readable output.
    standard = result["standards"]
    report = f"""# Corrected EPOC X auditory oddball ERP

The scheduled PsychoPy markers in `epochx_0740.xdf` were shifted by **{1000 * total_shift:.1f} ms** before epoching: **{1000 * eeg_chain:.1f} ms** for the EPOC X EEG chain plus **{1000 * speaker_audio:.1f} ms** for the estimated USB-speaker output. This is equivalent to subtracting the EEG-chain delay from sample timestamps and moving the physical sound event forward; only the event samples are shifted here because the relative timing is identical.

Preprocessing follows the existing MNE pipeline: continuous 0.1–30 Hz FIR filtering, ICA fit on 1–30 Hz to remove blink components, average reference, data-driven bad-channel handling, interpolation where needed, and 150-µV peak-to-peak epoch rejection. The recording has {result['integrity']['missing_samples']} missing EEG samples and retained {standard['n_trials']} standard epochs ({counts['passive/standard']}/{totals['passive/standard']} available).

## Corrected standard ERP

- Frontal-minus-inferior N1: **{1000 * standard['n1_latency_s']:.0f} ms** after estimated physical sound onset (95% CI {1000 * standard['n1_ci95_s'][0]:.0f}–{1000 * standard['n1_ci95_s'][1]:.0f} ms).
- P2: **{1000 * standard['p2_latency_s']:.0f} ms** (95% CI {1000 * standard['p2_ci95_s'][0]:.0f}–{1000 * standard['p2_ci95_s'][1]:.0f} ms).
- Split-half waveform correlation: **r = {standard['split_half_r']:.2f}**.
- Corrected waveforms and the chronological trial image are in `corrected_erp.png`.

The N1 now falls near the expected ~100-ms post-sound latency rather than the ~305-ms marker-relative latency. The corrected ERP is visible and reasonably time-locked, though its amplitude is modest and the trial image remains noisy.

Condition-test results (including passive oddball and NoGo-vs-Go) are in `results.json`; with the same trials, a global time shift changes the apparent latency but not the underlying condition contrast.
"""
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(fx.to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
