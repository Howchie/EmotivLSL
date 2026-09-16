#!/usr/bin/env python3
"""Measure the audio-output latency of the PsychoPy/WASAPI tone path on the EPOC X.

Setup
-----
A 3.5 mm headphone plug is pushed through the hole in the back of one EPOC X
electrode housing and rests against the sensor, with the headset on the desk and
the pads soaked in saline.  Every tone is therefore injected *electrically* into
that channel at the instant it reaches the DAC, which turns the EEG recording
into a microphone for the audio path.  Earlier versions of this test held the
plug by hand; the housing now supports it, which is what makes the per-trial
scatter small enough to be worth quoting.

PsychoPy plays each tone through a WASAPI exclusive-mode ``sounddevice`` stream
and timestamps the LSL marker from PortAudio's ``outputBufferDacTime`` estimate
of when the first sample hits the DAC (see ``data/Headphone_Timing/headphones.py``).
The marker is therefore already meant to be the physical sound onset, and what
this script measures is how far off that estimate is.

Method
------
The tone is 1 kHz, far above the headset's 128 Hz output, so the tone body does
not survive to the recording.  What does survive is the broadband energy at the
tone's onset and offset ramps, which appears as two bursts about one tone-length
apart.  Timing each burst by the centroid of its excess energy is insensitive to
pickup amplitude, which matters because the coupling varies from trial to trial
and between rigs -- a fixed threshold would trip at a different point on the
chain's filter ramp for a strong burst than a weak one.

PsychoPy fades generated tones in and out over 5 ms, so each burst's centroid
sits 2.5 ms inside the tone.  The spacing of the two centroids recovers the tone
duration and is used here as a self-check that the estimator is tracking the real
edges rather than noise.

The high-pass is applied zero-phase on purpose: it spreads each burst
symmetrically and so leaves its centroid where it was, whereas a causal filter
would add its own group delay to the number being measured.

    marker -> onset burst = EEG-chain latency + audio-output latency

The EEG-chain latency (radio, dongle, onboard filtering) is measured separately
by the DRT hardware test, so subtracting it leaves the audio path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import emotiv as em

LABELS = list(em.EPOCX.labels)

SEED = 7
FS = 128.0  # nominal; the true rate comes out of the grid fit
TONE_S = 0.100
RAMP_S = 0.005  # PsychoPy's fade-in/out on generated tones
HIGHPASS_HZ = 5.0
MARKER_PREFIX = "Stim"

# Search windows relative to the marker.
BASELINE_S = (-0.300, -0.020)
BURST_S = (0.030, 0.330)



# ---------------------------------------------------------------------------
# Loading


def load_run(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return the sample grid times, the EEG matrix, the tone marker times and diagnostics.

    This is the canonical EPOC X loader; nothing here corrects timing, because the
    chain latency is what the test exists to check rather than assume.
    """

    run = em.load_epocx(path)
    tones = run.markers.loc[run.markers["value"].str.startswith(MARKER_PREFIX), "time"].to_numpy()
    if not len(tones):
        raise ValueError(f"no markers starting with {MARKER_PREFIX!r}")

    residual = run.extra["arrival_residual_ms"]
    diagnostics = {
        "samples": int(len(run.t)),
        "duration_s": run.duration_s,
        "interpolated_samples": int(run.filled.sum()),
        "fitted_rate_hz": run.extra["segment_rates_hz"],
        "arrival_residual_sd_ms": float(np.std(residual)),
        "tones": int(len(tones)),
    }
    return run.t, run.x_uv, tones, diagnostics


def eeg_chain_s(override: float | None) -> tuple[float, str]:
    """The independently measured EEG-chain latency, in seconds.

    Taken from ``emotiv.devices``, which is the single place the DRT result is
    recorded, so re-running the hardware test updates this estimate too.
    """

    if override is not None:
        return override, "--eeg-chain"
    timing = em.EPOCX.timing
    return timing.latency_s, f"emotiv.devices.EPOCX.timing ({timing.source})"


# ---------------------------------------------------------------------------
# Measurement


def highpass(sig: np.ndarray) -> np.ndarray:
    return sosfiltfilt(butter(4, HIGHPASS_HZ, btype="high", fs=FS, output="sos"), sig)


def epoch(t: np.ndarray, sig: np.ndarray, tones: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Tone-locked epochs on a common lag axis (s), by interpolation onto the grid."""

    return np.array([np.interp(lags, t - e, sig) for e in tones])


def pickup_ranking(t: np.ndarray, x: np.ndarray, tones: np.ndarray) -> pd.DataFrame:
    """Rank channels by tone-locked energy, to find which sensor the plug is on.

    The plugged channel stands out by orders of magnitude, so this is a reliable
    way to record which sensor was actually used rather than trusting the label
    on the file.
    """

    lags = np.arange(BASELINE_S[0], BURST_S[1] + 0.05, 1 / FS)
    rows = []
    for j, label in enumerate(LABELS):
        power = (epoch(t, highpass(x[:, j]), tones, lags) ** 2).mean(axis=0)
        base = np.sqrt(power[lags < BASELINE_S[1]].mean())
        burst = np.sqrt(power[(lags > BURST_S[0]) & (lags < BURST_S[1])].max())
        rows.append({"channel": label, "baseline_uv": base, "burst_uv": burst,
                     "ratio": burst / base})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)


def edge_times(t: np.ndarray, sig: np.ndarray, tones: np.ndarray) -> pd.DataFrame:
    """Time the onset and offset ramps of each tone by their energy centroids.

    Within the search window the excess energy over baseline is split at its own
    centroid, and each half's centroid times one edge.
    """

    rows = []
    for e in tones:
        a, b = np.searchsorted(t, [e + BASELINE_S[0], e + BURST_S[1] + 0.17])
        lag, power = t[a:b] - e, sig[a:b] ** 2
        noise = power[lag < BASELINE_S[1]].mean()
        window = (lag > BURST_S[0]) & (lag < BURST_S[1])
        weight = np.clip(power - noise, 0, None) * window
        split = np.sum(lag * weight) / weight.sum()
        first, second = weight * (lag < split), weight * (lag >= split)
        rows.append({
            "marker": e,
            "peak_uv": float(np.sqrt(power[window].max())),
            "onset_burst_s": float(np.sum(lag * first) / first.sum()),
            "offset_burst_s": float(np.sum(lag * second) / second.sum()),
        })
    frame = pd.DataFrame(rows)
    # A trial where the plug lost contact carries almost no pickup and its
    # centroids are noise; drop it rather than let it widen the scatter.
    frame["contact"] = frame["peak_uv"] > 0.25 * frame["peak_uv"].median()
    frame["edge_spacing_s"] = frame["offset_burst_s"] - frame["onset_burst_s"]
    # Each centroid sits RAMP_S / 2 inside the tone.
    frame["sound_onset_s"] = frame["onset_burst_s"] - RAMP_S / 2
    frame["sound_onset_from_offset_s"] = frame["offset_burst_s"] + RAMP_S / 2 - TONE_S
    return frame


def summarise(frame: pd.DataFrame, chain_s: float, n_boot: int, rng) -> dict:
    """Latency summary from the trials that had contact."""

    good = frame[frame["contact"]]
    onset = good["sound_onset_s"].to_numpy()
    boot = rng.choice(onset, (n_boot, len(onset)), replace=True).mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])
    return {
        "trials": int(len(frame)),
        "trials_with_contact": int(len(good)),
        "peak_pickup_uv_median": float(good["peak_uv"].median()),
        "onset_burst_after_marker_ms": float(1000 * good["onset_burst_s"].mean()),
        "offset_burst_after_marker_ms": float(1000 * good["offset_burst_s"].mean()),
        "edge_spacing_ms": float(1000 * good["edge_spacing_s"].mean()),
        "edge_spacing_sd_ms": float(1000 * good["edge_spacing_s"].std(ddof=1)),
        "sound_onset_after_marker_ms": float(1000 * onset.mean()),
        "sound_onset_after_marker_sd_ms": float(1000 * onset.std(ddof=1)),
        "sound_onset_after_marker_ci95_ms": (1000 * ci).tolist(),
        "sound_onset_from_offset_edge_ms": float(1000 * good["sound_onset_from_offset_s"].mean()),
        "eeg_chain_latency_ms": float(1000 * chain_s),
        "audio_latency_ms": float(1000 * (onset.mean() - chain_s)),
        "audio_latency_ci95_ms": (1000 * (ci - chain_s)).tolist(),
        "audio_latency_from_offset_edge_ms": float(
            1000 * (good["sound_onset_from_offset_s"].mean() - chain_s)
        ),
    }


# ---------------------------------------------------------------------------
# Reporting


def plot(path: Path, t, sig, tones, frame, summary, channel, ranking) -> None:
    good = frame[frame["contact"]]
    lags = np.arange(BASELINE_S[0], BURST_S[1] + 0.17, 1 / FS)
    rms = np.sqrt((epoch(t, sig, tones, lags) ** 2).mean(axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(1000 * lags, rms, color="0.2", lw=1.2)
    for value, colour, label in (
        (summary["onset_burst_after_marker_ms"], "tab:red", "onset ramp"),
        (summary["offset_burst_after_marker_ms"], "tab:blue", "offset ramp"),
    ):
        ax.axvline(value, color=colour, ls="--", lw=1.2, label=f"{label} {value:.1f} ms")
    ax.axvline(0, color="0.6", lw=1)
    ax.set(xlabel="ms after tone marker", ylabel="RMS pickup (uV)",
           title=f"{channel}: tone-locked pickup (n={len(tones)})")
    ax.legend(fontsize=8)

    ax = axes[1]
    elapsed = (good["marker"] - t[0]) / 60
    ax.scatter(elapsed, 1000 * good["sound_onset_s"], s=12, color="tab:red", alpha=0.7)
    ax.axhline(summary["sound_onset_after_marker_ms"], color="0.3", lw=1.2)
    ax.axhspan(*summary["sound_onset_after_marker_ci95_ms"], color="0.7", alpha=0.4)
    ax.axhline(summary["eeg_chain_latency_ms"], color="tab:green", ls="--", lw=1.2,
               label=f"EEG chain {summary['eeg_chain_latency_ms']:.1f} ms")
    ax.set(xlabel="minutes into the recording", ylabel="marker -> sound onset (ms)",
           title=f"per trial (SD {summary['sound_onset_after_marker_sd_ms']:.1f} ms)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.barh(["EEG chain", "audio output"],
            [summary["eeg_chain_latency_ms"], summary["audio_latency_ms"]],
            color=["tab:green", "tab:orange"])
    ax.errorbar(summary["audio_latency_ms"], 1,
                xerr=[[summary["audio_latency_ms"] - summary["audio_latency_ci95_ms"][0]],
                      [summary["audio_latency_ci95_ms"][1] - summary["audio_latency_ms"]]],
                fmt="none", ecolor="0.2", capsize=4)
    ax.axvline(0, color="0.6", lw=1)
    ax.set(xlabel="ms", title="marker -> sound, split by stage")
    for i, value in enumerate([summary["eeg_chain_latency_ms"], summary["audio_latency_ms"]]):
        ax.text(value, i, f" {value:.1f}", va="center", fontsize=9)

    best = ranking.iloc[0]
    fig.suptitle(
        f"WASAPI headphone timing: sound lands {summary['sound_onset_after_marker_ms']:.1f} ms "
        f"after the marker; audio path {summary['audio_latency_ms']:.1f} ms "
        f"[{summary['audio_latency_ci95_ms'][0]:.1f}, {summary['audio_latency_ci95_ms'][1]:.1f}] "
        f"(plug on {best['channel']}, {best['ratio']:.0f}x baseline)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xdf", default="data/Headphone_Timing/data/headphone_wasapi_T7.xdf")
    parser.add_argument("--channel", default=None,
                        help="sensor the plug rests on (default: the strongest pickup)")
    parser.add_argument("--cross-check", default=None,
                        help="second channel to repeat the measurement on (default: runner-up)")
    parser.add_argument("--eeg-chain", type=float, default=None,
                        help="EEG-chain latency in seconds (default: emotiv.devices.EPOCX.timing)")
    parser.add_argument("--out", type=Path, default=Path("data/Headphone_Timing/headphone_timing"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    t, x, tones, diagnostics = load_run(args.xdf)
    chain, chain_source = eeg_chain_s(args.eeg_chain)
    ranking = pickup_ranking(t, x, tones)
    channel = args.channel or ranking.iloc[0]["channel"]
    cross = args.cross_check or ranking.iloc[1]["channel"]

    sig = highpass(x[:, LABELS.index(channel)])
    frame = edge_times(t, sig, tones)
    summary = summarise(frame, chain, args.bootstrap, rng)

    cross_sig = highpass(x[:, LABELS.index(cross)])
    cross_summary = summarise(edge_times(t, cross_sig, tones), chain, args.bootstrap, rng)

    results = {
        "xdf": args.xdf,
        "channel": channel,
        "eeg_chain_source": chain_source,
        "recording": diagnostics,
        "pickup_ranking": ranking.to_dict("records"),
        "main": summary,
        "cross_check": {"channel": cross, **cross_summary},
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    frame.to_csv(args.out / "trials.csv", index=False)
    plot(args.out / "headphone_timing.png", t, sig, tones, frame, summary, channel, ranking)

    print(f"{args.xdf}")
    print(f"  {diagnostics['samples']} samples, {diagnostics['duration_s']:.1f} s, "
          f"{diagnostics['interpolated_samples']} interpolated, "
          f"arrival residual SD {diagnostics['arrival_residual_sd_ms']:.2f} ms")
    print(f"  plug on {channel} ({ranking.iloc[0]['ratio']:.0f}x baseline); "
          f"runner-up {cross} ({ranking.iloc[1]['ratio']:.0f}x)")
    print(f"  {summary['trials_with_contact']}/{summary['trials']} trials with contact, "
          f"median pickup {summary['peak_pickup_uv_median']:.0f} uV")
    print(f"  edge spacing {summary['edge_spacing_ms']:.1f} ms "
          f"(SD {summary['edge_spacing_sd_ms']:.1f}) vs {1000 * TONE_S:.0f} ms tone")
    print(f"  marker -> sound onset {summary['sound_onset_after_marker_ms']:.1f} ms "
          f"(SD {summary['sound_onset_after_marker_sd_ms']:.1f}, "
          f"95% CI {summary['sound_onset_after_marker_ci95_ms'][0]:.1f}-"
          f"{summary['sound_onset_after_marker_ci95_ms'][1]:.1f})")
    print(f"  EEG chain {summary['eeg_chain_latency_ms']:.1f} ms ({chain_source})")
    print(f"  audio latency {summary['audio_latency_ms']:.1f} ms "
          f"[{summary['audio_latency_ci95_ms'][0]:.1f}, {summary['audio_latency_ci95_ms'][1]:.1f}]"
          f"; offset-edge estimate {summary['audio_latency_from_offset_edge_ms']:.1f} ms")
    print(f"  cross-check on {cross}: audio latency {cross_summary['audio_latency_ms']:.1f} ms")
    print(f"  wrote {args.out}/results.json, trials.csv, headphone_timing.png")


if __name__ == "__main__":
    main()
