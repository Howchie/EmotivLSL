#!/usr/bin/env python3
"""Separate audio-output latency from EEG-chain latency for the auditory oddball.

Inputs, all recorded on the same EPOC X (firmware 0x740) with the same PsychoPy script:

* ``epochx_0740.xdf``: tones through the USB speaker;
* ``headphones.xdf``: tones through headphones on the 3.5 mm jack, resting on the headset;
* ``DRT_Timing/drt_timing/results.json``: EPOC X EEG-chain latency from the tactile test;
* ``jack.xdf``: the same oddball tones with the headphone plug's tip held on O1.

The EEG chain is common to both oddball runs, so the shift of the tone-locked
N1/P2 between them is the difference in audio latency.  The marker-to-N1 time
of each run is EEG latency + audio latency + the N1's own latency after sound
onset.

The jack recording measures the headphone audio latency directly.  The 1 kHz
tone is far above the headset's passband, but its onset and offset leak
aliased energy into the 128 Hz output.  PsychoPy fades generated tones in and
out over 5 ms, so each edge's energy burst is centred 2.5 ms inside the 100 ms
tone.  The onset burst's time after the marker, less those 2.5 ms and the
EEG-chain latency, is the jack's audio latency (the offset burst gives a second
estimate).  The speaker's latency then follows from the N1 shift, and what
remains of each marker-to-N1 time is the N1's latency after the sound.  The
headphone run is also checked for direct pickup of the tones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyxdf
from scipy.signal import butter, sosfiltfilt

import drt_timing as dt
import epocx_sanity as ex
import flex_sanity as fx
from flex_sanity import FS, SEED, Run

mne.set_log_level("ERROR")

JACK_XDF = "data/Oddball/jack.xdf"
JACK_CHANNEL = "O1"
TONE_S = 0.100
RAMP_S = 0.005  # PsychoPy's fade-in/out on generated tones
RUNS = {
    "USB speaker": "data/Oddball/data/epochx_0740.xdf",
    "headphones": "data/Oddball/data/headphones.xdf",
}
DRT_RESULTS = "data/DRT_Timing/drt_timing/results.json"
EPOCX_RESULTS = "data/Oddball/data/epocx_sanity/results.json"
N1_ASSUMED_S = (0.090, 0.100, 0.120)  # low / typical / late N1 peak after sound onset


def load_oddball(key: str, path: str) -> Run:
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    grid = ex.epocx_grid(by_name("Epoc X")[0], by_name("Epoc X Packet Diagnostics")[0])
    t0 = grid["t"][0]
    t = grid["t"] - t0
    rows = sorted({(float(ts) - t0, str(v[0])) for s in by_name("PsychoPy Markers") for ts, v in zip(s["time_stamps"], s["time_series"])})
    markers = pd.DataFrame(rows, columns=["time", "value"])
    events = []
    for time, value in rows:
        prefix, _, stim = value.partition("-")
        if prefix == "Oddball" and stim in ("standard", "oddball"):
            sample = int(np.argmin(np.abs(t - time)))
            events.append({"time": time, "prefix": prefix, "block": "passive", "stim": stim, "sample": sample})
    diag = {"FILLED": grid["missing"].astype(float), "RESET_FLAG": np.zeros(len(t))}
    run = Run(key, t, grid["x"], ex.LABELS, diag, markers, None, None)
    run.tone_events = pd.DataFrame(events)
    return run


def pickup_check(run: Run) -> dict:
    """Sharp tone-locked transients above 15 Hz, where the slow ERP contributes little."""

    hp = sosfiltfilt(butter(4, 15, btype="high", fs=FS, output="sos"), run.x_uv, axis=0)
    ev = run.t[run.tone_events["sample"].to_numpy()]
    grid = np.arange(-0.1, 0.4, 0.0005)
    out = {}
    for name, sig in [("mean of all channels", hp.mean(axis=1)), ("T8 - T7", hp[:, 9] - hp[:, 4])]:
        rel, val, _ = dt.pooled(run.t, sig, run.filled, ev, pre=0.3, post=0.45)
        tpl = dt.kernel_template(rel, val, grid)
        base = tpl[grid < 0]
        z = (tpl - base.mean()) / base.std()
        post = grid > 0
        out[name] = {"max_abs_z_after_tone": float(np.abs(z[post]).max()),
                     "at_s": float(grid[post][np.argmax(np.abs(z[post]))]),
                     "max_abs_z_baseline": float(np.abs(z[~post]).max())}
    return out


def jack_edges(eeg_latency: float, n_boot: int, rng) -> dict:
    """Time the tone's onset and offset edges from the energy they alias into the plugged channel."""

    streams, _ = pyxdf.load_xdf(JACK_XDF, synchronize_clocks=True, dejitter_timestamps=False)
    by_name = lambda name: [s for s in streams if s["info"]["name"][0] == name]
    grid = ex.epocx_grid(by_name("Epoc X")[0], by_name("Epoc X Packet Diagnostics")[0])
    t = grid["t"]
    sig = sosfiltfilt(butter(4, 5, btype="high", fs=FS, output="sos"), grid["x"][:, ex.LABELS.index(JACK_CHANNEL)])
    markers = [float(ts) for s in by_name("PsychoPy Markers") for ts, v in zip(s["time_stamps"], s["time_series"])
               if str(v[0]).startswith("Oddball-")]
    rows = []
    for e in markers:
        a, b = np.searchsorted(t, e - 0.3), np.searchsorted(t, e + 0.5)
        r, p = t[a:b] - e, sig[a:b] ** 2
        noise = p[r < -0.02].mean()
        sel = (r > 0.03) & (r < 0.33)
        w = np.clip(p - noise, 0, None) * sel
        centre = np.sum(r * w) / w.sum()
        first, second = w * (r < centre), w * (r >= centre)
        rows.append({"marker": e, "peak_uv": float(np.sqrt(p[sel].max())),
                     "onset_burst_s": float(np.sum(r * first) / first.sum()),
                     "offset_burst_s": float(np.sum(r * second) / second.sum())})
    frame = pd.DataFrame(rows)
    # Trials where the plug slipped off the sensor carry almost no pickup.
    frame["contact"] = frame["peak_uv"] > 0.25 * frame["peak_uv"].median()
    good = frame[frame["contact"]]
    onset = good["onset_burst_s"].to_numpy() - RAMP_S / 2
    offset = good["offset_burst_s"].to_numpy() + RAMP_S / 2 - TONE_S
    boot = np.array([rng.choice(onset, len(onset)).mean() for _ in range(n_boot)])
    return {
        "trials": int(len(frame)), "trials_with_contact": int(len(good)),
        "onset_burst_after_marker_s": float(good["onset_burst_s"].mean()),
        "offset_burst_after_marker_s": float(good["offset_burst_s"].mean()),
        "edge_spacing_s": float((good["offset_burst_s"] - good["onset_burst_s"]).mean()),
        "onset_burst_trial_sd_s": float(good["onset_burst_s"].std(ddof=1)),
        "offset_burst_trial_sd_s": float(good["offset_burst_s"].std(ddof=1)),
        "sound_onset_in_eeg_time_from_onset_edge_s": float(onset.mean()),
        "sound_onset_in_eeg_time_from_offset_edge_s": float(offset.mean()),
        "jack_audio_latency_s": float(onset.mean() - eeg_latency),
        "jack_audio_latency_ci95_s": (np.percentile(boot, [2.5, 97.5]) - eeg_latency).tolist(),
        "jack_audio_latency_from_offset_edge_s": float(offset.mean() - eeg_latency),
        "_frame": frame, "_t": t, "_sig": sig,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/Oddball/data/audio_latency"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    eeg_latency = json.loads(Path(DRT_RESULTS).read_text())["eeg_chain_latency_s"]["mean_of_edges"]
    epocx_results = json.loads(Path(EPOCX_RESULTS).read_text())
    flex_lag = epocx_results["latency_comparison"]["epocx_minus_flex_lag_s"]
    flex_lag_ci = epocx_results["latency_comparison"]["lag_ci95_s"]

    results: dict = {"epocx_eeg_chain_latency_s": eeg_latency, "runs": {}}
    epochs, waves = {}, {}
    for key, path in RUNS.items():
        print(key)
        run = load_oddball(key, path)
        raw = fx.make_raw(run)
        noise = fx.channel_noise(raw)
        bads = noise.loc[noise["bad"], "channel"].tolist()
        ep, prep = fx.erp_epochs(run, raw, bads, eog_channels=ex.EOG, codes=ex.CODES)
        std = ep[(ep.metadata["stim"] == "standard").to_numpy()]
        lat = ex.n1_p2(std, ex.FRONTAL_ROI, ex.INFERIOR_ROI, rng)
        waves[key] = lat.pop("_wave")
        epochs[key] = std
        results["runs"][key] = {
            "missing_samples": int(run.filled.sum()), "bad_channels": bads, "erp_preprocessing": prep,
            "standards_kept": int(len(std)), **lat,
            "marker_to_n1_minus_eeg_latency_s": lat["n1_latency_s"] - eeg_latency,
        }
        if key == "headphones":
            results["runs"][key]["direct_pickup"] = pickup_check(run)

    times = epochs["USB speaker"].times
    lag = ex.lag_between(times, waves["USB speaker"], waves["headphones"], args.bootstrap, rng)
    diff = -lag["lag_s"]  # speaker audio latency minus headphone audio latency
    results["speaker_minus_headphones_audio_latency_s"] = {
        "from_waveform_alignment": diff, "ci95": [-lag["ci95_s"][1], -lag["ci95_s"][0]], "corr_at_lag": lag["corr_at_lag"],
        "from_n1_troughs": results["runs"]["USB speaker"]["n1_latency_s"] - results["runs"]["headphones"]["n1_latency_s"],
    }

    # Latency budget: marker-to-N1 = EEG latency + audio latency + N1 latency after sound onset.
    budget = {}
    for n1 in N1_ASSUMED_S:
        row = {}
        for key in RUNS:
            row[f"{key} audio latency"] = results["runs"][key]["n1_latency_s"] - eeg_latency - n1
        # Flex tones went through the USB speaker; its EEG chain = EPOC X chain - (EPOC X - Flex lag).
        row["Flex EEG chain latency"] = eeg_latency - flex_lag
        budget[f"N1 at {1000 * n1:.0f} ms"] = row
    results["latency_budget_s"] = budget
    results["flex_eeg_chain_latency_s"] = {"estimate": eeg_latency - flex_lag,
                                           "ci95": [eeg_latency - flex_lag_ci[1], eeg_latency - flex_lag_ci[0]]}
    results["headphone_n1_upper_bound_s"] = results["runs"]["headphones"]["n1_latency_s"] - eeg_latency

    print("jack")
    jack = jack_edges(eeg_latency, args.bootstrap, rng)
    results["jack"] = {k: v for k, v in jack.items() if not k.startswith("_")}
    a_jack = jack["jack_audio_latency_s"]
    diff_ci = results["speaker_minus_headphones_audio_latency_s"]["ci95"]
    results["measured_budget_s"] = {
        "epocx_eeg_chain": eeg_latency,
        "jack_audio": a_jack,
        "usb_speaker_audio": a_jack + diff,
        "usb_speaker_audio_range": [a_jack + diff_ci[0], a_jack + diff_ci[1]],
        "n1_after_sound_headphone_run": results["runs"]["headphones"]["n1_latency_s"] - eeg_latency - a_jack,
        "n1_after_sound_speaker_run": results["runs"]["USB speaker"]["n1_latency_s"] - eeg_latency - (a_jack + diff),
        "flex_eeg_chain": eeg_latency - flex_lag,
    }

    # Figure.
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, :2])
    colors = {"USB speaker": "tab:red", "headphones": "tab:blue"}
    for key, wave in waves.items():
        mean = wave.mean(axis=0)
        se = wave.std(axis=0, ddof=1) / np.sqrt(len(wave))
        ax.plot(times, mean, color=colors[key], lw=1.8, label=f"{key} (n={len(wave)}), N1 {1000 * results['runs'][key]['n1_latency_s']:.0f} ms")
        ax.fill_between(times, mean - se, mean + se, color=colors[key], alpha=0.2)
    ax.plot(times + lag["lag_s"], waves["USB speaker"].mean(axis=0), color="tab:red", ls="--", lw=1,
            label=f"USB speaker shifted {1000 * lag['lag_s']:+.0f} ms")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(eeg_latency, color="0.4", ls=":", lw=1)
    ax.text(eeg_latency + 0.005, ax.get_ylim()[0] * 0.9, "EEG chain\n(DRT)", fontsize=8, color="0.3")
    ax.set_xlabel("s from tone marker")
    ax.set_ylabel("µV")
    ax.set_title("EPOC X standards, (F3/F4/FC5/FC6) − (T7/T8/P7/P8)", fontsize=10)
    ax.legend(fontsize=8)
    for col, key in enumerate(RUNS):
        ax = fig.add_subplot(gs[0, 2 + col])
        epochs[key].average().plot_topomap(times=[results["runs"][key]["n1_latency_s"]], average=0.03, axes=ax,
                                          show=False, colorbar=False, sensors=True)
        ax.set_title(f"{key}: N1 at {1000 * results['runs'][key]['n1_latency_s']:.0f} ms", fontsize=9)
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(-1000 * lag["_boot"], bins=40, color="tab:gray")
    ax.axvline(1000 * diff, color="k")
    ci = results["speaker_minus_headphones_audio_latency_s"]["ci95"]
    ax.set_xlabel("speaker − headphone audio latency (ms)")
    ax.set_title(f"{1000 * diff:.0f} ms [{1000 * ci[0]:.0f}, {1000 * ci[1]:.0f}]", fontsize=9)
    ax = fig.add_subplot(gs[1, 1:])
    mb = results["measured_budget_s"]
    rows = [("EPOC X + headphone jack", mb["jack_audio"], mb["n1_after_sound_headphone_run"]),
            ("EPOC X + USB speaker", mb["usb_speaker_audio"], mb["n1_after_sound_speaker_run"])]
    for y, (label, audio, n1) in enumerate(rows):
        ax.barh(y, 1000 * eeg_latency, color="0.6", label="EEG chain (DRT tactile test)" if y == 0 else None)
        ax.barh(y, 1000 * audio, left=1000 * eeg_latency, color="tab:orange",
                label="audio output (jack loopback; speaker = jack + N1 shift)" if y == 0 else None)
        ax.barh(y, 1000 * n1, left=1000 * (eeg_latency + audio), color="tab:green", alpha=0.6,
                label="N1 after sound onset (what remains)" if y == 0 else None)
        for left, width in [(0, eeg_latency), (eeg_latency, audio), (eeg_latency + audio, n1)]:
            ax.text(1000 * (left + width / 2), y, f"{1000 * width:.0f}", ha="center", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("ms after tone marker")
    ax.set_title("Measured marker-to-N1 budget", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    frame, jt, jsig = jack["_frame"], jack["_t"], jack["_sig"]
    good = frame[frame["contact"]]
    rel, val, _ = dt.pooled(jt, jsig, np.zeros(len(jt), bool), good["marker"].to_numpy(), pre=0.3, post=0.45)
    env_grid = np.arange(-0.05, 0.35, 0.0005)
    env = np.sqrt(np.clip(dt.kernel_template(rel, val ** 2, env_grid), 0, None))
    ax = fig.add_subplot(gs[2, :3])
    ax.scatter(1000 * rel, val, s=3, color="0.6", alpha=0.4, label=f"{JACK_CHANNEL} samples, trials with contact (>5 Hz)")
    ax.plot(1000 * env_grid, env, color="k", lw=1.5, label="pooled RMS envelope")
    for col_name, color, label in [("onset_burst_s", "tab:blue", "onset edge"), ("offset_burst_s", "tab:purple", "offset edge")]:
        ax.axvline(1000 * good[col_name].mean(), color=color, ls="--", label=f"{label} burst {1000 * good[col_name].mean():.1f} ms")
    ax.axvline(1000 * eeg_latency, color="0.3", ls=":", label="EEG chain alone")
    ax.set_xlim(-50, 350)
    ax.set_xlabel("ms after tone marker")
    ax.set_ylabel("µV")
    ax.set_title(f"Jack loopback on {JACK_CHANNEL}: the 1 kHz tone's edges alias into the EEG "
                 f"({jack['trials_with_contact']}/{jack['trials']} trials with contact)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[2, 3])
    ax.hist(1000 * good["onset_burst_s"], bins=15, color="tab:blue", alpha=0.7,
            label=f"onset (SD {1000 * jack['onset_burst_trial_sd_s']:.1f} ms)")
    ax.hist(1000 * (good["offset_burst_s"] - TONE_S + RAMP_S), bins=15, color="tab:purple", alpha=0.5,
            label=f"offset − 95 ms (SD {1000 * jack['offset_burst_trial_sd_s']:.1f} ms)")
    ax.set_xlabel("ms after marker")
    ax.set_title("Per-trial edge times", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(args.out / "audio_latency.png", dpi=130)
    plt.close(fig)

    (args.out / "results.json").write_text(json.dumps(fx.to_jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps(fx.to_jsonable({k: v for k, v in results.items() if k not in ("runs", "latency_budget_s")}), indent=2))
    for key in RUNS:
        r = results["runs"][key]
        print(key, {k: r[k] for k in ("standards_kept", "n1_latency_s", "n1_ci95", "p2_latency_s", "p2_ci95", "n1_amp_uv", "p2_amp_uv", "bad_channels")},
              r.get("direct_pickup"), r["erp_preprocessing"])


if __name__ == "__main__":
    main()
