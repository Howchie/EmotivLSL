# EPOC X ERP processing pipeline

The reference implementation is `analysis/gng_erp.py`: `load_recording()`, `preprocess()`, `event_table()`, `make_epochs()`, `conventional()` and `n1_marker_check()`. The worked example is the 80/20 auditory Go/NoGo recording `data/GnG/gng.xdf`; its report is `data/GnG/gng_erp/report.md`. Parameters are module constants at the top of the script. Every run writes the values it used to `results.json → preprocessing.parameters`.

The pipeline exists because the first analysis of this recording (codex, since removed) produced uneven baselines and "no P3". The causes, in order of impact:

1. **Horizontal eye movements were left in.** Its ICA removed only the blink component. The saccade component (F7 vs F8, ~30% of variance) stayed in the data and produced a slow F7/F8 wave of up to 12 µV that differed between conditions.
2. **Average reference.** The 14 EPOC X sensors form a ring around the head. A broad, vertex-centred positivity such as the P3 reaches all of them, so the average reference subtracts most of it. The remainder shows up as a spurious negativity at P7/P8/O1/O2.
3. **Lenient rejection** (150 µV on 0.1–30 Hz epochs). Electrode pops and movement bursts went into the averages.
4. **ROI and window chosen without looking at the topography.** It used P7/P8 at 300–600 ms, which is after the P3 peak and at sites near the reference sensors.

---

## 0. Loading and event timing

| Step | What | Why |
|---|---|---|
| Sample grid | `epocx_sanity.epocx_grid`: rebuild the regular grid from the packet counter and fit arrival times against it. Lost samples are interpolated and flagged `FILLED`, which becomes a `BAD_fill` annotation. | LSL arrival timestamps jitter, and pyxdf dejittering bridges gaps incorrectly (commit 184a6d7). |
| Markers | Deduplicate identical (time, value) pairs across marker streams. Match Go/NoGo markers to CSV rows in order, and assert that stimulus identity and counts agree. | LabRecorder can record the same outlet twice. |
| Tone onset on the EEG timeline | marker + shift. Adding to the event time is the same as subtracting the delay from ERP latencies. | EPOC X samples are stamped **70.9 ms** (DRT test) after the scalp event. That is the only latency verified directly. |
| RT from the sound | (`choice_resp.rt` − `soa`) + ½ frame + EEG chain − shift | There is no keypress marker. PsychoPy resets the keyboard clock on the routine's first flip, 0–1 frame after the tone was scheduled. Keyboard latency is unmeasured. |
| **Timing check (every session)** | A reference-free N1: frontal (F3 F4 FC5 FC6) minus inferior (T7 T8 P7 P8), frequent tones, marker-locked trough (`n1_marker_check`, `erp.png` row 2). | This is the only in-session check of the audio path. Compare the trough with earlier sessions that used the same audio stack. |

**Audio path and the current shift.**

- **Oddball runs (PTB audio, `play(when=…)`).** The N1 fell 219 ms after the marker through the 3.5 mm headphone jack and 305 ms through the USB speaker. The jack loopback measured 54.7 ms of audio output latency.
- **`gng.xdf`.** PsychoPy's PTB backend was broken by an update, so this session used a custom `sounddevice` stream (`data/GnG/p3.psyexp`) through the **headphone jack**.
  - That stream timestamps each marker at PortAudio's *estimated* DAC time (`outputBufferDacTime`). The estimate typically leaves out driver and hardware buffering.
  - The N1 trough came 281 ms [273, 281] after the marker, about **60 ms later than the PTB headphone run**. The PTB loopback latency therefore does not transfer to this stack.
- **What the script applies now.** `gng_erp.py` still shifts by **210.6 ms** (70.9 EEG chain + 54.7 jack + 85.0 speaker−jack). That is the PTB + USB-speaker estimate, which is wrong for this session. The N1 lands at ~70 ms after "onset".
- **Intended change (not yet applied).** Shift by the verified 70.9 ms EEG chain only, and accept that unmeasured audio delay remains in the latencies. On that basis the `gng.xdf` N1 would sit at ~210 ms and every other latency would move +140 ms. Amplitudes, topographies and statistics are unaffected.
- A loopback recorded through the `sounddevice` stream is the only way to measure its audio latency.

## 1. Bad channels

Bad channels are **dropped, not interpolated**. Spherical-spline interpolation from 13 sparse sensors adds little information and blurs the ROIs. ROIs use the remaining channels (`roi_label()` prints which).

A channel is bad if any of these holds:

- **Known hardware problem** (`MANUAL_BADS`, with the reason recorded). For `gng.xdf`, F4 had an old saline pad.
- **Flat or noisy** (`flex_sanity.channel_noise`): RMS below 1 µV at 1–20 Hz, or 20–40 Hz power more than 4× the median channel.
- **Residual share**: after ICA, it exceeds `RESIDUAL_UV` in more than 10% of 1-s task windows.

**Electrode-pop diagnostic** (`electrode_pops`, `qc_artifacts.png` bottom row). It counts peaks of |channel − median of the other channels| above 60 µV at 1–20 Hz, with ±0.5 s around blinks masked. A pad that is drying out gives a stereotyped waveform: a ~200 ms ramp, a sharp spike, then a ~50 µV offset that recovers over ~2 s. These pops recur at shrinking intervals; F4 in `gng.xdf` went from about 45 s to 30 s apart, 1.6 pops/min. Any channel above ~0.5 pops/min should be re-wetted or have its pad replaced before the next recording.

## 2. Gross artifacts (excluded from the ICA fit only)

1–30 Hz data, 1-s windows with a 0.5-s step, peak-to-peak above **300 µV** on any good channel, padded by 0.25 s before and 0.5 s after. These are movement bursts and pops, which would pull ICA components toward themselves. Blinks (≤ ~250 µV at AF3/AF4) stay in, so ICA can learn them. In `gng.xdf` this excluded 1.5% of task time.

## 3. ICA: blinks **and** saccades

- FastICA on **1–30 Hz** good channels in the recorded reference, with n_components = n_good − 1 and `random_state=7`.
- EOG proxies (the EPOC X has no EOG channels): **VEOG** = mean(AF3, AF4); **HEOG** = F7 − F8. Both are band-passed to 1–10 Hz.
- A component is removed when its source correlates **|r| ≥ 0.5** with either proxy.
  - In `gng.xdf`: IC1 blink (r_VEOG = 0.75) and IC2 horizontal saccade (r_HEOG = 0.72). The next best correlation was 0.25.
- ICA is applied to the 0.1–20 Hz data. The unmixing is spatial, so slow potentials from sustained gaze offsets are removed along with the saccade steps.
- **Do not remove the uniform "common-mode" component**, the one with the same sign on every channel (IC0 here). In the recorded reference it carries broad brain activity, including the P3.
- **Checks** (`qc_artifacts.png` rows 2–3):
  - The blink-locked average at AF3/AF4 drops from ~180 µV to the level of other channels.
  - Single-trial F7−F8 epochs show no condition-locked slow wave.

## 4. ERP filtering

**0.1–20 Hz**, zero-phase FIR (MNE defaults), applied to the continuous data before epoching.

- The 0.1 Hz high-pass preserves the P3 and slow waves. Do not raise it without checking for distortion.
- The 20 Hz low-pass matters at a 128 Hz sampling rate: 20–30 Hz content on the EPOC X is mostly EMG and sensor noise, which made the baselines "wiggly" with ~70 trials. Removing it does not affect N1 latency.

## 5. Residual artifact rejection

On the ICA-cleaned 0.1–20 Hz data: 1-s windows with a 0.5-s step whose peak-to-peak exceeds **100 µV** on any good channel become `BAD_residual`, padded 0.25 s before and 0.5 s after to cover pop recovery.

- The spans are written to `bad_segments.csv`.
- Epochs overlapping a BAD span are dropped (`reject_by_annotation=True`).
- In `gng.xdf`, with F4 dropped, 3.9% of task time was excluded. 70/75 Go hits and 303/320 correct NoGo epochs were kept.
- The 100 µV threshold sits above the 99th percentile of clean-channel 1-s peak-to-peak after ICA (75–95 µV). Recheck that percentile (`sliding_p2p`) for a new headset or session.

## 6. Epochs, baseline, noise

- Epochs run from −300 to 1000 ms around the shifted onset, with baseline −200 to 0 ms. The minimum SOA in `gng.xdf` is 1.26 s, so the next tone never falls inside the epoch.
- **Is a non-flat baseline a problem?** Compare the baseline RMS with the RMS of the **± average** (half the trials sign-flipped, `plus_minus_rms`), which is pure noise at that trial count. If they match, the wiggle is noise and only more trials or less noise will help. If the baseline is clearly larger, look for overlap, anticipation or artifacts.
  - `gng.xdf`, frontal Go (n = 70): 0.71 µV vs 0.65 µV noise.
  - NoGo baselines run ~1.4× the ± average, but below 0.5 µV.
- Show 95% trial-bootstrap bands (2000 resamples) for every condition waveform and difference wave.

## 7. Reference: recorded (CMS) only

- The EPOC X records against its CMS/DRL sensors behind the ears (mastoid area). **Analyse in that reference and do not re-reference to the average.**
  - On this ring-shaped 14-channel montage, a broad component reaches every sensor.
  - At the Go − NoGo peak (250 ms) in `gng.xdf`, every channel was positive and the channel mean was +3.8 µV. An average reference would subtract that mean and flip P7/P8/O1/O2 negative (`results.json → go_minus_nogo_at_frontal_p3_peak`).
- The most superior sensors, **F3/F4/FC5/FC6**, are the best available proxy for Cz/Pz; there are no midline sites. P7/P8/O1/O2 sit low and near the reference, so broad components are small there.
- For the same reason a topography on this montage cannot distinguish P3a from P3b. "Frontal-maximal" means "maximal at the most superior sensors".

## 8. Statistics

- **Cluster permutation test** (`flex_sanity.cluster_test`): spatio-temporal, Welch t, cluster-forming threshold p < .01, 2000 permutations, over 0–1000 ms.
- **Peak latencies** with trial-bootstrap 95% CIs.
- **Per-block replication** of the key effect (`block_replication`, mean over 200–350 ms).
- This is a single participant, so all inference is within-session. Picking windows *after* looking at the topography is acceptable for descriptive reporting only, and should be stated.

## 9. Keypress / motor activity

Keypress activity is **not modelled**. A time-expanded regression on the continuous data, with tone kernels plus a keypress kernel placed from the CSV RTs, was tried on `gng.xdf` and dropped:

- **Overlap from neighbouring trials was negligible.** The tone-only regression reproduced the plain averages.
- **The keypress could not be separated.** With RTs tightly clustered (IQR 332–380 ms from onset), a parameter-recovery simulation showed the keypress model was unbiased but had a ~±5 µV 95% band at 200–600 ms, as large as the P3.
- **The recording can't settle it.** It cannot tell how much of the Go positivity is motor-related, beyond the fact that it peaks ~100 ms before the median keypress.

If motor activity has to be excluded, change the task rather than the analysis:

- counting blocks with no keypress;
- reversed mapping (press on the frequent tone);
- a delayed response cue (≥ 800 ms).

## 10. Per-session QC checklist

- [ ] Grid rebuild: missing samples, segment rates, arrival residuals.
- [ ] Bad channels: manual, noise and residual share; pops per minute per channel.
- [ ] ICA: the removed ICs are blink (AF3/AF4 frontal) and saccade (F7/F8 antisymmetric) topographies with |r| ≥ 0.5, and no other IC comes close.
- [ ] Blink-locked average flat after ICA; single-trial F7−F8 image free of condition-locked drift.
- [ ] Fraction of task data excluded, and epochs kept per condition.
- [ ] Baseline RMS vs ± average RMS.
- [ ] Audio stack recorded (PTB vs sounddevice, jack vs speaker); N1 timing check against a previous session on the same stack.

## Parameters

| Constant | Value | Used for |
|---|---|---|
| `ICA_BAND` | 1–30 Hz | ICA fit |
| `ERP_BAND` | 0.1–20 Hz | ERPs |
| `GROSS_UV` | 300 µV | 1-s p2p, excluded from the ICA fit |
| `RESIDUAL_UV` | 100 µV | 1-s p2p after ICA, BAD_residual |
| `WIN_S`, `STEP_S` | 1.0 s, 0.5 s | artifact windows |
| `PAD_S` | 0.25 s before, 0.5 s after | around flagged windows |
| `EOG_R` | 0.5 | IC vs VEOG/HEOG correlation |
| `CHANNEL_BAD_SHARE` | 0.10 | residual share that makes a channel bad |
| `POP_UV` | 60 µV | pop diagnostic |
| `EPOCH`, `BASELINE` | −0.3–1.0 s, −0.2–0 s | epochs |
| `N_BOOT` | 2000 | trial bootstrap |
