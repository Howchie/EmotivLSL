# EPOC X ERP processing pipeline

The reference implementation is the **`analysis/emotiv` package**: `loading.load()`, `events.events_from_markers()`, `preprocess.clean()` and `erp.erp_epochs()`. An analysis script supplies only what is specific to its task; `analysis/gng_erp.py` is the worked example, on the 80/20 auditory Go/NoGo recording `data/GnG/gng.xdf`.

```python
import emotiv as em

run      = em.load("data/GnG/gng.xdf")
events   = em.events_from_markers(run, em.split_condition({"GnG": "gng"}, ("high", "low")))
cleaned  = em.clean(run, task_mask=..., manual_bads={"F4": "old saline pad"})
epochs, timing = em.erp_epochs(run, cleaned, events)
```

**This document describes the EPOC X profile only.** The Flex 1.0 is different hardware — different noise floor, dynamic range, decoder and montage — and is processed differently: its recorded CMS reference, interpolated bad channels, and a 0.1–30 Hz ERP band. Those choices live in `devices.FLEX_PROCESSING` and must not be copied from here. `emotiv/preprocess.py` and `emotiv/erp.py` supply only the mechanisms; every band, threshold, reference and policy comes from `run.dev.processing`.

The differences that matter most:

| | EPOC X | Flex 1.0 |
|---|---|---|
| Reference | recorded (CMS) | recorded (CMS) |
| Bad channels | dropped | interpolated |
| ERP band | 0.1–20 Hz | 0.1–30 Hz |
| Epoch | −0.3–1.0 s | −0.2–1.0 s |
| Residual p2p | 100 µV | 120 µV (re-derived) |
| Fill pad | 0.25/0.5 s | 0.25/0.5 s (no DC-restore tail) |
| HEOG pair | F7−F8 | F7−F8 default; verify per session |

The Flex profile is marked validated: its reference, artifact thresholds and rail/fill handling were re-derived on the available Flex recordings. A new montage should still provide its own HEOG pair; `eog_proxies` warns when that pair is unavailable rather than silently skipping saccade removal.

Pipeline parameters are overridable per call; every run writes the values it used to `results.json → preprocessing.parameters`, including which device profile and which EOG proxies were used. Per-headset constants, including the measured chain latency, are in `emotiv/devices.py`.

The pipeline exists because the first analysis of this recording (codex, since removed) produced uneven baselines and "no P3". The causes, in order of impact:

1. **Horizontal eye movements were left in.** Its ICA removed only the blink component. The saccade component (F7 vs F8, ~30% of variance) stayed in the data and produced a slow F7/F8 wave of up to 12 µV that differed between conditions.
2. **Average reference.** The 14 EPOC X sensors form a ring around the head. A broad, vertex-centred positivity such as the P3 reaches all of them, so the average reference subtracts most of it. The remainder shows up as a spurious negativity at P7/P8/O1/O2.
3. **Lenient rejection** (150 µV on 0.1–30 Hz epochs). Electrode pops and movement bursts went into the averages.
4. **ROI and window chosen without looking at the topography.** It used P7/P8 at 300–600 ms, which is after the P3 peak and at sites near the reference sensors.

---

## 0. Loading and event timing

| Step | What | Why |
|---|---|---|
| Sample grid | `emotiv.loading.epocx_grid`: rebuild the regular grid from the packet counter and fit arrival times against it. The whole-recording fitted rate is carried on `Run.sfreq_hz`; lost samples are interpolated and flagged `FILLED`, which becomes a `BAD_fill` annotation. | LSL arrival timestamps jitter, and pyxdf dejittering bridges gaps incorrectly (commit 184a6d7). |
| Markers | Deduplicate identical (time, value) pairs across marker streams, then shift **all** markers by the fixed hardware chain latency. Match task markers to CSV rows in order, and assert that stimulus identity and counts agree. | LabRecorder can record the same outlet twice; all downstream block/rest/task analyses then share one raw-EEG time base. |
| Sound onset on the EEG timeline | Hardware latency is applied once by the loader. `erp_epochs` applies only an optional session-specific audio-path residual. | EPOC X samples are stamped one **chain latency** after the scalp event (`devices.EPOCX.timing`). |
| RT | (`choice_resp.rt` − `soa`), 0–1 frame short, **from the marker**. | There is no keypress marker, and PsychoPy resets the keyboard clock on the routine's first flip. Converting to RT-from-sound needs that session's audio latency; the chain latency does **not** enter, because it delays the EEG, not the keypress. |

**Timing.** Loading shifts every marker by `run.dev.timing.latency_s` and records it as `run.marker_shift_s`, so t=0 on every epoch is the physical stimulus. `erp_epochs` records and applies only any explicit additional audio-path correction. Updating `emotiv/devices.py` after a hardware test re-run updates every analysis.

**Audio path.** Chain latency and audio-output latency are separate delays and are measured separately:

- **`analysis/drt_timing.py`** times the DRT's electrical transient against its corrected marker: the **EEG chain**. The current P7 LED run gives 69.8 ms from the data-localized onset slope; the trailing slope is at 176.3 ms and implies a 106.5-ms pulse width. This is consistent with the 71.1-ms marker-to-sound onset in the headphone loopback.
- **`analysis/headphone_timing.py`** injects the tone electrically into a sensor and times it the same way, then subtracts the chain: the **audio output**. On the WASAPI exclusive-mode `sounddevice` stream this came out at 0.2 ms [−0.6, 0.9], i.e. negligible, so for sessions on that stack the chain latency alone is the whole correction.
- **`gng.xdf` predates that fix.** It used an earlier `sounddevice` path whose audio latency was never measured and is suspected to be ~60 ms, from the N1 arriving that much later than the PTB headphone runs. Its latencies therefore still carry an unmodelled audio delay; amplitudes, topographies and statistics are unaffected.
- **Per-session check.** The reference-free N1 (frontal minus inferior, frequent tones, `erp.n1_p2`) is the in-session sanity check on the audio path. Compare it against a previous session on the same audio stack.

Task-time QC uses the union of robustly paired `BlockStart-*`/`BlockEnd-*`
intervals rather than one first-to-last envelope, so between-block rest does not
inflate artifact percentages. Incomplete or repeated pairs are reported in the
task behaviour diagnostics; complete intervals remain usable.

## 1. Bad channels

Bad channels are **dropped, not interpolated**. Spherical-spline interpolation from 13 sparse sensors adds little information and blurs the ROIs. ROIs use the remaining channels; `Cleaned.good` and each result's `roi` field record which.

A channel is bad if any of these holds:

- **Known hardware problem** (`MANUAL_BADS`, with the reason recorded). For `gng.xdf`, F4 had an old saline pad.
- **Flat or noisy** (`emotiv.preprocess.channel_noise`): RMS below 1 µV at 1–20 Hz, or 20–40 Hz power more than 4× the median channel.
- **Residual share**: after ICA, it exceeds `RESIDUAL_UV` in more than 10% of 1-s task windows.

**Electrode-pop diagnostic** (`emotiv.preprocess.electrode_pops`). It counts peaks of |channel − median of the other channels| above 60 µV at 1–20 Hz, with ±0.5 s around blinks masked. A pad that is drying out gives a stereotyped waveform: a ~200 ms ramp, a sharp spike, then a ~50 µV offset that recovers over ~2 s. These pops recur at shrinking intervals; F4 in `gng.xdf` went from about 45 s to 30 s apart, 1.6 pops/min. Any channel above ~0.5 pops/min should be re-wetted or have its pad replaced before the next recording.

## 2. Gross artifacts (excluded from ICA and final epoching)

1–30 Hz data, 1-s windows with a 0.5-s step, peak-to-peak above **300 µV** on any good channel, padded by 0.25 s before and 0.5 s after. These are movement bursts and pops, which would pull ICA components toward themselves. Blinks (≤ ~250 µV at AF3/AF4) stay in, so ICA can learn them. The resulting `BAD_gross` annotations are retained on the final raw object, so overlapping epochs are rejected as well as excluded from the ICA fit.

## 3. ICA: blinks **and** saccades

- FastICA on **1–30 Hz** good channels in the recorded reference, with n_components = n_good − 1 and `random_state=7`.
- EOG proxies (the EPOC X has no EOG channels): **VEOG** = mean(AF3, AF4); **HEOG** = F7 − F8. Both are band-passed to 1–10 Hz.
- A component is removed when its source correlates **|r| ≥ 0.5** with either proxy.
  - In `gng.xdf`: IC1 blink (r_VEOG = 0.77) and IC2 horizontal saccade (r_HEOG = 0.72). The next best correlation was 0.25.
- ICA is applied to the 0.1–20 Hz data. The unmixing is spatial, so slow potentials from sustained gaze offsets are removed along with the saccade steps.
- **Do not remove the uniform "common-mode" component**, the one with the same sign on every channel (IC0 here). In the recorded reference it carries broad brain activity, including the P3.
- **Checks** (see `Cleaned.ica_corr` and `Cleaned.unclean`):
  - The blink-locked average at AF3/AF4 drops from ~180 µV to the level of other channels.
  - Single-trial F7−F8 epochs show no condition-locked slow wave.

## 4. ERP filtering

**0.1–20 Hz**, zero-phase FIR (MNE defaults), applied to the continuous data before epoching.

- The 0.1 Hz high-pass preserves the P3 and slow waves. Do not raise it without checking for distortion.
- The 20 Hz low-pass matters at a 128 Hz sampling rate: 20–30 Hz content on the EPOC X is mostly EMG and sensor noise, which made the baselines "wiggly" with ~70 trials. Removing it does not affect N1 latency.

## 5. Residual artifact rejection

On the ICA-cleaned 0.1–20 Hz data: 1-s windows with a 0.5-s step whose peak-to-peak exceeds **100 µV** on any good channel become `BAD_residual`, padded 0.25 s before and 0.5 s after to cover pop recovery.

- The spans are on the returned raw as `BAD_residual` annotations, and the masks are on `Cleaned`.
- Epochs overlapping a BAD span are dropped (`reject_by_annotation=True`).
- In `gng.xdf`, with F4 dropped, 4.8% of task time was excluded. 70/75 Go hits and 302/320 correct NoGo epochs were kept.
- The 100 µV threshold sits above the 99th percentile of clean-channel 1-s peak-to-peak after ICA (75–95 µV). Recheck that percentile (`emotiv.preprocess.sliding_p2p`) for a new headset or session.

## 6. Epochs, baseline, noise

- Epochs run from −300 to 1000 ms around the shifted onset, with baseline −200 to 0 ms. The minimum SOA in `gng.xdf` is 1.26 s, so the next tone never falls inside the epoch.
- **Is a non-flat baseline a problem?** Compare the baseline RMS with the RMS of the **± average** (half the trials sign-flipped, `emotiv.erp.plus_minus_rms`), which is pure noise at that trial count. If they match, the wiggle is noise and only more trials or less noise will help. If the baseline is clearly larger, look for overlap, anticipation or artifacts.
  - `gng.xdf`, frontal Go (n = 70): 0.36 µV vs 0.65 µV noise.
  - NoGo baselines run ~1.4× the ± average, but below 0.5 µV.
- Show 95% trial-bootstrap bands (2000 resamples) for every condition waveform and difference wave.

## 7. Reference: recorded (CMS) only

- The EPOC X records against its CMS/DRL sensors behind the ears (mastoid area). **Analyse in that reference and do not re-reference to the average.**
  - On this ring-shaped 14-channel montage, a broad component reaches every sensor.
  - At the Go − NoGo peak (422 ms after the sound) in `gng.xdf`, every channel was positive and the channel mean was +4.2 µV. An average reference would subtract that mean and flip P7/P8/O1/O2 negative (`results.json → topography_at_frontal_peak`).
- The most superior sensors, **F3/F4/FC5/FC6**, are the best available proxy for Cz/Pz; there are no midline sites. P7/P8/O1/O2 sit low and near the reference, so broad components are small there.
- For the same reason a topography on this montage cannot distinguish P3a from P3b. "Frontal-maximal" means "maximal at the most superior sensors".

## 8. Statistics

- **Cluster permutation test** (`emotiv.erp.cluster_test`): spatio-temporal, Welch t, cluster-forming threshold p < .01, 2000 permutations, over 0–1000 ms. In `gng.xdf`: a positive cluster at 352–445 ms, p = .003, over 10 channels.
- **Peak latencies** with trial-bootstrap 95% CIs.
- **Per-block replication** of the key effect (`by_block` in the analysis script).
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
| `GROSS_UV` | 300 µV | 1-s p2p, excluded from ICA and final epoching |
| `RESIDUAL_UV` | 100 µV | 1-s p2p after ICA, BAD_residual |
| `WIN_S`, `STEP_S` | 1.0 s, 0.5 s | artifact windows |
| `PAD_S` | 0.25 s before, 0.5 s after | around flagged windows |
| `EOG_R` | 0.5 | IC vs VEOG/HEOG correlation |
| `CHANNEL_BAD_SHARE` | 0.10 | residual share that makes a channel bad |
| `POP_UV` | 60 µV | pop diagnostic |
| `ERP_WINDOW`, `BASELINE` | −0.3–1.0 s, −0.2–0 s | epochs (`emotiv/erp.py`) |
| bootstrap draws | 2000 | trial bootstrap |
| `SEED` | 7 | ICA and every bootstrap (`emotiv/devices.py`) |
