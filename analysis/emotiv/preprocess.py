"""From a ``Run`` to clean, ICA-corrected continuous data, ready to epoch.

**The EPOC X and the Flex 1.0 are different hardware and are not processed the
same way.**  This module supplies the mechanisms only; every band, threshold and
policy comes from ``run.dev.processing`` (see ``devices.py``), because the two
headsets differ in noise floor, dynamic range, decoder and montage geometry:

* **Reference.**  The EPOC X is analysed in its recorded (CMS) reference: its 14
  sensors form a ring with no midline site, so an average reference subtracts the
  broad component of interest.  The Flex is analysed in an average reference,
  which measurably outperformed its recorded reference on this cap.  Getting this
  backwards is the single most damaging mistake available here.
* **Bad channels** are dropped on the EPOC X (spline interpolation from 13 sparse
  sensors blurs the ROIs) and interpolated on the denser Flex cap.
* **Passbands and artifact thresholds** are per-device and are not transferable;
  the thresholds are percentiles of that headset's own clean-channel amplitude.

What *is* shared: ICA removes blinks **and**, where the montage provides a
left/right frontal pair, saccades, found by correlating component sources with
EOG proxies.  Leaving the saccade component in produced condition-dependent slow
waves in the first Go/NoGo analysis.  The uniform common-mode component is never
removed; in a recorded reference it carries broad brain activity including the P3.

Artifacts are handled in two passes: gross windows are excluded from the ICA
*fit* only, and residual windows after ICA become ``BAD_residual`` annotations so
epochs overlapping them drop out.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, medfilt, sosfiltfilt, welch

from . import devices
from .devices import FS, SEED
from .loading import Run

mne.set_log_level("ERROR")

__all__ = ["clean", "Cleaned", "contiguous", "make_raw", "good_mask", "channel_noise",
           "bad_channels", "clean_psd", "line_check", "blink_times", "sliding_p2p",
           "flag_windows", "eog_proxies", "electrode_pops", "flex_counts", "flex_deltas",
           "rail_fraction"]

# Device-independent mechanics.  Everything that depends on the headset -- bands,
# thresholds, reference, bad-channel policy, fill padding -- is in
# ``devices.Processing`` and reaches this module through ``run.dev.processing``.
WIN_S, STEP_S = 1.0, 0.5  # artifact window and step
# Pad around a flagged artifact window, covering electrode-pop recovery.  This is
# about the shape of an artifact, not the headset, so it is not per-device; the
# padding around a *filled* stretch is (``Processing.fill_pad``).
PAD_S = (0.25, 0.5)
EOG_R = 0.5  # |r| between an IC source and the 1-10 Hz VEOG/HEOG proxy
CHANNEL_BAD_SHARE = 0.10  # a channel over the residual threshold in >10% of task windows is bad


def contiguous(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and stop indices of each run of True."""

    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def make_raw(run: Run, montage: str = "standard_1020") -> mne.io.RawArray:
    """Continuous data in the recorded reference, filled stretches annotated BAD_fill.

    The padding is ``Processing.fill_pad``, per-device so it can track how each
    reader fills.  Both are currently (0.25, 0.5) s: neither headset leaves a
    decaying recovery behind a gap now that the Flex decodes without the DC
    restore, and what a hold-filled Flex block does leave is a step, which the
    high pass and the epoch baseline remove without a long tail.
    """

    pad = run.dev.processing.fill_pad
    info = mne.create_info(list(run.labels), FS, "eeg")
    raw = mne.io.RawArray(run.x_uv.T * 1e-6, info)
    raw.set_montage(montage, on_missing="warn")
    starts, stops = contiguous(run.filled)
    onset = np.maximum(0.0, starts / FS - pad[0])
    duration = (stops - starts) / FS + sum(pad)
    raw.set_annotations(mne.Annotations(onset, duration, ["BAD_fill"] * len(onset)))
    return raw


# ---------------------------------------------------------------------------
# Flex delta recovery.  The Flex transmits seven-bit deltas, never a level, so
# the published signal is their running sum and the deltas come back exactly by
# differencing it.  Both the rail test below and ``qc.integrity`` work from these.


def flex_counts(run: Run) -> np.ndarray:
    """Published Flex signal back in ADC counts, centred on the 14-bit midpoint."""

    counts = run.x_uv / run.dev.lsb_uv
    # Recorded with remove_dc=False (the default), so the midpoint is still in it.
    # A pre-a3f1084 file recorded with remove_dc=True has already had it taken out.
    if np.median(counts) > devices.FLEX_ADC_MIDPOINT / 2:
        counts = counts - devices.FLEX_ADC_MIDPOINT
    return counts


def flex_deltas(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """The transmitted deltas, and a mask of the samples that came off the wire.

    Undoing any DC-restore leak the file was recorded with (``run.leak``) must
    return whole counts; ``qc.integrity`` checks that, and a non-integer result
    means the file was decoded with different settings than its metadata claims.
    """

    counts = flex_counts(run)
    return counts[1:] - run.leak * counts[:-1], ~run.filled[1:]


def rail_fraction(run: Run) -> pd.Series:
    """Percent of received samples whose delta hit the seven-bit endpoint, per channel.

    The encoder saturates at -63/+64 counts, i.e. 32.1 uV per sample or 4.1 mV/s
    at 128 Hz.  A rail is therefore either a real transient faster than that --
    blinks cost the frontal channels 1-2% of their samples -- or an electrode that
    has stopped tracking its site, which rails almost continuously.  That makes
    this the Flex's own dead-electrode test, and it flags sites the 20-40 Hz noise
    test passes: a badly positioned Oz railed 82% of samples at rest while its
    noise looked ordinary.  Empty on a headset that transmits absolute counts.
    """

    if run.device != "flex":
        return pd.Series(dtype=float)
    deltas, real = flex_deltas(run)
    whole = np.round(deltas).astype(int)
    limit = devices.FLEX_DELTA_LIMIT
    railed = (whole <= -run.delta_zero) | (whole >= 2 * limit + 1 - run.delta_zero)
    return pd.Series(100 * railed[real].mean(axis=0), index=list(run.labels))


def good_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """Samples outside BAD annotations."""

    mask = np.ones(raw.n_times, bool)
    for ann in raw.annotations:
        if ann["description"].startswith("BAD"):
            a = int(ann["onset"] * FS)
            mask[a:a + int(math.ceil(ann["duration"] * FS))] = False
    return mask


def good_channels(inst) -> list[str]:
    return [c for c in inst.ch_names if c not in inst.info["bads"]]


def channel_noise(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Per-channel noise on clean spans (median reference), with a bad-channel flag.

    A channel is flagged when its 20-40 Hz power (above the EEG, where the
    headset's own noise dominates) is more than 4x the median channel, or when it
    is flat.  Channel noise spans a ~20x range on these caps, so a z-score over
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


def bad_channels(raw: mne.io.BaseRaw) -> list[str]:
    return channel_noise(raw).query("bad")["channel"].tolist()


def sliding_p2p(x: np.ndarray, win_s: float = WIN_S,
                step_s: float = STEP_S) -> tuple[np.ndarray, np.ndarray]:
    """Peak-to-peak (channels x windows) in sliding windows, with the window starts."""

    n, step = int(round(win_s * FS)), int(round(step_s * FS))
    view = np.lib.stride_tricks.sliding_window_view(x, n, axis=1)[:, ::step]
    return np.ptp(view, axis=2), np.arange(view.shape[1]) * step


def flag_windows(p2p: np.ndarray, starts: np.ndarray, threshold: float, n_times: int,
                 pad_s: tuple[float, float] = PAD_S,
                 channels: list[int] | None = None) -> np.ndarray:
    """Sample mask of windows whose p2p exceeds ``threshold`` in any selected channel."""

    sel = p2p if channels is None else p2p[channels]
    n = int(round(WIN_S * FS))
    bad = np.zeros(n_times, bool)
    for s in starts[(sel > threshold).any(axis=0)]:
        bad[max(0, s - int(pad_s[0] * FS)):min(n_times, s + n + int(pad_s[1] * FS))] = True
    return bad


def mask_to_annotations(mask: np.ndarray, label: str) -> mne.Annotations:
    starts, stops = contiguous(mask)
    return mne.Annotations(starts / FS, (stops - starts) / FS, [label] * len(starts))


def eog_proxies(x_uv: np.ndarray, labels: list[str], run: Run,
                veog: tuple[str, ...] | None = None,
                heog: tuple[str, str] | None = None) -> dict[str, np.ndarray]:
    """VEOG (mean of the frontal pair) and HEOG (left minus right), in microvolts.

    A missing proxy is warned about rather than skipped quietly: without HEOG the
    saccade component stays in the data, which is the defect this pipeline was
    built to fix.  On a re-montaged cap, pass the session's own pair.
    """

    profile = run.dev.processing
    veog = veog if veog is not None else profile.veog
    heog = heog if heog is not None else profile.heog
    idx = {c: j for j, c in enumerate(labels)}
    out = {}

    present = [idx[c] for c in (veog or ()) if c in idx]
    if present:
        out["VEOG"] = x_uv[present].mean(axis=0)
    else:
        warnings.warn(f"no VEOG proxy for {run.device}: none of {veog} are in the montage; "
                      "blinks will not be removed")

    if heog is None:
        warnings.warn(f"no HEOG pair defined for {run.device}: saccades will NOT be removed. "
                      "Pass heog=(left, right) for this session's montage.")
    elif heog[0] in idx and heog[1] in idx:
        out["HEOG"] = x_uv[idx[heog[0]]] - x_uv[idx[heog[1]]]
    else:
        warnings.warn(f"HEOG pair {heog} is not in the montage for {run.device}: "
                      "saccades will NOT be removed")
    return out


def blink_times(raw: mne.io.BaseRaw, channels) -> np.ndarray:
    """Blink peak times (s), from the mean of the frontal channels given."""

    picks = [c for c in channels if c in raw.ch_names]
    if not picks:
        return np.array([])
    x = raw.get_data(picks=picks) * 1e6
    fp = sosfiltfilt(butter(4, [0.5, 15], btype="band", fs=FS, output="sos"), x.mean(axis=0))
    peaks, _ = find_peaks(fp, height=100, prominence=100, distance=int(0.3 * FS))
    return peaks / FS


def electrode_pops(x_uv: np.ndarray, labels: list[str], usable: np.ndarray,
                   blinks_s: np.ndarray, pop_uv: float = 60.0) -> dict:
    """Channel-specific transients: peaks of |channel - median(others)| > pop_uv at 1-20 Hz.

    Run on ICA-cleaned data, so blinks and saccades are already gone from the good
    channels.  Bad channels are not cleaned by ICA, so +/-0.5 s around each blink
    is masked for every channel.  This is the diagnostic for a saline pad drying
    out: anything above ~0.5 pops/min should be re-wetted before the next session.
    """

    hp = sosfiltfilt(butter(4, [1.0, 20.0], btype="band", fs=FS, output="sos"), x_uv, axis=1)
    usable = usable.copy()
    for b in blinks_s:
        usable[max(0, int((b - 0.5) * FS)):int((b + 0.5) * FS)] = False
    minutes = usable.sum() / FS / 60.0
    out = {"threshold_uv": pop_uv, "usable_minutes": float(minutes), "times_s": {}, "per_min": {}}
    for j, c in enumerate(labels):
        d = hp[j] - np.median(np.delete(hp, j, axis=0), axis=0)
        d[~usable] = 0.0
        pk, _ = find_peaks(np.abs(d), height=pop_uv, distance=int(FS))
        out["times_s"][c] = (pk / FS).tolist()
        out["per_min"][c] = float(len(pk) / minutes) if minutes else float("nan")
    return out


# ---------------------------------------------------------------------------
# The pipeline


@dataclass
class Cleaned:
    """Result of ``clean``: the data to epoch, plus everything needed to QC it."""

    raw: mne.io.RawArray  # ICA-cleaned, ERP-band, recorded reference, BADs annotated
    unclean: mne.io.RawArray  # same filtering and BADs, no ICA; for before/after checks
    bads: list[str]
    ica: mne.preprocessing.ICA
    ica_exclude: list[int]
    ica_corr: dict[str, np.ndarray]
    noise: pd.DataFrame
    rails: pd.Series  # percent of samples at the seven-bit delta endpoint (Flex only)
    pops: dict
    gross_mask: np.ndarray
    residual_mask: np.ndarray
    residual_share: pd.Series
    task_mask: np.ndarray
    parameters: dict = field(default_factory=dict)

    @property
    def good(self) -> list[str]:
        return good_channels(self.raw)

    def summary(self) -> dict:
        """The JSON-safe part, for a results file."""

        return {
            "bad_channels": self.bads,
            "good_channels": self.good,
            "ica_excluded": self.ica_exclude,
            "rail_pct_by_channel": {k: round(float(v), 2) for k, v in self.rails.items()},
            "ica_corr_max": {k: float(np.abs(v).max()) for k, v in self.ica_corr.items()},
            "ica_corr_excluded": {k: [float(v[i]) for i in self.ica_exclude]
                                  for k, v in self.ica_corr.items()},
            "pops_per_min": self.pops["per_min"],
            # Over task time, which is what the per-session QC numbers compare against.
            "pct_task_gross": float(100 * self.gross_mask[self.task_mask].mean()),
            "pct_task_residual": float(100 * self.residual_mask[self.task_mask].mean()),
            "task_minutes": float(self.task_mask.sum() / FS / 60),
            "residual_share_by_channel": self.residual_share.to_dict(),
            "parameters": self.parameters,
        }


def clean(run: Run, *, task_mask: np.ndarray | None = None, manual_bads: dict[str, str] | None = None,
          heog: tuple[str, str] | None = None,
          ica_band: tuple[float, float] | None = None, erp_band: tuple[float, float] | None = None,
          gross_uv: float | None = None, residual_uv: float | None = None,
          eog_r: float = EOG_R, channel_bad_share: float = CHANNEL_BAD_SHARE,
          seed: int = SEED) -> Cleaned:
    """Bad channels, ICA for blinks and saccades, residual annotation.

    Bands, thresholds and the reference come from ``run.dev.processing`` unless
    overridden here; they differ between the two headsets and are not
    interchangeable.

    ``task_mask`` marks the samples that belong to the task, so idle time between
    blocks does not count towards the residual-share and pop rates; it defaults to
    the whole recording.  ``manual_bads`` maps a channel to the reason it is known
    bad.  ``heog`` supplies the left/right frontal pair for a re-montaged cap that
    has no fixed one.
    """

    profile = run.dev.processing
    if not profile.validated:
        warnings.warn(
            f"{run.device} processing profile is unvalidated ({profile.source}). "
            "Its thresholds were not derived on this headset; check them against "
            "preprocess.sliding_p2p percentiles before trusting the output.")
    ica_band = ica_band or profile.ica_band
    erp_band = erp_band or profile.erp_band
    gross_uv = profile.gross_uv if gross_uv is None else gross_uv
    residual_uv = profile.residual_uv if residual_uv is None else residual_uv

    manual_bads = manual_bads or {}
    raw = make_raw(run)
    labels = raw.ch_names
    if task_mask is None:
        task_mask = np.ones(raw.n_times, bool)

    noise = channel_noise(raw)
    bads = [c for c in labels if c in manual_bads]
    bads += [c for c in noise.loc[noise["bad"], "channel"] if c not in bads]
    # The Flex's own dead-electrode test.  It runs before the ICA fit, so a
    # railing channel cannot pull components, and it is independent of the noise
    # test above: a saturated electrode can look quiet at 20-40 Hz.
    rails = rail_fraction(run)
    if profile.rail_pct is not None and len(rails):
        bads += [c for c in labels if c not in bads and rails[c] > profile.rail_pct]

    # 1. Gross windows, excluded from the ICA fit only: blinks must stay in so ICA can learn them.
    fit = raw.copy().filter(*ica_band)
    x_fit = fit.get_data() * 1e6
    p2p_fit, starts = sliding_p2p(x_fit)
    good_idx = [j for j, c in enumerate(labels) if c not in bads]
    gross = flag_windows(p2p_fit, starts, gross_uv, raw.n_times, channels=good_idx)
    fit.set_annotations(fit.annotations + mask_to_annotations(gross, "BAD_gross"))
    fit.info["bads"] = list(bads)

    # 2. ICA in the recorded reference, before any re-referencing: exclude
    #    components tracking VEOG or HEOG.
    n_comp = len(good_idx) - 1
    ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica",
                                random_state=seed, max_iter=3000)
    ica.fit(fit, reject_by_annotation=True)
    sources = ica.get_sources(fit).get_data()
    sos = butter(4, [1.0, 10.0], btype="band", fs=FS, output="sos")
    ok = task_mask & ~gross
    corr = {}
    for name, sig in eog_proxies(x_fit, labels, run, heog=heog).items():
        sig = sosfiltfilt(sos, sig)
        corr[name] = np.array([np.corrcoef(sources[k, ok], sig[ok])[0, 1] for k in range(n_comp)])
    exclude = sorted({k for v in corr.values() for k in np.flatnonzero(np.abs(v) >= eog_r)})

    # 3. Apply to the ERP band, then set this device's analysis reference.
    unclean = raw.copy().filter(*erp_band)
    cleaned = unclean.copy()
    cleaned.info["bads"] = list(bads)
    ica.apply(cleaned, exclude=exclude)
    if profile.reference == "average":
        for inst in (cleaned, unclean):
            inst.info["bads"] = list(bads)
            inst.set_eeg_reference("average")
    elif profile.reference != "recorded":
        raise ValueError(f"unknown reference policy {profile.reference!r}")

    # 4. Residual windows, and channels that trip the threshold too often.
    x = cleaned.get_data() * 1e6
    p2p, starts = sliding_p2p(x)
    end = np.minimum(starts + int(WIN_S * FS) - 1, len(task_mask) - 1)
    task_win = task_mask[starts] & task_mask[end]
    share = pd.Series((p2p[:, task_win] > residual_uv).mean(axis=1), index=labels)
    bads += [c for c in labels if c not in bads and share[c] > channel_bad_share]
    cleaned.info["bads"] = list(bads)
    good_idx = [j for j, c in enumerate(labels) if c not in bads]
    residual = flag_windows(p2p, starts, residual_uv, raw.n_times, channels=good_idx)
    cleaned.set_annotations(cleaned.annotations + mask_to_annotations(residual, "BAD_residual"))
    unclean.info["bads"] = list(bads)
    unclean.set_annotations(cleaned.annotations)

    pops = electrode_pops(x, labels, task_mask, blink_times(raw, profile.veog),
                          pop_uv=profile.pop_uv)
    return Cleaned(
        raw=cleaned, unclean=unclean, bads=bads, ica=ica, ica_exclude=[int(k) for k in exclude],
        ica_corr=corr, noise=noise, rails=rails, pops=pops, gross_mask=gross, residual_mask=residual,
        residual_share=share, task_mask=task_mask,
        parameters={"device": run.device, "profile_source": profile.source,
                    "profile_validated": profile.validated,
                    "ica_band": list(ica_band), "erp_band": list(erp_band),
                    "gross_uv": gross_uv, "residual_uv": residual_uv,
                    "win_s": WIN_S, "step_s": STEP_S, "fill_pad": list(profile.fill_pad),
                    "eog_r": eog_r, "channel_bad_share": channel_bad_share,
                    "pop_uv": profile.pop_uv, "rail_pct": profile.rail_pct,
                    "manual_bads": manual_bads,
                    "reference": profile.reference, "reference_why": profile.reference_why,
                    "bad_channel_handling": profile.bad_channels,
                    "eog_proxies_used": sorted(corr), "seed": seed},
    )


# ---------------------------------------------------------------------------
# Spectra


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
    if total is None:
        raise ValueError(f"no clean stretch at least {nperseg} samples long")
    return freqs, total / weight


def line_check(freqs: np.ndarray, psd_median: np.ndarray) -> dict:
    """Narrowband peaks over a smooth baseline: mains and where its harmonics alias."""

    baseline = np.exp(medfilt(np.log(psd_median), 41))
    ratio = psd_median / baseline
    out = {}
    for f0 in (50, 22, 28, 6):
        near = np.abs(freqs - f0) <= 0.3
        out[f"{f0}Hz_peak_ratio"] = float(ratio[near].max())
    out["largest_narrowband_ratio_2_60Hz"] = float(ratio[(freqs > 2) & (freqs < 60)].max())
    return out
