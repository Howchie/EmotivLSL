"""Epoching and ERP measurement, with the headset's chain latency already removed.

The epoch window, baseline, reference and bad-channel policy come from
``run.dev.processing``: the two headsets are different hardware and do not share
them.  This module only applies whatever that profile specifies.

``erp_epochs`` is the only place timing correction happens.  Markers are stamped
when the stimulus was *sent*; the sample carrying the response to it arrives one
chain latency later, so the event samples are shifted forward by that much before
epoching.  After that, t=0 on every epoch is the physical stimulus and latencies
read off the waveform are real latencies.

The correction uses ``run.dev.timing``.  If that device's latency has not been
measured yet the call raises, because silently epoching on uncorrected markers is
the failure this whole layer exists to prevent; pass ``chain_latency_s`` to
override, or ``correct_timing=False`` to deliberately work in marker time.
"""

from __future__ import annotations

import mne
import numpy as np
from scipy import stats

from .devices import FS, SEED
from .loading import Run

mne.set_log_level("ERROR")

__all__ = ["erp_epochs", "roi_trials", "peak_latency", "bootstrap_latency",
           "cluster_test", "n1_p2", "plus_minus_rms"]

# The epoch runs to 1 s because the minimum SOA used so far is 1.26 s, so the
# next stimulus never falls inside it.
ERP_WINDOW = (-0.3, 1.0)
BASELINE = (-0.2, 0.0)


def _chain_latency(run: Run, correct: bool, override: float | None) -> float:
    if not correct:
        return 0.0
    if override is not None:
        return float(override)
    timing = run.dev.timing
    if not timing.measured:
        raise ValueError(
            f"{run.device} chain latency is provisional ({timing.describe()}). "
            "Measure it, or pass chain_latency_s=... / correct_timing=False explicitly."
        )
    return timing.latency_s


def erp_epochs(run: Run, cleaned, events, *, codes: dict[str, int] | None = None,
               correct_timing: bool = True, chain_latency_s: float | None = None,
               window: tuple[float, float] | None = None,
               baseline: tuple[float, float] | None = None) -> tuple[mne.Epochs, dict]:
    """Epoch cleaned data on stimulus-corrected events, dropping bad channels.

    ``cleaned`` is the ``Cleaned`` from ``preprocess.clean`` (or a raw that has
    already been filtered, ICA-corrected and annotated).  Filtering, referencing
    and artifact handling all happened there; this function only shifts the
    events and cuts epochs, so the two stages cannot disagree about the
    reference.

    ``events`` needs ``sample`` and ``condition`` columns (see ``events.py``).
    ``codes`` maps condition to event id; fix it explicitly when recordings are
    to be compared or concatenated, otherwise ids are assigned alphabetically.
    """

    profile = run.dev.processing
    erp_raw = cleaned.raw if hasattr(cleaned, "raw") else cleaned
    window = window or profile.epoch
    baseline = baseline or profile.baseline
    latency = _chain_latency(run, correct_timing, chain_latency_s)
    shift = int(round(latency * FS))

    info: dict = {"chain_latency_s": latency, "shift_samples": shift,
                  "timing_source": run.dev.timing.source if chain_latency_s is None else "caller",
                  "reference": profile.reference,
                  "bad_channel_handling": profile.bad_channels,
                  "bad_channels": list(erp_raw.info["bads"])}

    # Bad channels are handled the way this headset needs: dropped on the sparse
    # EPOC X ring, interpolated on the denser Flex cap.
    if profile.bad_channels == "interpolate" and erp_raw.info["bads"]:
        erp_raw = erp_raw.copy().interpolate_bads(reset_bads=True)
    elif profile.bad_channels not in ("drop", "interpolate"):
        raise ValueError(f"unknown bad-channel policy {profile.bad_channels!r}")

    conditions = sorted(events["condition"].unique())
    codes = codes or {name: i + 1 for i, name in enumerate(conditions)}
    unknown = set(conditions) - set(codes)
    if unknown:
        raise KeyError(f"no event code for {sorted(unknown)}")

    samples = events["sample"].to_numpy() + shift
    keep = (samples >= 0) & (samples < erp_raw.n_times)
    if not keep.all():
        info["events_outside_recording"] = int((~keep).sum())
    events = events[keep]
    array = np.column_stack([samples[keep], np.zeros(keep.sum(), int),
                             [codes[c] for c in events["condition"]]])
    present = {k: v for k, v in codes.items() if v in array[:, 2]}

    # Epochs overlapping a BAD span are dropped by annotation, so no amplitude
    # threshold is applied here.
    epochs = mne.Epochs(erp_raw, array, present, tmin=window[0], tmax=window[1],
                        baseline=baseline, picks="data", reject=None,
                        reject_by_annotation=True, preload=True,
                        metadata=events.assign(run=run.key))
    if profile.bad_channels == "drop":
        epochs.drop_channels([c for c in erp_raw.info["bads"] if c in epochs.ch_names])

    drops: dict[str, int] = {}
    for log in epochs.drop_log:
        for reason in log:
            drops[reason] = drops.get(reason, 0) + 1
    info["drops"] = drops
    info["n_epochs"] = len(epochs)
    info["n_by_condition"] = {k: int(len(epochs[k])) for k in present}
    return epochs, info


# ---------------------------------------------------------------------------
# Measurement


def roi_trials(epochs: mne.Epochs, roi) -> np.ndarray:
    """Single-trial waveforms averaged over an ROI, in microvolts."""

    picks = [c for c in roi if c in epochs.ch_names]
    if not picks:
        raise ValueError(f"none of {list(roi)} are in the epochs")
    return epochs.get_data(picks=picks).mean(axis=1) * 1e6


def peak_latency(times: np.ndarray, wave: np.ndarray, window: tuple[float, float], sign: int) -> float:
    sel = (times >= window[0]) & (times <= window[1])
    return float(times[sel][np.argmax(sign * wave[sel])])


def boot_mean_ci(trials: np.ndarray, rng=None, n_boot: int = 2000):
    """Trial-bootstrap mean and 95% CI of a trials x times array."""

    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, len(trials), (n_boot, len(trials)))
    return trials.mean(0), *np.percentile(trials[idx].mean(axis=1), [2.5, 97.5], axis=0)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng=None, n_boot: int = 2000):
    """Difference wave, its 95% CI, and the bootstrap draws (for peak CIs)."""

    rng = rng or np.random.default_rng(SEED)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    boots = a[ia].mean(1) - b[ib].mean(1)
    return a.mean(0) - b.mean(0), *np.percentile(boots, [2.5, 97.5], axis=0), boots


def plus_minus_rms(trials: np.ndarray, sel: np.ndarray, rng=None, n_perm: int = 200) -> float:
    """RMS of the +/- average (a random half of trials inverted): the noise-only residual.

    Compare this with the baseline RMS.  If they match, a wiggly baseline is just
    noise at that trial count and only more trials or less noise will help; if the
    baseline is clearly larger, look for overlap, anticipation or artifacts.
    """

    rng = rng or np.random.default_rng(SEED)
    vals = []
    for _ in range(n_perm):
        sign = np.ones(len(trials))
        sign[rng.permutation(len(trials))[: len(trials) // 2]] = -1
        vals.append(np.sqrt(np.mean((trials * sign[:, None]).mean(0)[sel] ** 2)))
    return float(np.mean(vals))


def bootstrap_latency(times, trials, window, sign, n_boot: int = 2000, rng=None) -> list[float]:
    rng = rng or np.random.default_rng(SEED)
    lat = [peak_latency(times, trials[rng.integers(0, len(trials), len(trials))].mean(axis=0), window, sign)
           for _ in range(n_boot)]
    return [float(v) for v in np.percentile(lat, [2.5, 97.5])]


def n1_p2(epochs: mne.Epochs, frontal, inferior, rng=None) -> dict:
    """Trough/peak latencies of the frontal-minus-inferior waveform (sharpest N1/P2 view).

    With ``erp_epochs``' correction applied these are latencies after the sound,
    directly comparable with the literature.
    """

    rng = rng or np.random.default_rng(SEED)
    wave = roi_trials(epochs, frontal) - roi_trials(epochs, inferior)
    times = epochs.times
    grand = wave.mean(axis=0)
    n1_window = (0.05, 0.40)
    n1 = peak_latency(times, grand, n1_window, -1)
    p2_window = (n1 + 0.03, 0.55)
    p2 = peak_latency(times, grand, p2_window, +1)
    return {
        "n_trials": int(len(wave)),
        "n1_latency_s": n1, "n1_ci95": bootstrap_latency(times, wave, n1_window, -1, rng=rng),
        "p2_latency_s": p2, "p2_ci95": bootstrap_latency(times, wave, p2_window, +1, rng=rng),
        "n1_amp_uv": float(grand[times == n1][0]), "p2_amp_uv": float(grand[times == p2][0]),
        "_wave": wave,
    }


def cluster_test(a: mne.Epochs, b: mne.Epochs, tmin: float = 0.0, tmax: float = ERP_WINDOW[1],
                 n_perm: int = 2000, seed: int = SEED) -> dict:
    """Spatio-temporal cluster permutation test between two conditions (Welch t)."""

    adjacency, _ = mne.channels.find_ch_adjacency(a.info, "eeg")
    sel = (a.times >= tmin) & (a.times <= tmax)
    xa = a.get_data()[:, :, sel].transpose(0, 2, 1) * 1e6
    xb = b.get_data()[:, :, sel].transpose(0, 2, 1) * 1e6
    df = len(xa) + len(xb) - 2
    threshold = stats.t.ppf(1 - 0.01 / 2, df)
    stat_fun = lambda x, y: mne.stats.ttest_ind_no_p(x, y, equal_var=False)
    t_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_test(
        [xa, xb], adjacency=adjacency, threshold=threshold, n_permutations=n_perm, tail=0,
        stat_fun=stat_fun, seed=seed, out_type="mask", buffer_size=None)
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
    return {"n_a": len(xa), "n_b": len(xb), "t_obs": t_obs, "times": times,
            "clusters": found, "masks": clusters, "p_values": p_values}
