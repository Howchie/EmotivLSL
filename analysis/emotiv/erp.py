"""Epoching and ERP measurement, with hardware latency removed at load time.

The epoch window, baseline, reference and bad-channel policy come from
``run.dev.processing``: the two headsets are different hardware and do not share
them.  This module only applies whatever that profile specifies.

The loader shifts every marker by the fixed device chain latency, so block, rest
and stimulus markers are already in raw EEG time.  ``erp_epochs`` only applies an
optional session-specific residual (currently the one-off audio-output correction
for a known recording).  After that, t=0 on every epoch is the physical stimulus.

The fixed correction is recorded in ``run.marker_shift_s``; pass
``chain_latency_s`` only for an additional session-specific correction, or use
``correct_timing=False`` to suppress even that additional correction.
"""

from __future__ import annotations

import mne
import numpy as np
from scipy import stats

from .devices import SEED
from .loading import Run

mne.set_log_level("ERROR")

__all__ = ["erp_epochs", "roi_picks", "roi_trials", "complete_cases", "peak_latency", "peak_width",
           "bootstrap_latency", "cluster_test", "roi_cluster_test", "n1_p2",
           "plus_minus_rms", "contrast",
           "topography", "by_block"]

# The epoch runs to 1 s because the shortest SOA used so far is 1.20 s
# (epoc_gng_timingfixed.xdf), so the next stimulus never falls inside it.
ERP_WINDOW = (-0.3, 1.0)
BASELINE = (-0.2, 0.0)


def _chain_latency(run: Run, correct: bool, override: float | None) -> float:
    if not correct:
        return 0.0
    # The fixed hardware delay is applied to all markers by the loader.  This
    # argument is deliberately only an additional session-level correction.
    return 0.0 if override is None else float(override)


def erp_epochs(run: Run, cleaned, events, *, codes: dict[str, int] | None = None,
               correct_timing: bool = True, chain_latency_s: float | None = None,
               window: tuple[float, float] | None = None,
               baseline: tuple[float, float] | None = None,
               bad_channels: str | None = None) -> tuple[mne.Epochs, dict]:
    """Epoch cleaned data on stimulus-corrected events, dropping bad channels.

    ``cleaned`` is the ``Cleaned`` from ``preprocess.clean`` (or a raw that has
    already been filtered, ICA-corrected and annotated).  Filtering, referencing
    and artifact handling all happened there; this function only shifts the
    events and cuts epochs, so the two stages cannot disagree about the
    reference.

    ``events`` needs ``sample`` and ``condition`` columns (see ``events.py``).
    ``codes`` maps condition to event id; fix it explicitly when recordings are
    to be compared or concatenated, otherwise ids are assigned alphabetically.

    ``bad_channels`` overrides the device policy ("drop" or "interpolate").  Drop
    a bad channel rather than interpolating it when the analysis depends on the
    shape of a gradient across the cap -- interpolating the inferior-most
    electrodes from superior neighbours pulls the inferior ROI towards the
    superior one, which is exactly the contrast an MMN polarity check is reading.
    """

    profile = run.dev.processing
    erp_raw = cleaned.raw if hasattr(cleaned, "raw") else cleaned
    window = window or profile.epoch
    baseline = baseline or profile.baseline
    handling = bad_channels or profile.bad_channels
    sfreq = float(erp_raw.info["sfreq"])
    latency = _chain_latency(run, correct_timing, chain_latency_s)
    shift = int(round(latency * sfreq))

    info: dict = {"additional_latency_s": latency, "shift_samples": shift,
                  "sfreq_hz": sfreq, "hardware_marker_shift_s": run.marker_shift_s,
                  "timing_source": "loader_marker_shift" if chain_latency_s is None else "caller",
                  "reference": profile.reference,
                  "bad_channel_handling": handling,
                  "bad_channels": list(erp_raw.info["bads"])}

    if handling not in ("drop", "interpolate"):
        raise ValueError(f"unknown bad-channel policy {handling!r}")

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

    def build(bads: list[str]) -> tuple[mne.io.BaseRaw, mne.Epochs]:
        """Epoch with this bad-channel list, handled the way the headset needs:
        dropped on the sparse EPOC X ring, interpolated on the denser Flex cap."""

        raw = erp_raw
        if handling == "interpolate" and bads:
            raw = erp_raw.copy()
            raw.info["bads"] = list(bads)
            raw.interpolate_bads(reset_bads=True)
        # Epochs overlapping a BAD span are dropped by annotation, so no amplitude
        # threshold is applied here.
        ep = mne.Epochs(raw, array, present, tmin=window[0], tmax=window[1],
                        baseline=baseline, picks="data", reject=None,
                        reject_by_annotation=True, preload=True,
                        metadata=events.assign(run=run.key))
        if handling == "drop":
            ep.drop_channels([c for c in bads if c in ep.ch_names])
        return raw, ep

    # An electrode that is unusable in most epochs is not a per-epoch problem, it
    # is a bad electrode, and keeping it costs every other channel nothing but
    # costs the analysis a column of holes.  Re-epoch with it out.  Dropping one
    # channel does not change another's spans, so this settles in one extra pass;
    # the loop is bounded anyway.
    bads = list(erp_raw.info["bads"])
    found: dict[str, float] = {}
    for _ in range(3):
        raw_used, epochs = build(bads)
        cell_bad = _cell_flags(epochs, raw_used, cleaned, window, handling, bads)
        share = cell_bad.mean(axis=0)
        over = {c: float(share[j]) for j, c in enumerate(epochs.ch_names)
                if share[j] > profile.channel_max_bad_epoch_share and c not in bads}
        if not over:
            break
        found.update(over)
        bads = sorted(set(bads) | set(over))
    info["bad_channels"] = list(bads)
    info["bad_channels_from_epoch_share"] = {c: round(v, 4) for c, v in found.items()}
    info["channel_max_bad_epoch_share"] = profile.channel_max_bad_epoch_share

    # An epoch that has lost too much of the cap is not salvageable by averaging
    # the rest of it: the channels still standing were in the same head at the
    # same moment.  Drop it outright.
    n_ch = len(epochs.ch_names)
    cut = profile.epoch_max_bad_share
    too_many = np.flatnonzero(cell_bad.sum(axis=1) > cut * n_ch)
    info["epoch_max_bad_share"] = cut
    info["epochs_dropped_bad_channel_share"] = int(len(too_many))
    if len(too_many):
        epochs.drop(too_many, reason="BAD_channel_share")
        cell_bad = np.delete(cell_bad, too_many, axis=0)

    md = epochs.metadata.copy()
    for j, c in enumerate(epochs.ch_names):
        md[f"bad_{c}"] = cell_bad[:, j]
    epochs.metadata = md
    # Carried on the epochs so roi_trials applies the same share after selection,
    # and knows which channels are splines rather than electrodes.
    epochs.info["temp"] = {"epoch_max_bad_share": float(cut),
                           "interpolated": [c for c in bads if c in epochs.ch_names]}

    drops: dict[str, int] = {}
    for log in epochs.drop_log:
        for reason in log:
            drops[reason] = drops.get(reason, 0) + 1
    info["drops"] = drops
    info["n_epochs"] = len(epochs)
    info["n_by_condition"] = {k: int(len(epochs[k])) for k in present}
    n_per = {c: int(len(epochs) - md[f"bad_{c}"].sum()) for c in epochs.ch_names}
    info["n_per_channel"] = n_per
    info["n_per_channel_min"] = int(min(n_per.values()))
    info["n_per_channel_max"] = int(max(n_per.values()))
    info["channel_epochs_excluded"] = {c: int(len(epochs) - n) for c, n in n_per.items()
                                       if n < len(epochs)}
    info["n_epochs_all_channels_usable"] = int((~cell_bad).all(axis=1).sum())
    info["bad_channels_per_epoch_max"] = int(cell_bad.sum(axis=1).max()) if len(epochs) else 0
    return epochs, info


def _cell_flags(epochs: mne.Epochs, erp_raw, cleaned, window, handling,
                bads: list[str]) -> np.ndarray:
    """Epochs x channels: was that electrode usable in that epoch?

    ``preprocess.clean`` records single-electrode excursions separately from
    subject-level ones (``Cleaned.channel_bad``) precisely so they do not cost
    every channel the epoch.  Here they become a per-epoch, per-channel flag,
    which ``erp_epochs`` writes into the metadata as one ``bad_<channel>`` column
    each.  Metadata is carried through selection, so ``epochs["go"]`` keeps them,
    and ``roi_trials`` averages each trial over the channels usable in it.  The
    consequence is deliberate and has to be reported: **the trial count is per
    channel, not per analysis.**

    An interpolated channel is flagged usable throughout, because its samples no
    longer come from that electrode; its neighbours carry their own flags.
    """

    n_ch = len(epochs.ch_names)
    if getattr(cleaned, "channel_bad", None) is None or not len(epochs):
        return np.zeros((len(epochs), n_ch), bool)
    span, names = cleaned.channel_bad, erp_raw.ch_names
    sfreq = float(epochs.info["sfreq"])
    start = epochs.events[:, 0] + int(round(window[0] * sfreq))
    idx = np.clip(start[:, None] + np.arange(len(epochs.times))[None, :], 0, span.shape[1] - 1)
    cell = np.stack([span[names.index(c)][idx].any(axis=1) for c in epochs.ch_names], axis=1)
    if handling == "interpolate":
        # ``interpolate_bads(reset_bads=True)`` has already emptied
        # ``info["bads"]``, hence the list passed in.
        for c in bads:
            if c in epochs.ch_names:
                cell[:, epochs.ch_names.index(c)] = False
    return cell


def _usable(epochs: mne.Epochs, picks) -> np.ndarray:
    """Trials x picks boolean, True where that channel is usable in that trial."""

    md = epochs.metadata
    cols = [f"bad_{c}" for c in picks]
    if md is None or any(c not in md.columns for c in cols):
        return np.ones((len(epochs), len(picks)), bool)
    return ~md[cols].to_numpy(dtype=bool)


def complete_cases(epochs: mne.Epochs) -> mne.Epochs:
    """The epochs with every analysed channel usable.

    Needed wherever the method cannot tolerate a ragged array -- the cluster test
    works on a trials x times x channels block and has no way to represent a
    missing cell.  It is stricter than the ROI measurement, so report both counts
    rather than quoting one of them as "the" trial count.
    """

    keep = np.flatnonzero(_usable(epochs, epochs.ch_names).all(axis=1))
    return epochs[keep] if len(keep) < len(epochs) else epochs


def _cell_data(epochs: mne.Epochs, policy: str) -> tuple[np.ndarray, int]:
    """Rectangular data for a method that cannot take a missing cell, in microvolts.

    Returns the data and how many epochs were given up to get it.  ``policy`` is
    the device's ``Processing.bad_cells``: "complete" keeps only epochs with every
    channel usable, "interpolate" fills the unusable cells from their neighbours,
    which needs a cap dense enough for the spline to mean anything.
    """

    bad = ~_usable(epochs, epochs.ch_names)
    if policy == "complete" or not bad.any():
        keep = np.flatnonzero(~bad.any(axis=1))
        return epochs.get_data()[keep] * 1e6, len(epochs) - len(keep)
    if policy != "interpolate":
        raise ValueError(f"unknown bad-cell policy {policy!r}")
    # A trial with no usable channel has nothing to interpolate from.
    keep = np.flatnonzero(~bad.all(axis=1))
    data, bad = epochs.get_data()[keep], bad[keep]
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, row in enumerate(bad):
        groups.setdefault(tuple(np.flatnonzero(row)), []).append(i)
    for chans, idx in groups.items():
        if not chans:
            continue
        sub = mne.EpochsArray(data[idx], epochs.info.copy(), tmin=epochs.tmin,
                              baseline=None, verbose="error")
        sub.info["bads"] = [epochs.ch_names[c] for c in chans]
        sub.interpolate_bads(reset_bads=True)
        data[idx] = sub.get_data()
    return data * 1e6, len(epochs) - len(keep)


# ---------------------------------------------------------------------------
# Measurement


def roi_picks(epochs: mne.Epochs, roi) -> list[str]:
    """The ROI channels present in these epochs and carrying their own data.

    A channel the profile interpolated rather than dropped is still in
    ``ch_names``, but its samples are a spline of its neighbours -- on a dense cap
    usually including other members of the same ROI.  Averaging it in gives those
    neighbours a second vote, and because an interpolated channel is flagged
    usable in every epoch it also props up the coverage rule in ``roi_trials``,
    which is meant to count surviving electrodes.  So it is excluded from ROI
    averages.  ``cluster_test`` and ``topography`` keep it: both need a value at
    every montage position to produce a map at all, which is what the
    interpolation is for.

    Measured on gng_flex.xdf, where O1/PO9/PO10 are interpolated: the posterior
    ROI (O1 Oz O2) gives +0.73 uV either way -- O1's spline is largely Oz and O2,
    so it adds weight and no information -- and the inferior ROI (FT9 FT10 PO9
    PO10, half spline) moves +0.35 to +0.79 uV, null both ways.  No primary ROI on
    either headset contains an interpolated channel.
    """

    interpolated = set((epochs.info.get("temp") or {}).get("interpolated", ()))
    return [c for c in roi if c in epochs.ch_names and c not in interpolated]


def roi_trials(epochs: mne.Epochs, roi, *, counts: bool = False,
               max_bad_share: float | None = None):
    """Single-trial waveforms averaged over an ROI, in microvolts.

    Each trial is averaged over the ROI channels that were usable *in that
    trial*, so the number of contributing electrodes varies from trial to trial
    and the number of trials varies from electrode to electrode; ``counts``
    returns the per-trial channel count alongside the waveforms.

    ``max_bad_share`` is the most of the ROI a trial may lose and still be
    measured, defaulting to the run's ``Processing.epoch_max_bad_share`` via
    ``epochs.info["temp"]`` when ``erp_epochs`` recorded it, else 0.25.  Without
    it a four-channel ROI measured from two channels is silently pooled with one
    measured from four, and it is not the same measurement: on
    epoc_gng_timingfixed.xdf one such trial (two of four frontal channels, the
    other two excluded) returned -15.3 uV against a session mean of +2.4 and
    moved the pooled Go-NoGo by a quarter of a microvolt on its own.  The same
    share as the epoch guard is used deliberately -- an epoch that has lost more
    than a quarter of the cap is dropped, and an ROI that has lost more than a
    quarter of itself cannot be read on that trial.

    Such a trial comes back as a row of NaN rather than being dropped here, so
    two ROIs measured on the same epochs stay aligned and can be subtracted
    (``n1_p2`` does exactly that).  Callers that need complete rows select them
    with ``np.isfinite(w).all(axis=1)``.
    """

    picks = roi_picks(epochs, roi)
    if not picks:
        raise ValueError(f"none of {list(roi)} are usable electrodes in these epochs "
                         f"(present: {[c for c in roi if c in epochs.ch_names]})")
    if max_bad_share is None:
        max_bad_share = (epochs.info.get("temp") or {}).get("epoch_max_bad_share", 0.25)
    data = epochs.get_data(picks=picks) * 1e6
    good = _usable(epochs, picks)
    n = good.sum(axis=1)
    enough = n >= max(1, int(np.ceil((1.0 - max_bad_share) * len(picks))))
    wave = np.where(good[:, :, None], data, 0.0).sum(axis=1) / np.maximum(n, 1)[:, None]
    wave[~enough] = np.nan
    return (wave, n) if counts else wave


def peak_latency(times: np.ndarray, wave: np.ndarray, window: tuple[float, float], sign: int) -> float:
    sel = (times >= window[0]) & (times <= window[1])
    return float(times[sel][np.argmax(sign * wave[sel])])


def peak_width(times: np.ndarray, wave: np.ndarray, peak_i: int) -> float:
    """Full width at half maximum of the deflection containing ``peak_i``, in seconds.

    A width test is what separates a component from a peak-picked wiggle.  A P3
    is 150-300 ms wide; a difference wave whose maximum is 40 ms wide is a
    transient or a noise excursion that ``argmax`` has latched onto, however
    large it is.  Always report this next to a peak amplitude, and prefer the
    mean over an a-priori window when the two disagree.
    """

    half = wave[peak_i] / 2.0
    if not np.isfinite(half) or half <= 0:
        return float("nan")
    a = peak_i
    while a > 0 and wave[a] > half:
        a -= 1
    b = peak_i
    while b < len(wave) - 1 and wave[b] > half:
        b += 1
    return float(times[b] - times[a])


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
    wave = wave[np.isfinite(wave).all(axis=1)]  # needs both ROIs, so complete rows only
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
                 n_perm: int = 2000, seed: int = SEED, bad_cells: str = "complete",
                 picks=None) -> dict:
    """Spatio-temporal cluster permutation test between two conditions (Welch t).

    The test needs a full trials x times x channels block and has no way to
    represent a channel that was unusable in one trial, so ``bad_cells`` (the
    device's ``Processing.bad_cells``) decides: "complete" drops those epochs,
    "interpolate" fills the cell from its neighbours on a cap dense enough to
    support it.  Either way ``n_a``/``n_b`` can be below the ROI contrast's
    ``n_test``/``n_ref``, which needs no such block.  Both are reported; neither
    is "the" trial count.

    ``picks`` restricts the test to a subset of electrodes **before** either
    policy is applied, which is usually the better answer than choosing between
    them.  Completeness is then required only of the channels being tested, so an
    epoch is no longer thrown away -- or patched with a spline -- because of an
    electrode that could not have contributed to the cluster anyway.  On
    gng_flex.xdf the epochs that fail the all-channels rule fail it at FT10 (18),
    Oz (16) and FT9 (9), none of which is in the centro-parietal cluster; on
    oddball_flex.xdf it is O2 (35), T7 (20), Oz (17), T8 (16) and O1 (13).
    Dropping that ring from the test recovers most of them.

    ``picks`` is a decision about which electrodes the hypothesis concerns and
    must be made a priori, not after looking at where the cluster landed.  The
    channels used are returned in ``picks`` so a report can state them.
    """

    if picks is not None:
        picks = [c for c in picks if c in a.ch_names and c in b.ch_names]
        if len(picks) < 2:
            raise ValueError(f"cluster_test needs at least two channels, got {picks}")
        a, b = a.copy().pick(picks), b.copy().pick(picks)

    adjacency, _ = mne.channels.find_ch_adjacency(a.info, "eeg")
    sel = (a.times >= tmin) & (a.times <= tmax)
    da, drop_a = _cell_data(a, bad_cells)
    db, drop_b = _cell_data(b, bad_cells)
    xa = da[:, :, sel].transpose(0, 2, 1)
    xb = db[:, :, sel].transpose(0, 2, 1)
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
    return {"n_a": len(xa), "n_b": len(xb), "picks": list(a.ch_names),
            "bad_cells": bad_cells, "n_dropped_incomplete": [drop_a, drop_b],
            "t_obs": t_obs, "times": times,
            "clusters": found, "masks": clusters, "p_values": p_values}


def _time_mask(cluster, n: int) -> np.ndarray:
    """Normalise one MNE cluster to a boolean mask over time.

    ``out_type="mask"`` does not mean the same thing in one dimension as in two.
    For the spatio-temporal test it is a times x channels boolean array, but for
    a 1-D test MNE returns a one-element tuple holding a ``slice``.  Indexing
    ``t_obs`` with that tuple silently yields element zero rather than the
    cluster, which reads as a one-sample cluster at the start of the window --
    the wrong latency and, because it is a single sample, often the wrong sign
    too.  Everything that comes back is funnelled through here instead.
    """

    mask = np.zeros(n, bool)
    parts = cluster if isinstance(cluster, tuple) else (cluster,)
    for part in parts:
        if isinstance(part, slice):
            mask[part] = True
            continue
        arr = np.asarray(part)
        if arr.dtype == bool:
            mask |= arr.reshape(-1)[:n]
        else:
            mask[arr.astype(int)] = True
    return mask


def roi_cluster_test(a: mne.Epochs, b: mne.Epochs, roi, tmin: float = 0.0,
                     tmax: float = ERP_WINDOW[1], n_perm: int = 2000, seed: int = SEED,
                     tail: int = 0) -> dict:
    """Cluster permutation test over *time only*, on one ROI waveform per trial.

    This is the test to use when the hypothesis is about a named site or a small
    ROI -- Fz, Cz, Pz, the mean of F3 and F4 -- rather than about the whole cap.
    It is not a weaker version of ``cluster_test``; it is a different and usually
    more appropriate question, and it avoids that test's central compromise.

    ``cluster_test`` needs a rectangular trials x times x channels block, so a
    single unusable electrode in one epoch forces a choice between discarding
    that epoch and filling it with a spline.  This works from ``roi_trials``,
    which averages each trial over the ROI channels that were usable *in that
    trial*, so no epoch is lost to an electrode outside the ROI and nothing is
    interpolated.  The trials that do drop out are only those that lost more of
    this ROI than ``Processing.epoch_max_bad_share`` allows, and they are
    counted in ``n_dropped_roi``.

    Correcting over time alone also means no correction over 14 or 32 channels,
    so this is a more sensitive test -- which is exactly why the ROI has to be
    fixed in advance.  Choosing it after seeing a topography is circular.
    """

    picks = roi_picks(a, roi)
    if not picks:
        raise ValueError(f"none of {list(roi)} are usable electrodes in these epochs")
    wa, wb = roi_trials(a, picks), roi_trials(b, picks)
    ka, kb = np.isfinite(wa).all(axis=1), np.isfinite(wb).all(axis=1)
    wa, wb = wa[ka], wb[kb]
    sel = (a.times >= tmin) & (a.times <= tmax)
    xa, xb = wa[:, sel], wb[:, sel]
    df = len(xa) + len(xb) - 2
    threshold = stats.t.ppf(1 - 0.01 / 2, df)
    if tail != 0:
        threshold = abs(threshold) * (1 if tail > 0 else -1)
    stat_fun = lambda x, y: mne.stats.ttest_ind_no_p(x, y, equal_var=False)
    t_obs, clusters, p_values, _ = mne.stats.permutation_cluster_test(
        [xa, xb], threshold=threshold, n_permutations=n_perm, tail=tail,
        stat_fun=stat_fun, seed=seed, out_type="mask", buffer_size=None)
    times = a.times[sel]
    found = []
    for cluster, p in zip(clusters, p_values):
        mask = _time_mask(cluster, len(times))
        idx = np.flatnonzero(mask)
        found.append({
            "p": float(p),
            "sign": "positive" if t_obs[mask].sum() > 0 else "negative",
            "t_start": float(times[idx.min()]), "t_end": float(times[idx.max()]),
            "mass": float(np.abs(t_obs[mask]).sum()),
            "mean_diff_uv": float((xa.mean(0) - xb.mean(0))[mask].mean()),
        })
    found.sort(key=lambda c: c["p"])
    return {"roi": picks, "n_a": len(xa), "n_b": len(xb),
            "n_dropped_roi": [int((~ka).sum()), int((~kb).sum())],
            "t_obs": t_obs, "times": times, "clusters": found}


# ---------------------------------------------------------------------------
# Two-condition contrasts.  Generic over the pair being compared, so a Go/NoGo P3
# and an oddball MMN use the same measurement code and the same reporting.


def contrast(epochs: mne.Epochs, roi, rng, window: tuple[float, float],
             conditions: tuple[str, str] = ("go", "nogo"), sign: int = +1) -> dict:
    """Two conditions and their difference over an ROI, with trial-bootstrap CIs.

    ``conditions`` is (test, reference) and the difference is test minus
    reference.  ``sign`` is the polarity the component is expected to have, so a
    negative component such as the MMN is peak-picked as a trough; the window
    mean is signed either way and is the number to quote.
    """

    present = roi_picks(epochs, roi)
    test, ref = conditions
    a, na = roi_trials(epochs[test], present, counts=True)
    b, nb = roi_trials(epochs[ref], present, counts=True)
    # A trial survives if any ROI channel survived; how many did is reported.
    ka, kb = np.isfinite(a).all(axis=1), np.isfinite(b).all(axis=1)
    a, b, na, nb = a[ka], b[kb], na[ka], nb[kb]
    times = epochs.times
    diff, lo, hi, boots = boot_diff_ci(a, b, rng)
    sel = (times >= window[0]) & (times <= window[1])
    peak_i = int(np.flatnonzero(sel)[np.argmax(sign * diff[sel])])
    peak_lat = times[np.flatnonzero(sel)[np.argmax(sign * boots[:, sel], axis=1)]]
    baseline = (times >= BASELINE[0]) & (times < BASELINE[1])
    return {
        "roi": present, "conditions": list(conditions), "sign": int(sign),
        "n_test": int(len(a)), "n_ref": int(len(b)),
        f"n_{test}": int(len(a)), f"n_{ref}": int(len(b)),
        # With per-channel exclusion a trial is not all-or-nothing: these say how
        # much of the ROI each surviving trial actually had.
        "roi_channels": len(present),
        "roi_channels_per_trial_median": float(np.median(np.r_[na, nb])) if len(na) + len(nb) else 0.0,
        "roi_channels_per_trial_min": int(np.r_[na, nb].min()) if len(na) + len(nb) else 0,
        "trials_without_enough_roi": [int((~ka).sum()), int((~kb).sum())],
        "difference_uv": float(diff[sel].mean()),
        "ci95_uv": [float(lo[sel].mean()), float(hi[sel].mean())],
        "window_s": list(window),
        "peak_latency_s": float(times[peak_i]),
        "peak_latency_ci95_s": [float(v) for v in np.percentile(peak_lat, [2.5, 97.5])],
        "peak_uv": float(diff[peak_i]),
        # A component has a width.  Anything much narrower than its literature
        # width is a transient that argmax has found; see peak_width.
        "peak_fwhm_s": peak_width(times, sign * diff, peak_i),
        "baseline_rms_uv": float(np.sqrt((a.mean(0)[baseline] ** 2).mean())),
        "plus_minus_rms_uv": plus_minus_rms(a, baseline, rng),
        "_times": times, "_test": boot_mean_ci(a, rng), "_ref": boot_mean_ci(b, rng),
        "_diff": (diff, lo, hi), "_peak_i": peak_i,
    }


def topography(epochs: mne.Epochs, peak_i: int, reference: str,
               conditions: tuple[str, str] = ("go", "nogo")) -> dict:
    """The difference at every good channel at one sample.

    All-same-sign is a statement about coverage, not about the component.  The
    EPOC X ring sits entirely above the P3's null, so every one of its 14 sensors
    is positive at the peak; restricting the 32-channel Flex cap to the same 14
    sites in its own recorded reference reproduces that, while the full cap shows
    the component falling to zero at the mastoid ring.  Under an average
    reference the channels sum to zero by construction, so the same-sign test
    cannot be positive and says nothing -- ``reference`` is the policy actually
    applied, so the number is not read as a finding.
    """

    test, ref = conditions
    ea, eb = epochs[test], epochs[ref]
    a = ea.get_data()[:, :, peak_i] * 1e6
    b = eb.get_data()[:, :, peak_i] * 1e6
    ga, gb = _usable(ea, ea.ch_names), _usable(eb, eb.ch_names)
    # Each channel is averaged over its own usable trials, so n differs between
    # channels; it is reported per channel rather than assumed uniform.
    per, n_per = {}, {}
    for j, c in enumerate(epochs.ch_names):
        va, vb = a[ga[:, j], j], b[gb[:, j], j]
        per[c] = float(va.mean() - vb.mean()) if len(va) and len(vb) else float("nan")
        n_per[c] = [int(len(va)), int(len(vb))]
    return {"reference": reference, "conditions": list(conditions),
            "same_sign_test_meaningful": not str(reference).startswith("average"),
            "per_channel_uv": per, "per_channel_n": n_per,
            "channel_mean_uv": float(np.nanmean(list(per.values()))),
            "all_channels_same_sign": bool(all(v > 0 for v in per.values())
                                           or all(v < 0 for v in per.values()))}


def by_block(epochs: mne.Epochs, roi, rng, window: tuple[float, float],
             conditions: tuple[str, str] = ("go", "nogo"), sign: int = +1,
             clusters: bool = True, bad_cells: str = "complete") -> dict:
    """The same contrast within each block, plus that block's own cluster test.

    ``responding`` says whether a key was pressed in that block, where the task
    records it.  A block with no response contains no motor activity, so a P3
    that survives there is locked to the sound and not to the keypress.  Read
    ``peak_fwhm_s`` before believing a block-level peak: on
    epoc_gng_timingfixed.xdf the no-response block's maximum is 39 ms wide and
    its mean over the window is negative, so it is not a P3 at all, while the
    responding block's is 172 ms wide.  Thirty-odd trials is not enough for this
    comparison to settle anything on its own, and a block with no cluster is a
    null rather than evidence against the component.
    """

    out = {}
    test, ref = conditions
    for block in sorted(epochs.metadata["block"].unique()):
        sel = epochs[epochs.metadata["block"] == block]
        if not len(sel[test]) or not len(sel[ref]):
            continue
        res = contrast(sel, roi, rng, window, conditions, sign)
        res["block"] = int(block)
        res["responding"] = (bool(sel.metadata["responding"].any())
                             if "responding" in sel.metadata else None)
        res["clusters"] = (cluster_test(sel[test], sel[ref], tmin=0.0, tmax=1.0,
                                        bad_cells=bad_cells)["clusters"] if clusters else [])
        out[f"block{block}"] = res
    return out
