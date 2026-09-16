"""Per-device constants: channel layout, scaling, and measured chain latency.

Everything here is a property of the headset, not of any experiment.  The one
thing that has to be measured rather than looked up is ``chain_latency_s``: the
delay between a physical event and the sample that carries it, covering the
analog front end, the radio, the dongle and the onboard filtering.

How the latency is measured
---------------------------
A stimulus with a sharp edge is injected into one sensor and timed against its
own LSL marker, using the energy-centroid method in
``analysis/headphone_timing.py``.  Subtracting any stimulus-side delay leaves the
chain.  ``analysis/drt_timing.py`` does the same with the DRT's electrical
transient.  Both numbers, and the per-trial SD, belong in ``Timing`` below.
"""

from __future__ import annotations

from dataclasses import dataclass

FS = 128.0  # both headsets publish at 128 Hz nominal; the true rate comes out of the loader
SEED = 7


@dataclass(frozen=True)
class Timing:
    """Measured delay from a physical event to the sample carrying it."""

    latency_s: float
    jitter_sd_s: float | None
    source: str
    measured: bool = True

    def describe(self) -> str:
        jitter = "" if self.jitter_sd_s is None else f", SD {1000 * self.jitter_sd_s:.1f} ms"
        state = "measured" if self.measured else "PROVISIONAL"
        return f"{1000 * self.latency_s:.1f} ms{jitter} ({state}; {self.source})"


@dataclass(frozen=True)
class Processing:
    """How this headset's data must be processed.

    These are **not** interchangeable between headsets.  The two devices have
    different electronics, different noise floors and different montage geometry,
    so the reference, the passbands and the artifact thresholds all differ.  The
    shared code in ``preprocess`` and ``erp`` provides the mechanisms; this record
    supplies every number and policy they use.
    """

    reference: str  # "recorded" or "average"
    reference_why: str
    bad_channels: str  # "drop" or "interpolate"
    ica_band: tuple[float, float]
    erp_band: tuple[float, float]
    epoch: tuple[float, float]
    baseline: tuple[float, float]
    # 1-s peak-to-peak thresholds, in microvolts, scaled to this headset's noise floor.
    gross_uv: float  # excluded from the ICA fit only
    residual_uv: float  # annotated BAD_residual after ICA
    pop_uv: float  # electrode-pop diagnostic
    # Flex only.  Its wire format is a seven-bit delta, so the encoder rails at
    # +/-63 counts = 32.1 uV/sample; a channel that rails often is not tracking
    # its electrode.  None on a headset that transmits absolute counts.
    rail_pct: float | None
    fill_pad: tuple[float, float]  # s before/after a filled/missing stretch
    # EOG proxies: neither headset has a dedicated EOG channel.  VEOG is the mean
    # of the frontal pair; HEOG is a left-minus-right frontal pair, or None when
    # the montage has no such pair.
    veog: tuple[str, ...]
    heog: tuple[str, str] | None
    validated: bool
    source: str


@dataclass(frozen=True)
class Device:
    key: str
    stream: str  # XDF stream name
    labels: tuple[str, ...]
    lsb_uv: float
    frontal_roi: tuple[str, ...]
    inferior_roi: tuple[str, ...]
    posterior_roi: tuple[str, ...]
    timing: Timing
    processing: Processing
    # The canonical centro-parietal P3 ROI.  Empty on the EPOC X: its 14 sensors
    # form a ring with no midline site, which is the whole reason that headset is
    # analysed in its recorded reference.  Analyses skip an ROI that is empty or
    # has no channel in the montage.
    central_roi: tuple[str, ...] = ()

    @property
    def n_channels(self) -> int:
        return len(self.labels)

    @property
    def veog(self) -> tuple[str, ...]:
        return self.processing.veog

    @property
    def heog(self) -> tuple[str, str] | None:
        return self.processing.heog


# ---------------------------------------------------------------------------
# Timing constants measured by the hardware tests.  Update the Timing(...) call
# when a new run supersedes one of these references; every analysis picks the
# new value up automatically.

EPOCX_TIMING = Timing(
    latency_s=0.06975544868138632,
    jitter_sd_s=0.001676259854289953,
    source="DRT LED timing, data/DRT_Timing/drt_timing/results.json "
           "(207-trial P7 run; slope-centroid onset, offset and data-measured pulse width)",
    measured=True,
)

# Both headsets use the EPOC X number, by decision.  The Flex has now been
# measured directly -- 64.8 ms, 95% CI [48.0, 67.5] from 87 usable trials of
# data/DRT_Timing/drt_timing_flex_onhead.xdf -- but its seven-bit encoder rails
# the electrode the DRT is clipped to, so only the neighbours carry a usable edge
# and the interval is 20x wider than the EPOC X's.  The two estimates differ by
# ~6 ms, well under one sample at 128 Hz.  Deriving this from EPOCX_TIMING means
# the DRT re-run updates both headsets at once.
FLEX_TIMING = Timing(
    latency_s=EPOCX_TIMING.latency_s,
    jitter_sd_s=EPOCX_TIMING.jitter_sd_s,
    source="EPOC X DRT LED timing, applied to both headsets; the Flex's own "
           "87-trial DRT run gave 64.8 ms [48.0, 67.5], which agrees to within "
           "one sample at 128 Hz",
    measured=True,
)
# ---------------------------------------------------------------------------


EPOCX_PROCESSING = Processing(
    reference="recorded",
    reference_why=(
        "The 14 sensors form a ring with no midline site, so a broad vertex-centred "
        "component such as the P3 reaches all of them.  An average reference subtracts "
        "most of it and flips P7/P8/O1/O2 negative.  Measured on gng.xdf: every channel "
        "positive at the Go-NoGo peak, channel mean +4.2 uV."),
    bad_channels="drop",  # spline interpolation from 13 sparse sensors blurs the ROIs
    ica_band=(1.0, 30.0),
    erp_band=(0.1, 20.0),  # 20-30 Hz here is mostly EMG and sensor noise
    epoch=(-0.3, 1.0),
    baseline=(-0.2, 0.0),
    gross_uv=300.0,
    residual_uv=100.0,  # sits above the 99th percentile of clean-channel 1-s p2p (75-95 uV)
    pop_uv=60.0,
    rail_pct=None,  # absolute 16-bit counts, no delta encoder to rail
    # No DC-restore leak on this headset, so a grid gap needs no long recovery
    # tail; this matches the artifact-window pad rather than the Flex fill pad.
    fill_pad=(0.25, 0.5),
    veog=("AF3", "AF4"),
    heog=("F7", "F8"),
    validated=True,
    source="EPOCX_ERP_PIPELINE.md, derived on data/GnG/gng.xdf",
)

# ---------------------------------------------------------------------------
# Derived on the Flex, NOT carried over from the EPOC X: this headset uses an
# average reference, a wider ERP band and interpolated bad channels.  The
# thresholds below were re-derived on data/GnG/gng_flex.xdf the way the EPOC X's
# were -- the 99th percentile of clean-channel 1-s peak-to-peak on task time
# after ICA, via preprocess.sliding_p2p.
FLEX_PROCESSING = Processing(
    reference="average",
    reference_why=(
        "Measured on the oddball runs: the average reference roughly tripled SNR "
        "against the recorded TP9 reference, the opposite of the EPOC X.  The cap "
        "covers the head with midline sites, so averaging does not sit on top of "
        "the component of interest.  Re-checked on a P3, the component an average "
        "reference is most likely to erase: on gng_flex.xdf the Go-NoGo difference "
        "over the central ROI is +0.58 uV [-0.60, +1.72] in the average reference "
        "against +0.14 uV [-2.91, +3.02] in the recorded one -- the recorded "
        "reference has the larger peak but a 2.5x wider interval, and every ROI "
        "peaks on the same sample there, which is the signature of a common-mode "
        "component rather than a topography."),
    bad_channels="interpolate",
    ica_band=(1.0, 30.0),
    erp_band=(0.1, 30.0),
    epoch=(-0.2, 1.0),
    baseline=(-0.2, 0.0),
    gross_uv=300.0,  # the 99.9th percentile of 1-30 Hz task p2p is 299 uV
    # Per-channel 99th percentiles of post-ICA task-time 1-s p2p run 24-109 uV
    # (median 54), so 120 clears every clean channel.  flex_sanity's flat 150 uV
    # was inherited, not measured, and let through windows this now annotates.
    residual_uv=120.0,
    pop_uv=60.0,
    # Rail fraction is the Flex's own dead-electrode flag and catches things the
    # 20-40 Hz noise test misses: a dead or badly seated electrode rails the
    # seven-bit encoder almost continuously.  Measured spread: every channel of
    # gng_flex.xdf rails under 2.1% of samples and the on-head DRT run's 90th
    # percentile is 0.55%, while the DRT contact electrode rails 86% and a badly
    # positioned Oz railed 82% at rest.  Frontal channels rail 1.5-2.7% on blinks,
    # which is real signal, so the cut sits well above them.
    rail_pct=10.0,
    # The reader decodes with zero at 63 and no DC restore (a3f1084), so a lost
    # 8-sample block leaves a permanent ~18 uV step rather than a 1-s decay.  A
    # step is almost pure DC: the 0.1 Hz high pass and the per-epoch baseline
    # remove nearly all of it, leaving ~1.9 uV in the nearest surviving epoch
    # whatever the tail length.  So no long recovery tail, same as the EPOC X.
    # (flex_sanity's 1.0 s tail was sized for the retired 0.16 Hz leak.)
    fill_pad=(0.25, 0.5),
    veog=("Fp1", "Fp2"),
    # The cap is re-montaged per session (electrodes are named LA..RQ and mapped
    # to 10-20 labels), so there is no fixed left/right frontal pair to use.  Set
    # this per session once the montage is known, or saccade removal will not run.
    # F7/F8 are the default because every montage used so far has had them; if a
    # session's cap does not, eog_proxies warns rather than skipping silently.
    heog=("F7", "F8"),
    validated=True,
    source="re-derived on data/GnG/gng_flex.xdf (2026-09-16): reference checked "
           "against a P3, thresholds from post-ICA task-time 1-s p2p percentiles",
)
# ---------------------------------------------------------------------------


EPOCX = Device(
    key="epocx",
    stream="Epoc X",
    labels=("AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"),
    lsb_uv=0.128205128205129,
    frontal_roi=("F3", "F4", "FC5", "FC6"),
    inferior_roi=("T7", "T8", "P7", "P8"),
    posterior_roi=("O1", "O2", "P7", "P8"),
    timing=EPOCX_TIMING,
    processing=EPOCX_PROCESSING,
)

FLEX = Device(
    key="flex",
    stream="Epoc Flex 1.0",
    labels=(),  # the Flex cap is re-montaged per session; labels come from the stream
    lsb_uv=0.51,
    # Four non-overlapping ROIs, which the 14-sensor EPOC X ring cannot provide.
    frontal_roi=("Fz", "FC1", "FC2", "F3", "F4"),
    inferior_roi=("FT9", "FT10", "PO9", "PO10"),  # nearest sites to the mastoids
    posterior_roi=("O1", "Oz", "O2"),
    timing=FLEX_TIMING,
    processing=FLEX_PROCESSING,
    central_roi=("Cz", "CP1", "CP2", "Pz"),  # the canonical centro-parietal P3
)

DEVICES = {d.key: d for d in (EPOCX, FLEX)}

# The EPOC X has no AF3/AF4 equivalent on the Flex cap.
FLEX_EQUIVALENT = {"AF3": "Fp1", "AF4": "Fp2"}

# EPOC X reader constants.
EPOCX_OFFSET_UV = 4201.02564096001  # the reader's value for a raw count of 32768

# Flex reader constants.  The reader publishes ``adc * LSB_UV`` with the 14-bit
# midpoint still in it, so recovering the transmitted deltas means taking it out.
FLEX_ADC_MIDPOINT = 8192  # 1 << 13
FLEX_DELTA_LIMIT = 63     # the seven-bit encoder's negative endpoint; positive is +64


def get(key: str) -> Device:
    if key not in DEVICES:
        raise KeyError(f"unknown device {key!r}; known: {sorted(DEVICES)}")
    return DEVICES[key]
