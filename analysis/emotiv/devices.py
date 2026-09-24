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

    # "recorded", "average", or a tuple of channel names for a linked reference
    # (the mean of those channels is subtracted from every channel).  A linked
    # reference needs its own electrodes to be good, so ``reference_fallback``
    # says what to use when one of them is flagged bad; the choice actually
    # applied is always recorded in ``Cleaned.parameters``.
    reference: str | tuple[str, ...]
    reference_why: str
    bad_channels: str  # "drop" or "interpolate"
    ica_band: tuple[float, float]
    erp_band: tuple[float, float]
    epoch: tuple[float, float]
    baseline: tuple[float, float]
    # 1-s peak-to-peak thresholds, in microvolts, scaled to this headset's noise floor.
    gross_uv: float  # excluded from ICA and final epoch analysis
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
    # Used only when ``reference`` is a linked pair and one of its electrodes is bad.
    reference_fallback: str = "average"
    # Artifact detection runs in this reference, not in the analysis one, so which
    # windows are rejected does not depend on how the ERP is referenced.  It
    # matters whenever the analysis reference is a small set of electrodes: a
    # linked mastoid pushes its own electrodes' noise into every channel, and a
    # threshold applied after it would reject on the reference's quality rather
    # than on the data's.  None means "the analysis reference".
    detection_reference: str | tuple[str, ...] | None = None
    # What to do about a *single epoch* on an otherwise good electrode, where the
    # methods that need a rectangular trials x times x channels block (the
    # cluster test) cannot represent a missing cell.  This is a separate decision
    # from ``bad_channels``, which is about an electrode that is bad throughout.
    # "complete" uses only the epochs with every channel usable; "interpolate"
    # fills the cell from its neighbours and keeps the epoch.  It follows the
    # density of the cap, not the analysis: splines over the EPOC X's 13-sensor
    # ring are guesswork, while on the 32-electrode Flex cap a sporadic cell has
    # close neighbours.  ROI measurements never need this -- they average over
    # whichever channels survived and report a per-channel trial count.
    bad_cells: str = "complete"
    # Two guards on the per-channel exclusion, so it degrades into whole-epoch
    # and whole-channel rejection instead of quietly measuring a stump.
    #
    # Drop the epoch when more than this share of the analysed channels are
    # unusable in it.  ``preprocess.RESIDUAL_MIN_CHANNELS`` already annotates any
    # window where two channels are over *at the same instant*, so what reaches
    # here is channels failing at different moments inside one epoch; this is the
    # backstop for that, in the spirit of autoreject's kappa.
    epoch_max_bad_share: float = 0.25
    # Mark a channel bad outright when it is unusable in more than this share of
    # epochs, and re-epoch with it dropped or interpolated per ``bad_channels``.
    # This is the epoch-level twin of ``preprocess.CHANNEL_BAD_SHARE``, which
    # works on task windows; an epoch spans more than one window plus its padding,
    # so the same electrode scores two to three times as high here, and an
    # electrode can fail this test while passing that one.
    #
    # 10% is calibrated, not picked.  It flags exactly two electrodes across the
    # four sessions and both are known bad: gng.xdf's F4 at 11.0% (the failing
    # saline pad that had to be hand-listed in MANUAL_BADS -- its window share is
    # only 4.2%, under CHANNEL_BAD_SHARE, so this test is what recovers it
    # independently; next worst in that session 1.8%) and oddball_flex's FT10 at
    # 10.4% (next worst 6.6%).  The FT10 margin is the tight one; if a future
    # session puts a good electrode near 10% this cut needs re-deriving rather
    # than nudging.
    channel_max_bad_epoch_share: float = 0.10


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
        "positive at the Go-NoGo peak, channel mean +4.2 uV.  That all-positive pattern "
        "is the montage, not an artifact: restricting gng_flex.xdf to these 14 sites in "
        "its recorded reference reproduces it (14/14 positive, +2.9 uV), while the full "
        "32-channel cap shows the same component falling to zero at the mastoid ring."),
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
# Derived on the Flex, NOT carried over from the EPOC X: this headset uses a
# wider ERP band, interpolated bad channels and a rail-fraction test.  The
# thresholds below were re-derived on data/GnG/gng_flex.xdf the way the EPOC X's
# were -- the 99th percentile of clean-channel 1-s peak-to-peak on task time
# after ICA, via preprocess.sliding_p2p.
FLEX_PROCESSING = Processing(
    reference="recorded",
    # The recorded reference needs no electrode that might be bad, so the
    # fallback never fires; it is "recorded" rather than "average" so that a
    # session overriding the reference with a linked pair degrades to something
    # that preserves broad components instead of to the one reference that does
    # not.  Never fall back to the average reference on this cap.
    reference_fallback="recorded",
    # Artifact detection stays in the average reference whatever the analysis
    # reference is, so the surviving trials are a property of the data.  With a
    # single-ended reference that also stops the CMS electrode's own noise from
    # driving rejection; when this was coupled it cost 8 of 62 Go trials.
    detection_reference="average",
    reference_why=(
        "TP9 is the CMS reference and TP10 the DRL driven ground on this cap, so the "
        "recorded reference already IS a left-mastoid reference -- the same scheme the "
        "EPOC X uses, which makes the two headsets directly comparable.  TP9/TP10 are "
        "not available as data; FT9/FT10 and PO9/PO10 are the nearest sites to the "
        "mastoids but are ordinary electrodes over cortex.\n"
        "The choice that matters is average vs anything else.  All 32 electrodes sit "
        "above the P3's null, so the average reference subtracts a large positive "
        "spatial mean at the peak and takes most of the component with it.  On "
        "gng_flex.xdf, with artifact detection held fixed so all 62 Go trials survive "
        "in every condition, the central-ROI Go-NoGo peak and its +/-average SNR are: "
        "recorded TP9 4.79 uV (4.2), average 1.85 (2.2), linked FT9+FT10 4.97 (3.3), "
        "linked PO9+PO10 4.59 (5.1), linked T7+T8 2.45 (1.8), linked P7+P8 1.77 (3.0).  "
        "Every reference over sites near the null agrees on ~4.4-5.0 uV; the average is "
        "the outlier, and T7/T8 and P7/P8 fail because they carry 1.8-4.4 uV of the "
        "component themselves.\n"
        "Recorded is the default because it is the only candidate that cannot fail: "
        "PO9 and PO10 are the two flakiest electrodes on this cap in both sessions "
        "recorded so far (1.6/1.2 pops per minute on gng_flex.xdf, 1.9/2.5 and flagged "
        "bad outright on oddball_flex.xdf), which is where the cap seats worst, and "
        "FT9/FT10 are reliable but FT9 itself carries +1.9 uV of the P3.  The one thing "
        "recorded gives up is that a single-ended reference puts the CMS electrode's "
        "own noise into every channel: the NoGo N1/P2 peak-to-peak SNR is 4.6, against "
        "12.9 for linked PO9+PO10 and 5.7 for FT9+FT10.  For an N1/P2-only analysis on "
        "a session where the posterior electrodes measured well, --reference PO9,PO10 "
        "is worth taking; for the P3, and for anything compared with the EPOC X, use "
        "the default."),
    bad_channels="interpolate",
    bad_cells="interpolate",
    ica_band=(1.0, 30.0),
    erp_band=(0.1, 30.0),
    epoch=(-0.2, 1.0),
    baseline=(-0.2, 0.0),
    gross_uv=300.0,  # the 99.9th percentile of 1-30 Hz task p2p is 299 uV
    # A physiological ceiling for "too contaminated to contribute to an ERP", not
    # a percentile of this session: the EPOC X's 100 uV happens to sit at its own
    # p99 (104), whereas gng_flex.xdf's p99 is 281 uV in the detection reference,
    # so 120 costs it ~17% of task windows against the EPOC X's ~5%.  That gap is
    # the recording, not the threshold -- this cap's saline pads were drying, the
    # posterior ones worst.  Do not raise it to match the session; re-wet instead.
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
    source="re-derived on data/GnG/gng_flex.xdf (2026-09-16): reference chosen by "
           "effect-to-noise on the P3 and the N1/P2 and by electrode reliability "
           "across gng_flex.xdf and oddball_flex.xdf, thresholds from post-ICA "
           "task-time 1-s p2p percentiles",
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
