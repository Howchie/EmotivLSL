"""Canonical loading and processing for Emotiv EPOC X and EPOC Flex 1.0 recordings.

A typical analysis is one import and a handful of calls::

    import emotiv as em

    run = em.load("data/GnG/gng.xdf")
    events = em.events_from_markers(run, em.split_condition({"GnG": "gng"}, ("high", "low")))
    cleaned = em.clean(run)
    epochs, info = em.erp_epochs(run, cleaned, events)

The loader shifts all markers by the headset's fixed chain latency, so t=0 is the
physical stimulus and no analysis needs to apply that correction again.  An
explicit ``chain_latency_s`` passed to ``erp_epochs`` is only an additional
session-specific correction (for example, a known audio-path exception).

**The EPOC X and the Flex 1.0 are different hardware and are not processed the
same way.**  The reference, passbands, artifact thresholds and bad-channel policy
all differ, and each headset's are in ``devices.py`` as a ``Processing`` profile.
The shared code applies whichever profile the run's device carries; it never
assumes one headset's settings are right for the other.  ``devices.py`` is also
where the measured chain latency lives, so a hardware timing re-run is a one-line
change there.

Layers, if you need them directly:

* ``devices``    -- channel layout, scaling, measured chain latency per headset;
* ``loading``    -- XDF to a common ``Run``, including the EPOC X clock rebuild;
* ``events``     -- markers to an event table;
* ``preprocess`` -- continuous MNE data, bad spans, bad channels, spectra;
* ``erp``        -- epoching with timing correction, peak measurement, cluster tests;
* ``qc``         -- acquisition integrity and signal-quality checks.
"""

from __future__ import annotations

from . import devices, erp, events, figures, loading, preprocess, qc
from .devices import DEVICES, EPOCX, FLEX, FS, SEED, Device, Processing, Timing
from .erp import (BASELINE, ERP_WINDOW, boot_diff_ci, boot_mean_ci, bootstrap_latency,
                  by_block, cluster_test, complete_cases, contrast, erp_epochs, n1_p2,
                  peak_latency, peak_width, plus_minus_rms, roi_cluster_test, roi_picks,
                  roi_trials, topography)
from .events import block_mask, events_from_markers, find_eye_intervals, nearest_sample, paired_blocks, split_condition
from .loading import FLEX_DISPLAY_STREAM, Run, epocx_grid, load, load_epocx, load_flex
from .preprocess import (Cleaned, apply_reference, bad_channels, blink_times, channel_noise,
                         channel_spans, clean, clean_psd, contiguous, electrode_pops, eog_proxies, flag_windows,
                         flex_counts, flex_deltas, good_channels, good_mask, line_check,
                         make_raw, rail_fraction, sliding_p2p)
from .figures import COLORS, block_figure, contrast_figure
from .qc import alpha_scores, alpha_spectra, cortex_band_power, integrity, pair_stats, quality_vs_noise
from .util import to_jsonable

__all__ = [
    "devices", "loading", "events", "preprocess", "erp", "figures", "qc",
    "Device", "Timing", "Processing", "Run", "DEVICES", "EPOCX", "FLEX", "FS", "SEED",
    "load", "load_epocx", "load_flex", "epocx_grid", "FLEX_DISPLAY_STREAM",
    "events_from_markers", "split_condition", "find_eye_intervals", "nearest_sample", "paired_blocks", "block_mask",
    "clean", "Cleaned", "make_raw", "good_mask", "good_channels", "channel_noise",
    "bad_channels", "clean_psd", "line_check", "blink_times", "contiguous",
    "sliding_p2p", "flag_windows", "channel_spans", "eog_proxies", "electrode_pops",
    "flex_counts", "flex_deltas", "rail_fraction", "apply_reference",
    "erp_epochs", "roi_picks", "roi_trials", "complete_cases", "peak_latency", "peak_width", "bootstrap_latency", "n1_p2", "cluster_test", "roi_cluster_test",
    "contrast", "topography", "by_block", "contrast_figure", "block_figure", "COLORS",
    "boot_mean_ci", "boot_diff_ci", "plus_minus_rms",
    "ERP_WINDOW", "BASELINE",
    "integrity", "quality_vs_noise", "alpha_spectra", "alpha_scores", "cortex_band_power",
    "pair_stats", "to_jsonable",
]
