"""Load an Emotiv XDF recording into a common ``Run``, whatever the headset.

This layer knows about headsets and nothing about experiments.  It does not look
for task markers, does not read a PsychoPy CSV and does not assume a recording
contains any particular block, so it works on a new paradigm without edits.
Turning markers into events is ``events.py``; scaling, referencing and epoching
are ``preprocess.py`` and ``erp.py``.

The two headsets need different work to get a trustworthy time axis:

* **EPOC X** streams arrival times and does not fill lost samples, so the regular
  sample grid is rebuilt from the packet counter and the arrival times are fitted
  against it.
* **Flex 1.0** fills lost samples itself and flags them.  Its packet timestamps
  are fit over the whole recording to measure the effective rate; a regular
  grid at that rate is exposed for event matching, while the fill flags become
  BAD annotations downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyxdf

from . import devices
from .devices import FS, Device

__all__ = ["Run", "load", "load_epocx", "load_flex", "epocx_grid", "robust_line",
           "stream_labels", "FLEX_DISPLAY_STREAM"]

# The Flex reader publishes a second, viewer-only copy of its EEG with a one-pole
# high pass on top of the headset's own.  It is named, typed and channelled
# exactly like the real stream, so it is refused here by name rather than left to
# be picked up by accident: analysing it would cost slow ERP components amplitude
# and add an undershoot after every large deflection.
FLEX_DISPLAY_STREAM = "Epoc Flex 1.0 Display"


def stream_labels(stream: dict) -> list[str]:
    return [c["label"][0] for c in stream["info"]["desc"][0]["channels"][0]["channel"]]


@dataclass
class Run:
    """One recording on a common time base.

    ``t`` is seconds from the first sample, and marker times use the same raw-EEG
    origin after the fixed hardware delay has been applied.
    """

    device: str
    key: str
    t: np.ndarray  # seconds from the first sample
    x_uv: np.ndarray  # samples x channels, microvolts as published
    labels: list[str]
    diag: dict[str, np.ndarray]
    markers: pd.DataFrame  # time, value
    cq: pd.DataFrame | None = None  # Cortex contact quality
    eq: pd.DataFrame | None = None  # Cortex EEG quality
    pow: pd.DataFrame | None = None  # Cortex band power
    extra: dict = field(default_factory=dict)  # device-specific loader diagnostics
    # Effective rate fitted from the complete recorded EEG time base.  ``FS`` is
    # the device's nominal rate; this is the rate that maps sample indices to the
    # LSL clock used by the markers.
    sfreq_hz: float = FS
    # Flex decoder settings, read from the stream's cap metadata when present.
    # The reader gained that metadata in the same commit (a3f1084) that moved the
    # delta zero to 63 and turned the DC restore off, so a file carrying neither
    # entry is necessarily pre-fix: the defaults below decode it the way it was
    # recorded.  Current recordings publish dc_restore_hz=0.0 (leak 1.0, a pure
    # accumulator) and delta_zero=63.
    dc_restore_hz: float = 0.16
    delta_zero: int = 64

    @property
    def dev(self) -> Device:
        return devices.get(self.device)

    @property
    def filled(self) -> np.ndarray:
        """Samples the reader had to invent (Flex fill, or EPOC X grid gaps)."""

        return self.diag["FILLED"].astype(bool)

    @property
    def leak(self) -> float:
        """Flex DC-restore leak per sample."""

        return math.exp(-2 * math.pi * self.dc_restore_hz / self.sfreq_hz)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    @property
    def sfreq(self) -> float:
        """Alias matching MNE's ``info['sfreq']`` terminology."""

        return float(self.sfreq_hz)

    @property
    def marker_shift_s(self) -> float:
        """Fixed hardware delay already applied to every loaded marker."""

        return float(self.extra.get("marker_shift_s", 0.0))

    def shift_markers(self, delta_s: float) -> None:
        """Apply an explicit marker-time adjustment to every marker in-place."""

        if delta_s:
            self.markers = self.markers.copy()
            self.markers["time"] += float(delta_s)
            self.extra["marker_shift_s"] = self.marker_shift_s + float(delta_s)


# ---------------------------------------------------------------------------
# Shared XDF helpers


def _open(path: str) -> dict:
    streams, _ = pyxdf.load_xdf(path, synchronize_clocks=True, dejitter_timestamps=False)
    by_name: dict[str, list[dict]] = {}
    for stream in streams:
        by_name.setdefault(stream["info"]["name"][0], []).append(stream)
    return by_name


def _markers(by_name: dict, t0: float, shift_s: float = 0.0) -> pd.DataFrame:
    # LabRecorder can pick the same marker outlet up twice; keep one copy of each.
    rows = set()
    for stream in by_name.get("PsychoPy Markers", []):
        for ts, value in zip(stream["time_stamps"], stream["time_series"]):
            # Markers are sent at the physical event, while the EEG sample that
            # carries that event appears after the fixed hardware chain delay.
            # Shift every marker once here so block, rest and stimulus markers all
            # share raw EEG time.  Session-specific audio corrections remain an
            # explicit per-epoch addition in the task scripts.
            rows.add((round(float(ts) - t0 + shift_s, 4), str(value[0])))
    return pd.DataFrame(sorted(rows), columns=["time", "value"])


def _quality(by_name: dict, prefix: str, t0: float) -> dict[str, pd.DataFrame | None]:
    out = {}
    for suffix, short in [("Contact Quality", "cq"), ("EEG Quality", "eq"), ("Band Power", "pow")]:
        streams = by_name.get(f"{prefix} {suffix}", [])
        if streams and len(streams[0]["time_stamps"]):
            frame = pd.DataFrame(np.asarray(streams[0]["time_series"], float),
                                 columns=stream_labels(streams[0]))
            frame.insert(0, "time", np.asarray(streams[0]["time_stamps"]) - t0)
            out[short] = frame
        else:
            out[short] = None
    return out


# ---------------------------------------------------------------------------
# EPOC X


def robust_line(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    keep = np.ones(len(x), bool)
    for _ in range(6):
        fit = np.polyfit(x[keep], y[keep], 1)
        res = y - np.polyval(fit, x)
        keep = np.abs(res - np.median(res[keep])) < tol
    return fit


def _effective_rate(timestamps: np.ndarray, tol: float = 0.004) -> tuple[float, np.ndarray]:
    """Fit one robust whole-recording sample rate and return residuals in seconds."""

    timestamps = np.asarray(timestamps, float)
    if len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("cannot estimate a sample rate from fewer than two finite timestamps")
    index = np.arange(len(timestamps), dtype=float)
    fit = robust_line(index, timestamps, tol)
    residual = timestamps - np.polyval(fit, index)
    return float(1.0 / fit[0]), residual


def epocx_grid(eeg: dict, diag_stream: dict) -> dict:
    """Rebuild the regular sample grid from the counter and fit arrival times against it.

    Returns grid timestamps (absolute LSL seconds), grid samples, the mask of
    interpolated (missing) samples and the fit diagnostics.
    """

    t_arrival = np.asarray(eeg["time_stamps"], float)
    x = np.asarray(eeg["time_series"], float)
    diag_names = stream_labels(diag_stream)
    diag_values = np.asarray(diag_stream["time_series"], float)
    diag_t = np.asarray(diag_stream["time_stamps"], float)

    # Diagnostics rows carry the EEG sample's timestamp; match them up (a recording can
    # start a row later on one stream).
    pos = np.clip(np.searchsorted(diag_t, t_arrival), 0, len(diag_t) - 1)
    pos_prev = np.clip(pos - 1, 0, len(diag_t) - 1)
    pos = np.where(np.abs(diag_t[pos_prev] - t_arrival) < np.abs(diag_t[pos] - t_arrival), pos_prev, pos)
    matched = np.abs(diag_t[pos] - t_arrival) < 5e-4
    col = {name: i for i, name in enumerate(diag_names)}
    counter = np.where(matched, diag_values[pos, col["COUNTER"]], np.nan)
    for i in np.flatnonzero(~matched):  # infer from a neighbour
        if i + 1 < len(counter) and matched[i + 1]:
            counter[i] = (counter[i + 1] - 1) % 128
        elif i > 0 and matched[i - 1]:
            counter[i] = (counter[i - 1] + 1) % 128

    # The two streams can end ragged, leaving trailing samples with no counter on
    # either side.  Casting those NaNs to int would invent a huge counter step and
    # pad the grid with phantom samples, so drop them instead.
    keep = len(counter) - np.argmin(np.isnan(counter[::-1]))
    if keep < len(counter):
        t_arrival, x, matched, counter = t_arrival[:keep], x[:keep], matched[:keep], counter[:keep]
    if np.isnan(counter).any():
        raise ValueError("unresolved packet counters inside the recording")
    counter = counter.astype(int)

    # Regular grid: each report advances by its counter step; a whole lost cycle
    # is invisible to the counter, so long arrival gaps add cycles.
    steps = np.diff(counter) % 128
    steps[steps == 0] = 128
    # This is only used to identify whole lost cycles.  Estimate the cadence from
    # this recording first instead of baking in one EPOC X file's rate.
    rate_guess, _ = _effective_rate(t_arrival)
    gap_periods = np.diff(t_arrival) * rate_guess
    extra_cycles = np.maximum(0, np.round((gap_periods - steps) / 128)).astype(int)
    steps = steps + 128 * extra_cycles
    grid_index = np.r_[0, np.cumsum(steps)]
    n_grid = grid_index[-1] + 1
    missing = np.ones(n_grid, bool)
    missing[grid_index] = False
    x_grid = np.column_stack([np.interp(np.arange(n_grid), grid_index, x[:, j]) for j in range(x.shape[1])])

    # Timestamps: arrival times fitted against the grid, separately across long gaps.
    breaks = np.flatnonzero(extra_cycles > 0) + 1
    seg_edges = [0, *breaks.tolist(), len(grid_index)]
    t_grid = np.empty(n_grid)
    residual_ms = np.empty(len(grid_index))
    rates = []
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):
        fit = robust_line(grid_index[a:b].astype(float), t_arrival[a:b], 0.004)
        lo = grid_index[a]
        hi = grid_index[b] if b < len(grid_index) else n_grid
        t_grid[lo:hi] = np.polyval(fit, np.arange(lo, hi))
        residual_ms[a:b] = 1000 * (t_arrival[a:b] - np.polyval(fit, grid_index[a:b]))
        rates.append(float(1 / fit[0]))
    return {"t": t_grid, "x": x_grid, "missing": missing, "grid_index": grid_index, "breaks": breaks,
            "residual_ms": residual_ms, "rates": rates, "matched": matched, "x_received": x,
            "diag_values": diag_values, "diag_col": col, "rate_guess_hz": rate_guess}


def load_epocx(path: str, key: str = "epocx") -> Run:
    return _epocx(_open(path), key)


def _epocx(by_name: dict, key: str) -> Run:
    eeg = by_name[devices.EPOCX.stream][0]
    labels = stream_labels(eeg)
    if tuple(labels) != devices.EPOCX.labels:
        raise ValueError(f"unexpected EPOC X channel order: {labels}")

    grid = epocx_grid(eeg, by_name["Epoc X Packet Diagnostics"][0])
    t_grid, x_grid = grid["t"], grid["x"]
    t0 = t_grid[0]

    reset = np.zeros(len(t_grid))
    reset[grid["grid_index"][grid["breaks"]]] = 1
    diag = {"FILLED": grid["missing"].astype(float), "RESET_FLAG": reset}

    received = grid["x_received"]
    diag_values, col = grid["diag_values"], grid["diag_col"]
    raw_counts = np.round((received - devices.EPOCX_OFFSET_UV) / devices.EPOCX.lsb_uv)
    extra = {
        "arrival_residual_ms": grid["residual_ms"],
        "arrival_grid_index": grid["grid_index"],
        "diag_rows_unmatched": int((~grid["matched"]).sum()),
        "dropped_repeats": int(((diag_values[:, col["GAP_FLAG"]] > 0)
                                & (diag_values[:, col["MISSING_REPORTS"]] == 0)).sum()),
        "reader_missing_reports": int(diag_values[:, col["MISSING_REPORTS"]].sum()),
        "reader_resets": int(diag_values[:, col["RESET_FLAG"]].sum()),
        "segment_rates_hz": grid["rates"],
        "decoder_max_fraction": float(
            np.abs((received - devices.EPOCX_OFFSET_UV) / devices.EPOCX.lsb_uv - raw_counts).max()),
        "raw_count_range": [int(raw_counts.min()), int(raw_counts.max())],
        "rail_samples": int(((raw_counts <= -32768) | (raw_counts >= 32767)).sum()),
        # Absolute LSL clock of this run's first sample.  ``t`` is deliberately
        # relative, but two recordings made in one sitting share the LSL clock,
        # so this is what lets them be placed on a single wall-clock timeline.
        "lsl_t0": float(t0),
    }
    # ``rates`` is normally one fit per contiguous packet segment.  Keep the
    # whole-recording fit as a fallback for a very short/degenerate recording
    # where no segment clears the fitter's minimum.
    rate = float(np.median(grid["rates"])) if grid["rates"] else float(grid["rate_guess_hz"])
    extra["nominal_srate_hz"] = FS
    extra["effective_srate_hz"] = rate
    extra["rate_residual_ms_percentiles"] = dict(zip(
        ["min", "p1", "p50", "p99", "max"],
        np.percentile(grid["residual_ms"], [0, 1, 50, 99, 100]).round(3).tolist()))
    extra["marker_shift_s"] = float(devices.EPOCX.timing.latency_s)
    quality = _quality(by_name, "Epoc X", t0)
    return Run("epocx", key, t_grid - t0, x_grid, labels, diag,
               _markers(by_name, t0, devices.EPOCX.timing.latency_s),
               quality["cq"], quality["eq"], quality["pow"], extra, rate)


# ---------------------------------------------------------------------------
# Flex 1.0


def load_flex(path: str, key: str = "flex") -> Run:
    return _flex(_open(path), key)


def _flex(by_name: dict, key: str) -> Run:
    if devices.FLEX.stream not in by_name and FLEX_DISPLAY_STREAM in by_name:
        raise ValueError(
            f"this recording has only {FLEX_DISPLAY_STREAM!r}, which is a high-passed "
            "copy for viewers and must not be analysed. Record the plain "
            f"{devices.FLEX.stream!r} stream.")
    eeg = by_name[devices.FLEX.stream][0]
    diag_stream = by_name["Epoc Flex 1.0 Packet Diagnostics"][0]
    t_abs = np.asarray(eeg["time_stamps"], float)
    t0 = t_abs[0]
    diag_values = np.asarray(diag_stream["time_series"], float)
    diag = {name: diag_values[:, i] for i, name in enumerate(stream_labels(diag_stream))}

    rate, residual = _effective_rate(t_abs)
    # Flex timestamps have small packet-level jitter.  The samples themselves are
    # sequential, so expose a regular whole-recording grid for event matching and
    # use the robust fitted rate as the MNE sampling frequency.
    t = np.arange(len(t_abs), dtype=float) / rate
    quality = _quality(by_name, "Epoc Flex 1.0", t0)
    extra = {"lsl_t0": float(t0), "nominal_srate_hz": FS,
             "effective_srate_hz": float(rate),
             # Keep the packet-clock residual separately from the regular grid
             # exposed as ``Run.t``.  The latter is intentionally regular for
             # MNE; the residual is what QC needs to show whether the recorded
             # timestamps stayed linear over the session.
             "timestamp_residual_ms": residual * 1000.0,
             "rate_residual_ms_percentiles": dict(zip(
                 ["min", "p1", "p50", "p99", "max"],
                 np.percentile(residual * 1000, [0, 1, 50, 99, 100]).round(3).tolist())),
             "marker_shift_s": float(devices.FLEX.timing.latency_s)}
    run = Run("flex", key, t, np.asarray(eeg["time_series"], float), stream_labels(eeg),
              diag, _markers(by_name, t0, devices.FLEX.timing.latency_s), quality["cq"],
              quality["eq"], quality["pow"], extra, rate)
    # See the EPOC X loader: the absolute origin, so two sessions recorded on one
    # cap wetting can be put on a common timeline.
    if FLEX_DISPLAY_STREAM in by_name:
        run.extra["ignored_display_stream"] = FLEX_DISPLAY_STREAM
    cap = eeg["info"]["desc"][0].get("cap", [{}])[0]
    if "dc_restore_hz" in cap:
        run.dc_restore_hz = float(cap["dc_restore_hz"][0])
    if "delta_zero" in cap:
        run.delta_zero = int(cap["delta_zero"][0])
    return run


# ---------------------------------------------------------------------------


def load(path: str, key: str | None = None) -> Run:
    """Load a recording, picking the loader from the EEG stream present in the file."""

    by_name = _open(path)
    if devices.EPOCX.stream in by_name:
        return _epocx(by_name, key or "epocx")
    if devices.FLEX.stream in by_name or FLEX_DISPLAY_STREAM in by_name:
        return _flex(by_name, key or "flex")
    raise ValueError(f"no known Emotiv EEG stream in {path}; found {sorted(by_name)}")
