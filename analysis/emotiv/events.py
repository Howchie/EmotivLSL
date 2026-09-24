"""Turn LSL markers into an event table an analysis can epoch on.

Task knowledge lives in the analysis script, not here.  What this module
provides is the mechanical part every paradigm repeats: map a marker time to the
nearest sample, record how far off that sample was, and split marker strings into
conditions.

An event table is a DataFrame with at least:

* ``time``      -- marker time in the loader's raw-EEG time base, seconds from
  the first sample (including the fixed hardware marker shift);
* ``sample``    -- nearest sample index;
* ``condition`` -- the label an analysis groups by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loading import Run

__all__ = ["nearest_sample", "events_from_markers", "find_eye_intervals", "paired_blocks", "block_mask"]

EYE_SEQUENCE = ["CloseEyes_Start", "CloseEyes_End", "OpenEyes_Start", "OpenEyes_End"] * 4


def nearest_sample(t: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest sample index for each time, and the signed error in ms."""

    times = np.asarray(times, float)
    idx = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    idx -= (np.abs(t[idx - 1] - times) < np.abs(t[idx] - times)).astype(int)
    return idx, 1000 * (t[idx] - times)


def events_from_markers(run: Run, condition_of, drop_before: float | None = None) -> pd.DataFrame:
    """Build an event table from the run's markers.

    ``condition_of`` is called with each marker string and returns a condition
    name, or ``None`` to ignore that marker.  It is the only place a paradigm's
    naming shows up, so a new task needs a new function here and nothing else.

    ``drop_before`` discards markers earlier than that time, which is how an
    aborted start-of-session run is excluded.
    """

    rows = []
    for time, value in run.markers.itertuples(index=False):
        if drop_before is not None and time < drop_before:
            continue
        condition = condition_of(value)
        if condition is None:
            continue
        rows.append({"time": time, "value": value, "condition": condition})
    frame = pd.DataFrame(rows, columns=["time", "value", "condition"])
    if frame.empty:
        raise ValueError("no markers matched; check condition_of against run.markers['value']")
    frame["sample"], frame["sample_error_ms"] = nearest_sample(run.t, frame["time"].to_numpy())
    return frame


def paired_blocks(run: Run) -> tuple[pd.DataFrame, dict]:
    """Pair ``BlockStart-<id>`` and ``BlockEnd-<id>`` markers safely.

    Markers are processed in timestamp order.  A repeated start for an open
    block replaces the earlier start (the earlier attempt was abandoned); an end
    without an open start is recorded as unmatched.  The returned intervals are
    complete, positive-duration blocks sorted by start time.  Diagnostics are
    returned rather than silently discarded so callers can report a malformed
    marker stream without making the valid intervals unusable.
    """

    pending: dict[str, float] = {}
    intervals: list[dict] = []
    duplicate_starts, unmatched_ends, invalid = [], [], []
    markers = run.markers.sort_values("time")
    for time, value in markers.itertuples(index=False):
        value = str(value)
        if value.startswith("BlockStart-"):
            block_id = value[len("BlockStart-"):]
            if block_id in pending:
                duplicate_starts.append(block_id)
            pending[block_id] = float(time)
        elif value.startswith("BlockEnd-"):
            block_id = value[len("BlockEnd-"):]
            if block_id not in pending:
                unmatched_ends.append(block_id)
                continue
            start = pending.pop(block_id)
            end = float(time)
            if end <= start:
                invalid.append(block_id)
                continue
            intervals.append({"marker_id": block_id, "start": start, "end": end})

    intervals.sort(key=lambda row: row["start"])
    overlap = any(a["end"] > b["start"]
                  for a, b in zip(intervals[:-1], intervals[1:]))
    diagnostics = {
        "duplicate_starts": duplicate_starts,
        "unmatched_starts": sorted(pending),
        "unmatched_ends": unmatched_ends,
        "invalid_intervals": invalid,
        "overlap": bool(overlap),
        "n_complete": len(intervals),
    }
    # Keep non-overlapping complete intervals usable even when one malformed
    # pair overlaps another; callers receive the flag and can decide whether the
    # recording is acceptable.  The mask is a union, so overlap itself cannot
    # double-count task samples.
    return pd.DataFrame(intervals, columns=["marker_id", "start", "end"]), diagnostics


def block_mask(run: Run, pad_s: float = 2.0,
               fallback_times: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return a union-of-blocks mask on the run's measured sample grid.

    Incomplete marker pairs are excluded from the interval union.  If no complete
    block exists, the optional event times provide a conservative envelope
    fallback; this keeps older/partial recordings analyzable while exposing the
    marker problem in the diagnostics.
    """

    intervals, diagnostics = paired_blocks(run)
    mask = np.zeros(len(run.t), bool)
    for row in intervals.itertuples(index=False):
        a, b = np.searchsorted(run.t, [row.start - pad_s, row.end + pad_s])
        mask[a:b] = True
    if not mask.any() and fallback_times is not None and len(fallback_times):
        times = np.asarray(fallback_times, float)
        a, b = np.searchsorted(run.t, [times.min() - pad_s, times.max() + pad_s])
        mask[a:b] = True
        diagnostics["fallback_envelope"] = True
    else:
        diagnostics["fallback_envelope"] = False
    diagnostics["task_samples"] = int(mask.sum())
    return mask, diagnostics


def split_condition(prefix_map: dict[str, str], stims: tuple[str, ...]):
    """``condition_of`` for the common ``"<Prefix>-<stim>"`` marker style.

    ``prefix_map`` maps a marker prefix to a block name; ``stims`` is the set of
    stimulus names accepted after the dash.  Returns ``"block/stim"``.
    """

    def condition_of(value: str) -> str | None:
        prefix, _, stim = str(value).partition("-")
        if prefix in prefix_map and stim in stims:
            return f"{prefix_map[prefix]}/{stim}"
        return None

    return condition_of


def find_eye_intervals(run: Run, min_duration: float = 20.0,
                       require_pair: bool = True) -> list[dict]:
    """Eyes-closed and eyes-open rest intervals, paired where both halves survive.

    The markers come in ``<state>Eyes_Start`` / ``<state>Eyes_End`` pairs, but the
    stream cannot be assumed to be a tidy block of sixteen: a rest block that was
    started, abandoned and restarted leaves several runs behind (gng.xdf has 52
    eye markers and three false starts), and in every recording so far some of
    the four pairs were clicked straight through in well under a second.  So each
    Start is closed by the next End of the **same state**, a second Start of that
    state abandons the first -- the block was restarted and how long the eyes
    were actually shut is then unknown -- and an End with no Start is ignored.
    Intervals shorter than ``min_duration`` are dropped.

    Two earlier versions of this got it wrong in opposite directions and both are
    worth knowing about, because results computed under either are not
    reproducible:

    * requiring all sixteen markers to form one clean sequence returned an empty
      list for **every** recording in this repository, and ``qc.alpha_spectra``
      and ``qc.cortex_band_power`` read an empty list as "this task had no rest
      block", so the eyes-open/closed analysis was skipped in silence;
    * matching a rigid sixteen-marker window anywhere in the stream locks onto
      the wrong offset once a block has been restarted, which made gng.xdf --
      which has a perfectly good 32 s closed / 21 s open pair -- look as though
      it had none.

    ``require_pair`` keeps a closed interval only when its following open
    interval also survived, which is what a paired closed-minus-open contrast
    needs.  Set it False to keep a long interval whose partner was skipped:
    epoc_gng_timingfixed.xdf has a 47 s eyes-closed followed by a 1.5 s "open",
    usable for a noise measurement but not for an alpha contrast.

    Returns an empty list when the recording has no usable rest block, so a task
    without one is not an error.
    """

    pending: dict[str, float | None] = {"closed": None, "open": None}
    found: list[dict] = []
    for time, value in zip(run.markers["time"], run.markers["value"]):
        value = str(value)
        if value.startswith("CloseEyes_"):
            state = "closed"
        elif value.startswith("OpenEyes_"):
            state = "open"
        else:
            continue
        if value.endswith("_Start"):
            pending[state] = float(time)  # a second Start abandons the first
        elif value.endswith("_End") and pending[state] is not None:
            found.append({"state": state, "start": pending[state], "end": float(time)})
            pending[state] = None

    long = [iv for iv in found if iv["end"] - iv["start"] >= min_duration]
    # Number the pairs in order: an open interval joins the closed one before it.
    pair, out = 0, []
    for iv in long:
        if iv["state"] == "closed" or not out or out[-1]["pair"] != pair:
            pair += 1
        out.append({**iv, "pair": pair})
    if not require_pair:
        return out
    counts: dict[int, set] = {}
    for iv in out:
        counts.setdefault(iv["pair"], set()).add(iv["state"])
    return [iv for iv in out if counts[iv["pair"]] == {"closed", "open"}]
