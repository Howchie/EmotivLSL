"""Turn LSL markers into an event table an analysis can epoch on.

Task knowledge lives in the analysis script, not here.  What this module
provides is the mechanical part every paradigm repeats: map a marker time to the
nearest sample, record how far off that sample was, and split marker strings into
conditions.

An event table is a DataFrame with at least:

* ``time``      -- marker time, seconds from the first sample;
* ``sample``    -- nearest sample index;
* ``condition`` -- the label an analysis groups by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loading import Run

__all__ = ["nearest_sample", "events_from_markers", "find_eye_intervals"]

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


def find_eye_intervals(run: Run, min_duration: float = 20.0) -> list[dict]:
    """The one complete 4x closed/open sequence whose intervals all last >= min_duration.

    Returns an empty list when the recording has no eyes closed/open block, so a
    task without one is not an error.
    """

    values = run.markers["value"].tolist()
    times = run.markers["time"].to_numpy()
    for start in range(len(values) - len(EYE_SEQUENCE) + 1):
        if values[start:start + len(EYE_SEQUENCE)] != EYE_SEQUENCE:
            continue
        intervals = []
        for rep in range(4):
            base = start + 4 * rep
            intervals.append({"state": "closed", "pair": rep + 1,
                              "start": times[base], "end": times[base + 1]})
            intervals.append({"state": "open", "pair": rep + 1,
                              "start": times[base + 2], "end": times[base + 3]})
        if min(iv["end"] - iv["start"] for iv in intervals) >= min_duration:
            return intervals
    return []
