"""Re-decode an EPOC Flex 1.0 XDF recorded with the two's-complement delta bug.

Recordings made before the offset-binary fix in ``emotiv_lsl/emotiv_flex.py``
contain accumulated ADC values built from mis-signed deltas.  The damage is
mostly reversible: the reader's transform (accumulate 7-bit deltas, clamp to
14 bits, scale by 0.51 uV) is invertible almost everywhere, so the original
wire values can be recovered from the stored signal, re-decoded with the
correct rule, and re-accumulated with the DC restore.

Two things cannot be recovered and are marked ``BAD_unrecoverable``:

* rows where the old reader reset the accumulator to the midpoint after a
  packet gap (the true deltas for that report were never stored), and
* samples where the accumulator was clamped at the 14-bit rail, which silently
  discarded part of the delta.

Usage::

    python examples/repair_flex_xdf.py INPUT.xdf [-o OUTPUT_raw.fif] [--dc-restore-hz HZ]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mne
import numpy as np
import pyxdf

LSB_UV = 0.51
ADC_BITS = 14
ADC_MAX = (1 << ADC_BITS) - 1
ADC_MIDPOINT = 1 << (ADC_BITS - 1)
SAMPLE_RATE = 128.0
EEG_STREAM_NAME = "Epoc Flex 1.0"
DEFAULT_DC_RESTORE_HZ = 0.16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="XDF recorded before the delta fix")
    parser.add_argument("-o", "--output", type=Path,
                        help="output FIF (default: alongside the input, *_repaired_raw.fif)")
    parser.add_argument("--dc-restore-hz", type=float, default=DEFAULT_DC_RESTORE_HZ,
                        metavar="HZ",
                        help=f"accumulator high-pass corner (default {DEFAULT_DC_RESTORE_HZ}; "
                             "0 reproduces the unbounded pure accumulator)")
    parser.add_argument("--stream", default=EEG_STREAM_NAME,
                        help=f"EEG stream name (default {EEG_STREAM_NAME!r})")
    return parser.parse_args()


def load_streams(path: Path, stream_name: str):
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=False)
    eeg = None
    markers = []
    for stream in streams:
        info = stream["info"]
        if info["name"][0] == stream_name and info["type"][0] == "EEG":
            eeg = stream
        elif info["channel_format"][0] == "string":
            markers.append(stream)
    if eeg is None:
        names = ", ".join(repr(s["info"]["name"][0]) for s in streams)
        raise SystemExit(f"no EEG stream named {stream_name!r}; found: {names}")
    return eeg, markers


def channel_labels(eeg) -> list[str]:
    try:
        return [c["label"][0] for c in eeg["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, IndexError, TypeError):
        return [f"ch{i}" for i in range(int(eeg["info"]["channel_count"][0]))]


def recover_adc(values: np.ndarray) -> np.ndarray:
    """Undo --remove-dc and the microvolt scaling to get the stored ADC state."""

    adc = values / LSB_UV
    if values.min() < 0:                      # midpoint-subtracted on the wire
        adc = adc + ADC_MIDPOINT
    return adc


def repair(adc: np.ndarray, leak: float) -> tuple[np.ndarray, np.ndarray]:
    """Return repaired microvolts and a per-sample 'unrecoverable' mask."""

    stored = np.rint(adc).astype(np.int64)
    deltas = np.diff(stored, axis=0)

    # A row the old reader reset: at least one channel moved further than the
    # 7-bit delta range allows, which the protocol cannot produce.
    reset_rows = (np.abs(deltas) > 64).any(axis=1)

    # Samples where clamping ate part of the delta.  Either endpoint of the
    # difference sitting on a rail makes that delta untrustworthy.
    railed = (stored <= 0) | (stored >= ADC_MAX)
    clamped = railed[:-1] | railed[1:]

    # Invert the old sign rule, then apply the correct one.
    raw = np.mod(deltas, 128)                 # the unsigned 7-bit wire value
    corrected = (raw - 64).astype(np.float64)
    corrected[reset_rows, :] = 0.0
    corrected[clamped] = 0.0

    state = np.empty((deltas.shape[0] + 1, deltas.shape[1]), dtype=np.float64)
    state[0] = 0.0
    acc = np.zeros(deltas.shape[1], dtype=np.float64)
    for index in range(deltas.shape[0]):
        acc = acc * leak + corrected[index]
        state[index + 1] = acc

    bad = np.zeros_like(state, dtype=bool)
    bad[1:][reset_rows, :] = True
    bad[1:][clamped] = True
    bad[0] = True                             # no delta produced the first sample
    return state * LSB_UV, bad


def to_annotations(bad: np.ndarray, labels: list[str], timestamps: np.ndarray):
    """One BAD_unrecoverable annotation per contiguous run, per channel."""

    onsets, durations, descriptions, ch_names = [], [], [], []
    t0 = timestamps[0]
    for channel in range(bad.shape[1]):
        column = bad[:, channel]
        if not column.any():
            continue
        edges = np.diff(column.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        stops = list(np.where(edges == -1)[0] + 1)
        if column[0]:
            starts.insert(0, 0)
        if column[-1]:
            stops.append(len(column))
        for start, stop in zip(starts, stops):
            onsets.append(timestamps[start] - t0)
            durations.append(max(1, stop - start) / SAMPLE_RATE)
            descriptions.append("BAD_unrecoverable")
            ch_names.append((labels[channel],))
    return onsets, durations, descriptions, ch_names


def main() -> None:
    args = parse_args()
    if args.dc_restore_hz < 0:
        raise SystemExit("--dc-restore-hz must not be negative")
    leak = (1.0 if args.dc_restore_hz == 0
            else math.exp(-2 * math.pi * args.dc_restore_hz / SAMPLE_RATE))

    eeg, marker_streams = load_streams(args.input, args.stream)
    labels = channel_labels(eeg)
    values = np.asarray(eeg["time_series"], dtype=np.float64)
    timestamps = np.asarray(eeg["time_stamps"], dtype=np.float64)
    adc = recover_adc(values)

    print(f"{args.input.name}: {values.shape[0]} samples x {values.shape[1]} channels, "
          f"{(timestamps[-1] - timestamps[0]) / 60:.1f} min")
    print(f"stored ADC range [{adc.min():.0f}, {adc.max():.0f}] "
          f"(valid 0..{ADC_MAX}, midpoint {ADC_MIDPOINT})")

    repaired, bad = repair(adc, leak)

    pct = bad.mean() * 100
    per_channel = bad.mean(axis=0) * 100
    worst = int(np.argmax(per_channel))
    print(f"dc restore {args.dc_restore_hz} Hz (leak {leak:.5f})")
    print(f"unrecoverable samples: {pct:.2f}% overall, worst channel "
          f"{labels[worst]} at {per_channel[worst]:.2f}%")
    before = np.sqrt((values ** 2).mean(axis=0))
    after = np.sqrt((repaired ** 2).mean(axis=0))
    print(f"channel RMS: {np.median(before):.1f} uV before -> "
          f"{np.median(after):.1f} uV after (median over channels)")

    info = mne.create_info(labels, SAMPLE_RATE, ch_types="eeg")
    raw = mne.io.RawArray(repaired.T * 1e-6, info, verbose="ERROR")

    onsets, durations, descriptions, ch_names = to_annotations(bad, labels, timestamps)
    for stream in marker_streams:
        t0 = timestamps[0]
        for value, when in zip(stream["time_series"], stream["time_stamps"]):
            text = value[0] if isinstance(value, (list, tuple, np.ndarray)) else value
            onsets.append(float(when) - t0)
            durations.append(0.0)
            descriptions.append(str(text))
            ch_names.append(())
        print(f"markers from {stream['info']['name'][0]!r}: {len(stream['time_stamps'])}")

    if onsets:
        order = np.argsort(onsets)
        raw.set_annotations(mne.Annotations(
            onset=np.asarray(onsets)[order],
            duration=np.asarray(durations)[order],
            description=np.asarray(descriptions)[order],
            ch_names=[ch_names[i] for i in order],
            orig_time=None,
        ), verbose="ERROR")

    output = args.output or args.input.with_name(args.input.stem + "_repaired_raw.fif")
    raw.save(output, overwrite=True, verbose="ERROR")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
