from __future__ import annotations

from dataclasses import dataclass, replace
import sys
import time

import hid
from pylsl import StreamInfo, StreamOutlet


PACKET_DIAGNOSTIC_CHANNELS = (
    ("COUNTER", "count", "Raw packet counter from the decrypted HID report."),
    ("EXPECTED_COUNTER", "count", "Counter expected after the previous report."),
    ("GAP_FLAG", "boolean", "One or more packet counters were skipped or duplicated."),
    ("MISSING_REPORTS", "count", "Estimated reports missing immediately before this report."),
    ("CUMULATIVE_MISSING", "count", "Estimated missing reports since this reader started."),
    ("RESET_FLAG", "boolean", "Packet counter restarted without a normal modulo wrap."),
    ("CUMULATIVE_GAPS", "count", "Counter discontinuity events since this reader started."),
    ("CUMULATIVE_RESETS", "count", "Counter restart events since this reader started."),
)


@dataclass(frozen=True)
class PacketDiagnostics:
    """Packet-sequence state emitted alongside each decoded EEG sample."""

    counter: int
    expected_counter: int
    gap: bool
    missing_reports: int
    cumulative_missing: int
    reset: bool
    cumulative_gaps: int
    cumulative_resets: int
    # Console-only: this report was a dropped repeat, not a published sample.
    repeat: bool = False

    def as_lsl_sample(self) -> list[float]:
        return [
            float(self.counter),
            float(self.expected_counter),
            float(self.gap),
            float(self.missing_reports),
            float(self.cumulative_missing),
            float(self.reset),
            float(self.cumulative_gaps),
            float(self.cumulative_resets),
        ]


class PacketCounterTracker:
    """Track a wrapping HID packet counter without altering EEG samples.

    ``reset_hint`` is used by protocols that emit a recognizable baseline
    packet when their counter restarts.  A restart is reported separately from
    packet loss so a reconnect is not mistaken for dozens of missing reports.
    """

    def __init__(self, modulus: int) -> None:
        if modulus < 2:
            raise ValueError("packet counter modulus must be at least 2")
        self.modulus = int(modulus)
        self.last_counter: int | None = None
        self.cumulative_gaps = 0
        self.cumulative_missing = 0
        self.cumulative_resets = 0
        self.cumulative_repeats = 0
        self._flag_next_sample = False

    def skip_repeat(self, counter: int) -> PacketDiagnostics:
        """Account for a dropped byte-identical repeat of the previous report.

        The repeat counts as one discontinuity with nothing missing.  It is not
        published, so its GAP_FLAG is carried to the next published sample.
        """
        diagnostics = replace(self.update(counter), repeat=True)
        self.cumulative_repeats += 1
        self._flag_next_sample = True
        return diagnostics

    def update(self, counter: int, *, reset_hint: bool = False) -> PacketDiagnostics:
        counter = int(counter)
        if not 0 <= counter < self.modulus:
            raise ValueError(
                f"packet counter {counter} is outside 0..{self.modulus - 1}"
            )

        if self.last_counter is None:
            expected = counter
            gap = False
            missing = 0
            reset = False
        else:
            expected = (self.last_counter + 1) % self.modulus
            delta = (counter - self.last_counter) % self.modulus
            reset = bool(reset_hint and counter != expected)
            if reset:
                gap = False
                missing = 0
                self.cumulative_resets += 1
            else:
                gap = delta != 1
                # delta == 0 is a duplicate/reordered report, not a skipped
                # report.  It is still a discontinuity worth flagging.
                missing = max(0, delta - 1)
                if gap:
                    self.cumulative_gaps += 1
                    self.cumulative_missing += missing

        if self._flag_next_sample:
            # The previous report was a dropped repeat; flag this sample instead.
            gap = True
            self._flag_next_sample = False
        self.last_counter = counter
        return PacketDiagnostics(
            counter=counter,
            expected_counter=expected,
            gap=gap,
            missing_reports=missing,
            cumulative_missing=self.cumulative_missing,
            reset=reset,
            cumulative_gaps=self.cumulative_gaps,
            cumulative_resets=self.cumulative_resets,
        )


class RepeatedReportFilter:
    """Recognize a report delivered twice in a row, byte for byte.

    EPOC X headsets on firmware 0x720 and 0x740 occasionally hand the host the
    previous report again before the next one arrives.  In recordings with
    diagnostics, the counter never skips around a repeat.  The rate is also
    identical across recordings (128.066 Hz) once repeats are removed, so the
    copy is an extra, not a stand-in for a lost sample.  Publishing it inserts
    a fake sample.  Consecutive real samples always differ, because the
    counter advances.
    """

    def __init__(self) -> None:
        self._previous: bytes | None = None

    def is_repeat(self, report) -> bool:
        report = bytes(report)
        repeat = report == self._previous
        self._previous = report
        return repeat


class PacketLossReporter:
    """Rate-limited console summary of packet-counter discontinuities.

    The diagnostics LSL stream carries the full per-sample detail.  This only
    exists so that a run started from a launcher script shows packet loss in
    the console instead of requiring an LSL consumer to notice it.
    """

    def __init__(self, label: str, interval_seconds: float = 10.0) -> None:
        self.label = label
        self.interval_seconds = float(interval_seconds)
        self._next_gap_report = 0.0
        self._reported_a_gap = False

    def _emit(self, message: str) -> None:
        print(f"{self.label}: {message}", file=sys.stderr, flush=True)

    def report(self, diagnostics: PacketDiagnostics) -> None:
        if diagnostics.reset:
            self._emit(
                f"packet counter restarted at {diagnostics.counter} "
                f"(expected {diagnostics.expected_counter}); "
                f"{diagnostics.cumulative_resets} restart(s) so far"
            )
            return
        if not diagnostics.gap:
            return
        if (
            not diagnostics.repeat
            and not diagnostics.missing_reports
            and diagnostics.counter == diagnostics.expected_counter
        ):
            # A GAP_FLAG carried over from a dropped repeat, which was already
            # reported when it was dropped.
            return

        # Always announce the first gap, then summarize periodically so a bad
        # radio link cannot flood the console at the sample rate.
        now = time.monotonic()
        if self._reported_a_gap and now < self._next_gap_report:
            return
        self._reported_a_gap = True
        self._next_gap_report = now + self.interval_seconds
        if diagnostics.repeat:
            message = (
                f"dropped a repeated report: counter {diagnostics.counter} arrived twice "
                "with identical bytes (an extra copy; no reports missing)"
            )
        elif diagnostics.missing_reports:
            message = (
                f"dropped packets: counter {diagnostics.counter} arrived where "
                f"{diagnostics.expected_counter} was expected "
                f"({diagnostics.missing_reports} report(s) missing)"
            )
        else:
            message = (
                f"packet counter discontinuity: counter {diagnostics.counter} arrived "
                f"where {diagnostics.expected_counter} was expected "
                "(duplicate or reordered report; no reports missing)"
            )
        self._emit(
            f"{message}; {diagnostics.cumulative_gaps} discontinuity event(s) and "
            f"{diagnostics.cumulative_missing} missing report(s) so far"
        )


class SampleRateMonitor:
    """Warn once if a stream's real rate disagrees with its declared rate.

    An LSL outlet's nominal rate is fixed when the outlet is created and cannot
    change afterwards, so this cannot repair a wrong rate.  It exists so that a
    wrong rate is loud rather than silent: a recording made at the wrong
    nominal rate looks completely normal and misplaces every frequency in it.

    This runs while streaming, so pinning the rate explicitly still gets the
    check without paying a startup delay to measure first.
    """

    def __init__(
        self,
        label: str,
        declared_rate: float,
        window_seconds: float = 4.0,
        tolerance: float = 0.15,
    ) -> None:
        self.label = label
        self.declared_rate = float(declared_rate)
        self.window_seconds = float(window_seconds)
        self.tolerance = float(tolerance)
        self.samples = 0
        self.started: float | None = None
        self.done = self.declared_rate <= 0

    def observe(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.started is None:
            self.started = now
            return
        self.samples += 1
        elapsed = now - self.started
        if elapsed < self.window_seconds:
            return
        self.done = True
        measured = self.samples / elapsed
        if abs(measured - self.declared_rate) / self.declared_rate <= self.tolerance:
            return
        print(
            f"{self.label}: WARNING - reports are arriving at {measured:.1f} Hz but the "
            f"LSL stream declares {self.declared_rate:g} Hz. Every frequency in a "
            f"recording made now will be wrong by {measured / self.declared_rate:.2f}x. "
            "Restart with the correct --sample-rate, or let it be measured.",
            file=sys.stderr,
            flush=True,
        )


def make_packet_diagnostics_stream_info(
    name: str,
    nominal_srate: float,
    counter_modulus: int,
) -> StreamInfo:
    """Build the stable metadata for a packet diagnostics LSL stream."""

    info = StreamInfo(
        name,
        "EmotivPacketDiagnostics",
        len(PACKET_DIAGNOSTIC_CHANNELS),
        nominal_srate,
        "float32",
    )
    info.desc().append_child_value("counter_modulus", str(counter_modulus))
    info.desc().append_child_value(
        "description",
        "Packet sequence diagnostics; GAP_FLAG marks the sample after a HID counter discontinuity.",
    )
    channels = info.desc().append_child("channels")
    for label, unit, description in PACKET_DIAGNOSTIC_CHANNELS:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        channel.append_child_value("type", "PacketDiagnostic")
        channel.append_child_value("description", description)
    return info


def print_all_hid_interfaces() -> None:
    """Print every HID interface this machine reports, Emotiv or not.

    Device discovery can only match what the operating system enumerates, so
    this is the first thing to look at when a headset is not recognized.
    """

    devices = hid.enumerate()
    if not devices:
        print("No HID interfaces were enumerated at all.", flush=True)
        return
    print(f"{len(devices)} HID interface(s) enumerated:", flush=True)
    for index, device in enumerate(devices):
        print(
            f"  [{index}] vendor_id=0x{device.get('vendor_id', 0):04x} "
            f"product_id=0x{device.get('product_id', 0):04x} "
            f"manufacturer={device.get('manufacturer_string')!r} "
            f"product={device.get('product_string')!r} "
            f"serial={device.get('serial_number')!r} "
            f"usage_page={device.get('usage_page')} "
            f"usage={device.get('usage')} "
            f"interface_number={device.get('interface_number')}",
            flush=True,
        )


class EmotivBase():
    READ_SIZE = 32

    def get_hid_device(self):
        pass

    def get_stream_info(self) -> StreamInfo:
        pass

    def decode_data(self) -> list:
        pass

    def validate_data(self, data) -> bool:
        pass

    def main_loop(self):
        outlet = StreamOutlet(self.get_stream_info())

        device = self.get_hid_device()
        hid_device = hid.device()
        hid_device.open_path(device['path'])

        while True:
            data = hid_device.read(self.READ_SIZE)
            if self.validate_data(data):
                decoded = self.decode_data(data)
                outlet.push_sample(decoded)
