from __future__ import annotations

from dataclasses import dataclass, replace
import math
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
    (
        "RESET_FLAG",
        "boolean",
        "Counter continuity was lost: the counter restarted, or a dropout lasted over half a "
        "counter cycle, so MISSING_REPORTS is estimated from arrival time.",
    ),
    ("CUMULATIVE_GAPS", "count", "Counter discontinuity events since this reader started."),
    ("CUMULATIVE_RESETS", "count", "Counter continuity losses since this reader started."),
    (
        "FILLED",
        "boolean",
        "This sample was not received; the reader inserted it in place of a lost report.",
    ),
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
    # Published: the sample stands in for a lost report.
    filled: bool = False

    @property
    def flag_from_dropped_repeat(self) -> bool:
        """GAP_FLAG is set only because the previous report was a dropped repeat.

        A normal +1 step never sets GAP_FLAG on its own, so this case is exact.
        """
        return (
            self.gap
            and not self.repeat
            and not self.missing_reports
            and self.counter == self.expected_counter
        )

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
            float(self.filled),
        ]

    def filled_sample(self, counter: int) -> PacketDiagnostics:
        """Diagnostics for a sample inserted in place of a lost report.

        The report after the loss carries GAP_FLAG and MISSING_REPORTS; filled
        rows only mark themselves, with the counter the lost report would have had.
        """
        return replace(
            self,
            counter=counter,
            expected_counter=counter,
            gap=False,
            missing_reports=0,
            reset=False,
            repeat=False,
            filled=True,
        )


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

    def update(
        self,
        counter: int,
        *,
        reset_hint: bool = False,
        elapsed_periods: float | None = None,
    ) -> PacketDiagnostics:
        """Account for one received report.

        ``elapsed_periods`` is the arrival time since the previous report, in
        sample periods.  The counter alone cannot see whole cycles lost in a
        dropout, so when it is given, whole cycles are added from arrival time.
        A gap longer than half a cycle is also flagged as a continuity loss
        (RESET_FLAG), because arrival jitter makes that count an estimate.
        """
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
            long_dropout = elapsed_periods is not None and elapsed_periods > self.modulus / 2
            if reset:
                gap = False
                missing = 0
                self.cumulative_resets += 1
            elif long_dropout:
                wraps = max(0, round((elapsed_periods - delta) / self.modulus))
                steps = delta + wraps * self.modulus
                gap = True
                reset = True
                missing = max(0, steps - 1)
                self.cumulative_gaps += 1
                self.cumulative_missing += missing
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

    EPOC X (firmware 0x720 and 0x740) and Flex 1.0 dongles occasionally hand
    the host the previous report again.  Every repeat captured so far is
    identical in all 32 bytes, counter included.  Publishing it inserts a fake
    sample, and on Flex it applies the same channel deltas twice.

    The whole report must match, not just its EEG payload.  Consecutive real
    reports always differ because the counter advances.  Payloads alone can
    match: ``data/flex2.pcap`` has two real consecutive Flex reports (counters
    17 and 18) whose channels all sit at the maximum slew.
    """

    def __init__(self) -> None:
        self._previous: bytes | None = None

    def is_repeat(self, report) -> bool:
        report = bytes(report)
        repeat = report == self._previous
        self._previous = report
        return repeat


class CounterClock:
    """Timestamp samples by their place in the headset's sample sequence.

    A dongle can deliver reports late by a varying number of whole sample
    periods.  A Flex 1.0 dongle resends a report when the next one is late,
    and every later report then arrives one period behind until the dongle
    discards a block of 8 samples to catch up.  Arrival time is therefore off
    by 0-70 ms in steps of one period.  The counter is not: the headset
    samples at a steady rate.

    The clock turns the unwrapped counter index into a timestamp:

    * Arrivals fall on the same phase within the sample period, whatever the
      backlog, so a phase-locked loop on each arrival's error, wrapped into one
      period, tracks the headset's period and phase on the PC clock.
    * The lowest whole-period backlog level seen is taken as zero backlog; a
      lower one moves the clock back by whole periods.  Until the dongle's
      first catch-up after start, timestamps can be late by the backlog the
      session started with.
    * After a long dropout the dongle first dumps queued, old reports, so the
      first arrivals do not show the delay floor.  Stamps are held for
      ``RESTART_HOLD_SECONDS``, then the phase and floor are taken from the
      held arrivals and applied backwards.
    * Stamps always increase.  Filled samples are spaced evenly between the
      stamps of the received samples on either side.

    Offline replay of two bench recordings put these stamps within +/-0.1 ms
    of a whole-recording fit once the delay floor was known.
    """

    LOOP_TIME_CONSTANT_SECONDS = 10.0
    PHASE_CLIP_PERIODS = 0.25
    SLEW_PERIODS = 0.9
    RESTART_HOLD_SECONDS = 1.0

    def __init__(self, nominal_rate: float) -> None:
        self.period = 1.0 / float(nominal_rate)
        loop_samples = self.LOOP_TIME_CONSTANT_SECONDS * float(nominal_rate)
        self._kp = 2.0 / loop_samples
        self._ki = self._kp ** 2 / 4.0
        self._index: int | None = None
        self._phase = 0.0            # zero-backlog arrival time of index 0
        self._last_index: int | None = None
        self._last_stamp: float | None = None
        self._held: list[tuple[int, float, int]] = []
        self._hold_until: float | None = None
        self._ready: list[float] = []

    @property
    def holding(self) -> bool:
        return self._hold_until is not None

    def add(self, arrival: float, missing: int = 0, *, restart: bool = False, fill: bool = True) -> None:
        """Register a received report that follows ``missing`` lost ones.

        ``fill`` says whether a stamp is wanted for each lost report as well.
        """
        arrival = float(arrival)
        first = self._index is None
        self._index = 0 if first else self._index + int(missing) + 1
        slots = 1 if first or not fill else int(missing) + 1
        if self.holding and (restart or arrival >= self._hold_until):
            self._release_hold()
        if first:
            self._phase = arrival
            self._emit(self._index, arrival, slots)
        elif restart:
            self._held = [(self._index, arrival, slots)]
            self._hold_until = arrival + self.RESTART_HOLD_SECONDS
        elif self.holding:
            self._held.append((self._index, arrival, slots))
        else:
            self._track(self._index, arrival)
            self._emit(self._index, self._clamped(self._phase + self._index * self.period), slots)

    def take_ready(self) -> list[float]:
        """Stamps finalized since the last call, one per slot, in order."""
        ready, self._ready = self._ready, []
        return ready

    def flush(self) -> None:
        """Finalize held stamps, e.g. when the stream stops during a hold."""
        if self.holding:
            self._release_hold()

    def _track(self, index: int, arrival: float) -> None:
        error = arrival - (self._phase + index * self.period)
        level = math.floor(error / self.period + 0.5)
        if level < 0:
            self._phase += level * self.period
        clip = self.PHASE_CLIP_PERIODS * self.period
        wrapped = min(max(error - level * self.period, -clip), clip)
        self._phase += self._kp * wrapped
        self.period += self._ki * wrapped

    def _clamped(self, target: float) -> float:
        steps = self._index - self._last_index
        base = self._last_stamp + steps * self.period
        slack = self.SLEW_PERIODS * self.period
        return min(max(target, base - slack), base + slack)

    def _release_hold(self) -> None:
        held, self._held, self._hold_until = self._held, [], None
        # Phase from the circular mean of the held arrivals, floor from their
        # lowest whole-period level.  Dumped old reports sit on higher levels.
        residuals = [arrival - index * self.period for index, arrival, _ in held]
        angle = sum(complex(math.cos(2 * math.pi * r / self.period), math.sin(2 * math.pi * r / self.period))
                    for r in residuals)
        phase = math.atan2(angle.imag, angle.real) / (2 * math.pi) * self.period
        lowest = min(math.floor((r - phase) / self.period + 0.5) for r in residuals)
        self._phase = phase + lowest * self.period
        for index, _, slots in held:
            stamp = self._phase + index * self.period
            minimum = self._last_stamp + 0.1 * self.period * slots
            self._emit(index, max(stamp, minimum), slots)

    def _emit(self, index: int, stamp: float, slots: int) -> None:
        if self._last_stamp is not None and slots > 1:
            step = (stamp - self._last_stamp) / slots
            self._ready.extend(self._last_stamp + step * i for i in range(1, slots))
        self._ready.append(stamp)
        self._last_index = index
        self._last_stamp = stamp


def declare_can_drop_samples(info: StreamInfo) -> None:
    """Tell XDF readers the timestamps already account for lost samples.

    pyxdf otherwise replaces a stream's timestamps with a straight-line fit
    over sample number, which would undo the reader's own timestamps.
    """

    info.desc().append_child("synchronization").append_child_value("can_drop_samples", "true")


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
        if diagnostics.reset and diagnostics.missing_reports:
            self._emit(
                f"dropout: about {diagnostics.missing_reports} report(s) missing before counter "
                f"{diagnostics.counter} (estimated from arrival time); "
                f"{diagnostics.cumulative_missing} missing report(s) so far"
            )
            return
        if diagnostics.reset:
            self._emit(
                f"packet counter restarted at {diagnostics.counter} "
                f"(expected {diagnostics.expected_counter}); "
                f"{diagnostics.cumulative_resets} restart(s) so far"
            )
            return
        if not diagnostics.gap or diagnostics.flag_from_dropped_repeat:
            # A carried flag was already reported when the repeat was dropped.
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
    can_drop_samples: bool = False,
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
    if can_drop_samples:
        declare_can_drop_samples(info)
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
