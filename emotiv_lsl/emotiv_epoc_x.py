import csv
import hashlib
from pathlib import Path
import sys
import time

import hid
from Crypto.Cipher import AES
from pylsl import StreamInfo, StreamOutlet, local_clock

from emotiv_lsl.emotiv_base import (
    EmotivBase,
    PacketCounterTracker,
    PacketDiagnostics,
    PacketLossReporter,
    RepeatedReportFilter,
    SampleRateMonitor,
    declare_can_drop_samples,
    make_packet_diagnostics_stream_info,
)
from emotiv_lsl.emotiv_epoc_x_legacy import EmotivEpocXLegacy
from config import SRATE


class EmotivEpocX(EmotivBase):
    READ_SIZE = 32
    FEATURE_REPORT_LENGTH = 32
    LEGACY_PACKET_XOR = 0x55
    FW740_PACKET_XOR = 0x14
    FW740_MIN_VERSION = 0x73F
    FW740_FEATURE_MARKER = b'\x06\xff'
    LOG_FLUSH_INTERVAL = 256
    PROBE_READ_ATTEMPTS = 8
    PROBE_TIMEOUT_MS = 250
    # The wire counter runs over one second of samples: 0..127 at 128 Hz and
    # 0..255 at 256 Hz. Keep the fallback value here until the measured or
    # configured rate is available; the startup path selects the correct
    # modulus before any buffered samples are published.
    PACKET_COUNTER_MODULUS = 1 << 8
    LEGACY_PACKET_COUNTER_MODULUS = 1 << 7
    FW740_PACKET_COUNTER_MODULUS = 1 << 8
    PACKET_DIAGNOSTICS_NAME = "Epoc X Packet Diagnostics"
    EMOTIV_VENDOR_ID = 0x1234
    EMOTIV_PRODUCT_IDS = (0xED02,)
    EEG_USAGE = 2
    EEG_PRODUCT_MARKER = 'eeg'
    # Byte pairs holding the 14 EEG channels; 0, 1, 16 and 17 are metadata.
    EEG_BYTE_RANGES = ((2, 16), (18, 32))
    # convertEPOC_PLUS maps this pair to the ADC midpoint, which is what an
    # idle/baseline report contains on every channel.
    BASELINE_SAMPLE_PAIR = (0x00, 0x80)
    FIRMWARE_MODES = ('auto', 'legacy', '0740')
    # Exact messages the frozen legacy reader raises when the dongle is absent
    # or the headset is not streaming yet; see wait_for_legacy_reader.
    LEGACY_NOT_READY_ERRORS = (
        'Emotiv Epoc X not found',
        'No Emotiv HID interface produced EEG-sized packets during probing.',
    )
    LEGACY_RETRY_SECONDS = 2.0
    STREAM_WAIT_SECONDS = 2.0
    VERIFY_PACKET_COUNT = 12
    # EPOC X streams at either 128 or 256 Hz depending on how the headset is
    # configured, and nothing in the HID report says which.  A consumer that
    # trusts a wrong nominal rate misplaces every frequency by a factor of two,
    # so the rate is measured at startup instead of assumed.
    SUPPORTED_SAMPLE_RATES = (128.0, 256.0)
    SAMPLE_RATE_PROBE_SECONDS = 2.0
    SAMPLE_RATE_MIN_PACKETS = 32
    SAMPLE_RATE_TOLERANCE = 0.15
    VERIFY_MIN_STEPS = 4
    VERIFY_TIMEOUT_MS = 500

    CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
                'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    def __init__(
        self,
        emit_debug: bool = False,
        packet_log_path: str | None = None,
        firmware_mode: str = 'auto',
        sample_rate: float | None = None,
    ) -> None:
        self.delimiter = ','
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError('sample_rate must be positive')
        # None means "measure it at startup"; SRATE is only the fallback.
        self.requested_sample_rate = float(sample_rate) if sample_rate else None
        self.sample_rate = self.requested_sample_rate or float(SRATE)
        self.emit_debug = emit_debug
        self.packet_log_path = Path(packet_log_path) if packet_log_path else None
        if firmware_mode not in self.FIRMWARE_MODES:
            raise ValueError(
                f"firmware_mode must be one of {', '.join(self.FIRMWARE_MODES)}"
            )
        self.firmware_mode = firmware_mode

        # The legacy serial-derived key is not valid on firmware 0x740, and the
        # firmware-0x740 key is not valid on older headsets.  Both are built
        # after the streaming HID collection is open and the correct one is
        # confirmed against the decrypted packet counter, so one build supports
        # both generations.
        self.packet_xor = self.LEGACY_PACKET_XOR
        self.cipher = None
        self.firmware_version = None
        self.decryption_path = None
        self.PACKET_COUNTER_MODULUS = self.counter_modulus_for_rate(self.sample_rate)
        self._packet_tracker = PacketCounterTracker(self.PACKET_COUNTER_MODULUS)
        self._repeats = RepeatedReportFilter()
        self._last_packet_diagnostics: PacketDiagnostics | None = None
        self.packet_gaps = 0
        self.missing_reports = 0
        self.packet_resets = 0

    @classmethod
    def counter_modulus_for_rate(cls, rate: float) -> int:
        """Return the one-second packet-counter modulus for a nominal rate."""

        nearest = min(cls.SUPPORTED_SAMPLE_RATES, key=lambda value: abs(value - rate))
        return int(nearest)

    def set_counter_format(self, path: str) -> None:
        """Select the counter format after the HID cipher is confirmed."""

        # Both firmware generations use a one-second counter cycle. The
        # configured/fallback rate is used until startup measurement completes.
        self.PACKET_COUNTER_MODULUS = self.counter_modulus_for_rate(self.sample_rate)
        self._packet_tracker = PacketCounterTracker(self.PACKET_COUNTER_MODULUS)
        self._last_packet_diagnostics = None
        self.packet_gaps = 0
        self.missing_reports = 0
        self.packet_resets = 0

    def set_counter_rate(self) -> None:
        """Apply the measured/configured rate to packet diagnostics."""

        self.PACKET_COUNTER_MODULUS = self.counter_modulus_for_rate(self.sample_rate)
        self._packet_tracker = PacketCounterTracker(self.PACKET_COUNTER_MODULUS)
        self._last_packet_diagnostics = None
        self.packet_gaps = 0
        self.missing_reports = 0
        self.packet_resets = 0

    def is_eeg_packet(self, data: bytearray) -> bool:
        """Return whether a decrypted report contains an EEG sample.

        On the legacy protocol, the high counter bit identifies the battery /
        status half of the report stream.  Those reports are valid HID input,
        but they are not EEG samples and must not affect the LSL rate or packet
        loss accounting. Firmware 0x740 uses the full one-second counter range
        at the selected rate (0..127 at 128 Hz, 0..255 at 256 Hz).
        """

        if (
            self.decryption_path == 'legacy'
            and self.PACKET_COUNTER_MODULUS == self.LEGACY_PACKET_COUNTER_MODULUS
        ):
            return int(data[0]) < self.LEGACY_PACKET_COUNTER_MODULUS
        return True

    @classmethod
    def is_emotiv_device(cls, device: dict) -> bool:
        """Recognize an Emotiv HID collection across receiver generations.

        Windows does not always report a manufacturer string for these
        collections, so the vendor id is accepted as well; a device that names
        some *other* manufacturer is still rejected, and every candidate is
        probed before it is used.
        """
        if (
            device.get('vendor_id') == cls.EMOTIV_VENDOR_ID
            and device.get('product_id') in cls.EMOTIV_PRODUCT_IDS
        ):
            return True
        manufacturer = str(device.get('manufacturer_string') or '').strip().casefold()
        if 'emotiv' in manufacturer:
            return True
        if manufacturer:
            return False
        return device.get('vendor_id') == cls.EMOTIV_VENDOR_ID

    def get_hid_devices(self) -> list[dict]:
        return [device for device in hid.enumerate() if self.is_emotiv_device(device)]

    @classmethod
    def probe_priority(cls, device: dict) -> tuple[int, int, int]:
        """Sort key placing the most likely EEG collection first."""
        product = str(device.get('product_string') or '').casefold()
        interface = device.get('interface_number')
        return (
            0 if cls.EEG_PRODUCT_MARKER in product else 1,
            0 if device.get('usage') == cls.EEG_USAGE else 1,
            interface if isinstance(interface, int) and interface >= 0 else 99,
        )

    @staticmethod
    def no_device_message() -> str:
        return (
            'No Emotiv HID interface was found. Connect the receiver, power on '
            'the headset, and wait for both indicator lights. Run '
            '"python main.py --list-hid" to print every HID interface this '
            'machine reports.'
        )

    def describe_hid_device(self, device: dict) -> str:
        return (
            f"path={device.get('path')} "
            f"product={device.get('product_string')} "
            f"serial={device.get('serial_number')} "
            f"usage_page={device.get('usage_page')} "
            f"usage={device.get('usage')} "
            f"interface_number={device.get('interface_number')}"
        )

    def print_hid_devices(self, devices: list[dict]) -> None:
        for index, device in enumerate(devices):
            print(
                "INFO| hidif "
                f"'{index}' ({self.describe_hid_device(device)})",
                file=sys.stderr,
                flush=True,
            )

    def get_hid_device(self):
        devices = self.get_hid_devices()
        if not devices:
            raise RuntimeError(self.no_device_message())
        return sorted(devices, key=self.probe_priority)[0]

    def get_crypto_key(self, device: dict | None = None) -> bytearray:
        if device is None:
            device = self.get_hid_device()
        serial = device.get('serial_number') or ''
        if len(serial) < 4:
            raise RuntimeError(
                'the HID interface did not report a serial number long enough '
                'to build the legacy EPOC X key'
            )
        sn = bytearray()
        for i in range(0, len(serial)):
            sn += bytearray([ord(serial[i])])

        return bytearray([sn[-1], sn[-2], sn[-4], sn[-4], sn[-2], sn[-1], sn[-2], sn[-4], sn[-1], sn[-4], sn[-3], sn[-2], sn[-1], sn[-2], sn[-2], sn[-3]])

    @classmethod
    def parse_firmware_feature_report(cls, report) -> tuple[int, bytes] | None:
        """Extract the firmware version and session seed from a feature report.

        The firmware-0x740 report observed on the EPOC X collection is:

            20 30 06 ff 07 40 e5 02 01 ce ...

        The marker is followed by a big-endian firmware number and four seed
        bytes.  Scanning for the marker tolerates HID backends that prepend a
        report-ID byte to the returned buffer.
        """
        data = bytes(report)
        marker = cls.FW740_FEATURE_MARKER
        for offset in range(0, len(data) - 7):
            if data[offset:offset + len(marker)] != marker:
                continue
            firmware = int.from_bytes(data[offset + 2:offset + 4], 'big')
            seed = data[offset + 4:offset + 8]
            if len(seed) == 4:
                return firmware, seed
        return None

    @staticmethod
    def get_firmware_crypto_key(seed: bytes, firmware: int) -> bytes:
        seed_hex = bytes(seed).hex().upper()
        key_material = (
            b'Vohcha7e' + seed_hex[:4].encode('ascii') +
            b'Ut3phaej' + seed_hex[4:8].encode('ascii') +
            f'{firmware:04X}'.encode('ascii')
        )
        return hashlib.sha256(key_material).digest()

    def read_firmware_feature_report(self, hid_device) -> tuple[int, bytes] | None:
        """Query the EEG collection's feature report for firmware and seed.

        Older firmware does not answer this request, or answers without the
        marker.  Neither is an error: the caller still has the legacy key.
        """
        try:
            report = hid_device.get_feature_report(0, self.FEATURE_REPORT_LENGTH)
        except Exception as exc:
            print(
                f'Could not query the EPOC X feature report ({exc}); '
                'assuming pre-0x740 firmware.',
                file=sys.stderr,
                flush=True,
            )
            return None

        parsed = self.parse_firmware_feature_report(report or [])
        if parsed is None:
            print(
                'The EPOC X feature report carried no firmware marker; '
                'assuming pre-0x740 firmware.',
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f'The EPOC X feature report says firmware 0x{parsed[0]:03x}.',
                file=sys.stderr,
                flush=True,
            )
        return parsed

    def candidate_ciphers(self, hid_device, device: dict) -> list[tuple[str, int, object]]:
        """Build the decryption candidates for this headset, best guess first.

        Both EPOC X generations are served from one code path.  The feature
        report chooses the preferred candidate; the other one stays in the list
        so a headset whose feature report is missing, silent, or unexpected
        still streams.  ``firmware_mode`` pins the choice when a headset needs
        to be forced down one path.
        """
        parsed = None
        if self.firmware_mode != 'legacy':
            parsed = self.read_firmware_feature_report(hid_device)

        firmware_candidate = None
        if parsed is not None:
            firmware, seed = parsed
            self.firmware_version = firmware
            firmware_candidate = (
                f'firmware-0x{firmware:03x}',
                self.FW740_PACKET_XOR,
                AES.new(self.get_firmware_crypto_key(seed, firmware), AES.MODE_ECB),
            )

        legacy_candidate = None
        if self.firmware_mode != '0740':
            try:
                legacy_candidate = (
                    'legacy',
                    self.LEGACY_PACKET_XOR,
                    AES.new(self.get_crypto_key(device), AES.MODE_ECB),
                )
            except Exception as exc:
                print(
                    f'The legacy EPOC X key is unavailable ({exc}).',
                    file=sys.stderr,
                    flush=True,
                )

        if self.firmware_mode == 'legacy':
            candidates = [legacy_candidate]
        elif self.firmware_mode == '0740':
            candidates = [firmware_candidate]
        elif parsed is not None and parsed[0] >= self.FW740_MIN_VERSION:
            candidates = [firmware_candidate, legacy_candidate]
        else:
            candidates = [legacy_candidate, firmware_candidate]

        if self.firmware_version is None:
            self.firmware_version = 0
        return [candidate for candidate in candidates if candidate is not None]

    def collect_verification_packets(self, hid_device) -> list[list[int]]:
        """Buffer a few encrypted reports so a candidate key can be checked."""
        packets: list[list[int]] = []
        for _ in range(self.VERIFY_PACKET_COUNT * 2):
            if len(packets) >= self.VERIFY_PACKET_COUNT:
                break
            try:
                report = hid_device.read(self.READ_SIZE, timeout_ms=self.VERIFY_TIMEOUT_MS)
            except Exception as exc:
                print(
                    f'Stopped confirming the EPOC X key after a HID read error ({exc}).',
                    file=sys.stderr,
                    flush=True,
                )
                break
            if not report:
                # The headset is connected but not streaming yet; the preferred
                # candidate is used unverified rather than blocking here.
                break
            normalized = self.normalize_encrypted_packet(report)
            if normalized is not None:
                packets.append(list(normalized))
        return packets

    def measure_sample_rate(self, hid_device) -> list[list[int]]:
        """Time incoming reports and set ``self.sample_rate`` from the result.

        Returns the EEG packets consumed while measuring so the caller can
        publish them. Non-EEG HID reports, when present on the legacy path,
        are consumed but intentionally not returned.
        """

        if self.requested_sample_rate:
            self.set_counter_rate()
            print(
                f'Using configured EPOC X sample rate {self.sample_rate:g} Hz.',
                file=sys.stderr,
                flush=True,
            )
            return []

        packets: list[list[int]] = []
        started: float | None = None
        last = 0.0
        deadline = time.monotonic() + self.SAMPLE_RATE_PROBE_SECONDS
        while time.monotonic() < deadline:
            try:
                report = hid_device.read(self.READ_SIZE, timeout_ms=self.VERIFY_TIMEOUT_MS)
            except Exception as exc:
                print(
                    f'Stopped measuring the EPOC X sample rate after a HID read error ({exc}).',
                    file=sys.stderr,
                    flush=True,
                )
                break
            if not report:
                break
            normalized = self.normalize_encrypted_packet(report)
            if normalized is None:
                continue
            # HID input can contain non-EEG reports on the legacy protocol.
            # They are deliberately consumed here but are not samples to
            # publish or include in the rate estimate.
            decrypted = self.decrypt_data(normalized)
            if not self.is_eeg_packet(decrypted):
                continue
            last = time.monotonic()
            if started is None:
                started = last
            packets.append(list(normalized))

        elapsed = (last - started) if started is not None else 0.0
        if len(packets) < self.SAMPLE_RATE_MIN_PACKETS or elapsed <= 0:
            self.set_counter_rate()
            print(
                f'Could not measure the EPOC X sample rate ({len(packets)} report(s) in '
                f'{elapsed:.2f}s); assuming {self.sample_rate:g} Hz. '
                'Pass --sample-rate to set it explicitly.',
                file=sys.stderr,
                flush=True,
            )
            return packets

        measured = (len(packets) - 1) / elapsed
        nearest = min(self.SUPPORTED_SAMPLE_RATES, key=lambda rate: abs(rate - measured))
        if abs(measured - nearest) / nearest > self.SAMPLE_RATE_TOLERANCE:
            self.set_counter_rate()
            print(
                f'Measured EPOC X report rate {measured:.1f} Hz is not close to '
                f"{' or '.join(f'{rate:g}' for rate in self.SUPPORTED_SAMPLE_RATES)} Hz; "
                f'assuming {self.sample_rate:g} Hz. Pass --sample-rate to override.',
                file=sys.stderr,
                flush=True,
            )
            return packets

        self.sample_rate = nearest
        self.set_counter_rate()
        print(
            f'Measured EPOC X sample rate {measured:.1f} Hz -> declaring {nearest:g} Hz.',
            file=sys.stderr,
            flush=True,
        )
        return packets

    def count_counter_steps(
        self,
        packets,
        packet_xor: int,
        cipher,
        *,
        modulus: int | None = None,
        eeg_only: bool = False,
    ) -> int:
        """Count consecutive reports whose decrypted counter advances by one.

        With the right key the EPOC X counter increments every report; with the
        wrong key byte 0 is effectively random, so this separates the two
        candidates without needing to interpret the EEG payload.
        """
        modulus = modulus or self.PACKET_COUNTER_MODULUS
        counters = [
            cipher.decrypt(bytearray(el ^ packet_xor for el in packet))[0]
            for packet in packets
        ]
        if eeg_only:
            counters = [counter for counter in counters if counter < modulus]
        return sum(
            1
            for previous, current in zip(counters, counters[1:])
            if (current - previous) % modulus == 1
        )

    def configure_cipher(self, hid_device, device: dict) -> list[list[int]]:
        """Select and confirm the cipher for one HID session.

        Returns the reports consumed while confirming the choice so the caller
        can publish them instead of discarding them.
        """
        candidates = self.candidate_ciphers(hid_device, device)
        if not candidates:
            raise RuntimeError(
                'No EPOC X decryption key could be derived for this headset: '
                'the feature report carried no firmware seed and the legacy '
                'serial-derived key was unavailable.'
            )

        packets = self.collect_verification_packets(hid_device)
        pairs = max(0, len(packets) - 1)
        scores = []
        for name, packet_xor, cipher in candidates:
            modulus = self.counter_modulus_for_rate(self.sample_rate)
            scores.append(
                self.count_counter_steps(
                    packets,
                    packet_xor,
                    cipher,
                    modulus=modulus,
                    eeg_only=name == 'legacy',
                )
            )
        # max() keeps the first maximum, so a tie leaves the preferred
        # candidate in place.
        best_index = max(range(len(candidates)), key=scores.__getitem__)
        required = min(self.VERIFY_MIN_STEPS, pairs)

        if pairs >= 2 and scores[best_index] >= required:
            selected = best_index
            confirmation = f'confirmed on {scores[best_index]}/{pairs} packet steps'
        else:
            selected = 0
            if pairs >= 2:
                confirmation = (
                    'NOT confirmed - no candidate produced a sequential packet '
                    'counter, so the EEG values may be wrong'
                )
            else:
                confirmation = 'unconfirmed, the headset sent no packets to check against'

        name, packet_xor, cipher = candidates[selected]
        self.packet_xor = packet_xor
        self.cipher = cipher
        self.decryption_path = name
        self.set_counter_format(name)
        print(
            f'Using {name} EPOC X HID decryption ({confirmation}).',
            file=sys.stderr,
            flush=True,
        )
        return packets

    def get_stream_info(self) -> StreamInfo:
        n_channels = len(self.CH_NAMES)

        info = StreamInfo('Epoc X', 'EEG', n_channels, self.sample_rate, 'float32')
        chns = info.desc().append_child("channels")
        for label in self.CH_NAMES:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
            ch.append_child_value("scaling_factor", "1")

        cap = info.desc().append_child("cap")
        cap.append_child_value("name", "easycap-M1")
        cap.append_child_value("labelscheme", "10-20")
        info.desc().append_child_value("sample_rate_source",
                                       "configured" if self.requested_sample_rate else "measured")
        declare_can_drop_samples(info)

        return info

    def get_packet_diagnostics_stream_info(self) -> StreamInfo:
        return make_packet_diagnostics_stream_info(
            self.PACKET_DIAGNOSTICS_NAME,
            self.sample_rate,
            self.PACKET_COUNTER_MODULUS,
            can_drop_samples=True,
        )

    def get_packet_diagnostics_sample(self) -> list[float]:
        if self._last_packet_diagnostics is None:
            raise RuntimeError("packet diagnostics are unavailable before the first EEG packet")
        return self._last_packet_diagnostics.as_lsl_sample()

    def get_debug_stream_info(self) -> StreamInfo:
        info = StreamInfo('Epoc X Debug', 'EmotivDebug', 4, self.sample_rate, 'float32')
        chns = info.desc().append_child("channels")
        for label in ['COUNTER', 'DATA_MODE', 'BYTE16', 'BYTE17']:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "unknown")
            ch.append_child_value("type", "Debug")
        declare_can_drop_samples(info)
        return info

    def decrypt_data(self, data) -> bytearray:
        if self.cipher is None:
            raise RuntimeError('HID cipher has not been configured')
        data = [el ^ self.packet_xor for el in data]
        return self.cipher.decrypt(bytearray(data))

    def normalize_encrypted_packet(self, data) -> list[int] | None:
        if len(data) == self.READ_SIZE:
            return data
        if len(data) == self.READ_SIZE + 1 and data[0] == 0:
            return data[1:]
        return None

    def probe_report_lengths(self, hid_device) -> tuple[int | None, set[int]]:
        """Read a few reports; return the first EEG-sized length and all seen."""
        observed_lengths: set[int] = set()
        for _ in range(self.PROBE_READ_ATTEMPTS):
            packet = hid_device.read(self.READ_SIZE, timeout_ms=self.PROBE_TIMEOUT_MS)
            if not packet:
                continue
            observed_lengths.add(len(packet))
            if self.normalize_encrypted_packet(packet) is not None:
                return len(packet), observed_lengths
        return None, observed_lengths

    def open_streaming_hid_device(self):
        """Open the Emotiv interface that carries EEG reports, waiting until one does.

        Nothing about the session can be decided before the headset streams.
        The key is confirmed and the rate measured against real packets.  Until
        the current headset connects, the dongle's feature report can still
        describe the headset it was last paired with.  Continuing with a silent
        interface therefore locks in an unverified key and a guessed rate that
        are wrong once packets arrive.  A missing dongle or a silent headset is
        waited for instead, and the handle is returned open once reports flow.
        """
        verbose = True
        announced = None
        while True:
            devices = self.get_hid_devices()
            if devices:
                opened = self.probe_streaming_hid_device(devices, verbose=verbose)
                if opened is not None:
                    return opened
                verbose = False
                reason = 'no Emotiv HID interface is sending data yet.'
            else:
                reason = self.no_device_message()
            if reason != announced:
                print(
                    f'Waiting for the EPOC X headset to start streaming: {reason} '
                    'Switch the headset on and let EMOTIV Launcher connect it; '
                    'retrying until it sends data.',
                    file=sys.stderr,
                    flush=True,
                )
                announced = reason
            time.sleep(self.STREAM_WAIT_SECONDS)

    def probe_streaming_hid_device(self, devices: list[dict], verbose: bool = True):
        """Probe each interface once; return ``(device, open handle)`` or None.

        Interfaces are tried best-guess first, and one that cannot be opened or
        stays silent never stops the remaining ones from being tried: Windows
        claims some HID collections for itself and another application may hold
        one open.  Silent interfaces are closed.  ``verbose`` is off for repeat
        probes while waiting, so the console is not flooded.
        """
        if verbose:
            self.print_hid_devices(devices)
        candidates = sorted(devices, key=self.probe_priority)
        opened_any = False

        for device in candidates:
            hid_device = hid.device()
            try:
                hid_device.open_path(device['path'])
            except Exception as exc:
                if verbose:
                    print(
                        f"Could not open Emotiv HID interface ({self.describe_hid_device(device)}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            opened_any = True

            try:
                length, observed_lengths = self.probe_report_lengths(hid_device)
            except Exception as exc:
                if verbose:
                    print(
                        f"Emotiv HID interface ({self.describe_hid_device(device)}) failed while probing: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                hid_device.close()
                continue

            if length is not None:
                print(
                    f"Using Emotiv HID device with packet length {length} "
                    f"({self.describe_hid_device(device)}).",
                    file=sys.stderr,
                    flush=True,
                )
                return device, hid_device

            if verbose and observed_lengths:
                print(
                    f"Emotiv HID interface ({self.describe_hid_device(device)}) produced "
                    f"unsupported packet lengths: {sorted(observed_lengths)}",
                    file=sys.stderr,
                    flush=True,
                )
            elif verbose:
                print(
                    f"Emotiv HID interface ({self.describe_hid_device(device)}) produced "
                    "no data during probing.",
                    file=sys.stderr,
                    flush=True,
                )
            hid_device.close()

        if not opened_any:
            raise RuntimeError(
                'No Emotiv HID interface could be opened. Interfaces seen: '
                + '; '.join(self.describe_hid_device(device) for device in candidates)
            )
        return None

    def get_streaming_hid_device_info(self):
        device, hid_device = self.open_streaming_hid_device()
        hid_device.close()
        return device

    def _looks_like_counter_reset(self, data: bytearray) -> bool:
        """Recognize the baseline packet emitted when an EPOC X stream restarts."""

        previous = self._packet_tracker.last_counter
        counter = int(data[0])
        if previous is None or counter != 0:
            return False
        # A normal wrap is the last value in the selected counter format -> 0.
        # The captured EPOC X restart packet arrived from a substantially later
        # counter and parked every channel on the ADC midpoint, so require both
        # features before calling it a reset.
        if previous >= self.PACKET_COUNTER_MODULUS - 4:
            return False
        return self.is_baseline_packet(data)

    def is_baseline_packet(self, data: bytearray) -> bool:
        """True when every EEG channel sits exactly on the ADC midpoint.

        Each channel is a byte pair, so the captured baseline report is
        ``00 80`` repeated - not a run of ``0x80`` bytes.
        """
        for start, stop in self.EEG_BYTE_RANGES:
            for index in range(start, stop, 2):
                if (data[index], data[index + 1]) != self.BASELINE_SAMPLE_PAIR:
                    return False
        return True

    def _skip_repeated_packet(self, data: bytearray) -> PacketDiagnostics:
        """Count a dropped repeat; the next published sample carries its flag."""

        diagnostics = self._packet_tracker.skip_repeat(int(data[0]))
        self.packet_gaps = diagnostics.cumulative_gaps
        self.missing_reports = diagnostics.cumulative_missing
        self.packet_resets = diagnostics.cumulative_resets
        return diagnostics

    def _update_packet_diagnostics(
        self,
        data: bytearray,
        elapsed_periods: float | None = None,
    ) -> PacketDiagnostics:
        diagnostics = self._packet_tracker.update(
            int(data[0]),
            reset_hint=self._looks_like_counter_reset(data),
            elapsed_periods=elapsed_periods,
        )
        self._last_packet_diagnostics = diagnostics
        self.packet_gaps = diagnostics.cumulative_gaps
        self.missing_reports = diagnostics.cumulative_missing
        self.packet_resets = diagnostics.cumulative_resets
        return diagnostics

    def decode_data(self, data) -> list:
        data = self.decrypt_data(data)
        if not self.is_eeg_packet(data):
            raise ValueError('decrypted HID report does not contain an EEG sample')
        self._update_packet_diagnostics(data)
        return self.decode_eeg_sample(data)

    def decode_eeg_sample(self, data: bytearray) -> list:
        # Bytes 16 and 17 are intentionally excluded here because this repo only
        # publishes the 14 EEG channels. Use the debug outlet/logger to inspect
        # them without breaking downstream EEG consumers.
        packet_data = ""
        for i in range(2, 16, 2):
            packet_data = packet_data + \
                str(self.convertEPOC_PLUS(
                    str(data[i]), str(data[i+1]))) + self.delimiter

        for i in range(18, len(data), 2):
            packet_data = packet_data + \
                str(self.convertEPOC_PLUS(
                    str(data[i]), str(data[i+1]))) + self.delimiter

        packet_data = packet_data[:-len(self.delimiter)]
        packet_data = packet_data.split(self.delimiter)
        packet_data = [float(i) for i in packet_data]

        # swap positions of AF3 and F3
        packet_data[0], packet_data[2] = packet_data[2], packet_data[0]

        # swap positions of AF4 and F4
        packet_data[13], packet_data[11] = packet_data[11], packet_data[13]

        # swap positions of F7 and FC5
        packet_data[1], packet_data[3] = packet_data[3], packet_data[1]

        # swap positions of FC6 and F8
        packet_data[10], packet_data[12] = packet_data[12], packet_data[10]

        return packet_data

    def decode_debug_sample(self, data: bytearray) -> list:
        return [float(data[0]), float(data[1]), float(data[16]), float(data[17])]

    def log_decrypted_packet(self, writer: csv.writer, data: bytearray) -> None:
        writer.writerow(list(data))

    def wait_for_legacy_reader(self) -> EmotivEpocXLegacy:
        """Build the frozen pre-0x740 reader once the headset is streaming.

        That reader raises if no dongle is enumerated, or if no interface sends
        a report during its short probe, which is the normal state before the
        headset is switched on or paired.  Those two failures are retried here
        rather than crashing the EEG thread.  Anything else still propagates.
        The reader's own outlets are only created after this returns, so an
        LSL recorder never sees a stream appear and vanish while waiting.
        """
        announced = None
        while True:
            try:
                reader = EmotivEpocXLegacy(
                    emit_debug=self.emit_debug,
                    packet_log_path=self.packet_log_path,
                    sample_rate=self.requested_sample_rate,
                )
                reader.get_streaming_hid_device_info()
                return reader
            except Exception as exc:
                if str(exc) not in self.LEGACY_NOT_READY_ERRORS:
                    raise
                if announced != str(exc):
                    print(
                        f'Waiting for the EPOC X headset to start streaming ({exc}). '
                        'Switch the headset on; retrying until it sends data.',
                        file=sys.stderr,
                        flush=True,
                    )
                    announced = str(exc)
            time.sleep(self.LEGACY_RETRY_SECONDS)

    def main_loop(self):
        if self.firmware_mode == 'legacy':
            # Hand the whole session to the frozen pre-0x740 reader. None of the
            # discovery, feature-report, verification or rate-measurement steps
            # below run for legacy headsets.
            return self.wait_for_legacy_reader().main_loop()

        log_handle = None
        log_writer = None
        logged_packets = 0
        warned_lengths = set()
        if self.packet_log_path:
            self.packet_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.packet_log_path.open('w', newline='')
            log_writer = csv.writer(log_handle)
            log_writer.writerow([f'byte_{i}' for i in range(self.READ_SIZE)])
            log_handle.flush()
            print(f"Logging decrypted packets to {self.packet_log_path.resolve()}", file=sys.stderr, flush=True)

        device, hid_device = self.open_streaming_hid_device()
        buffered = self.configure_cipher(hid_device, device)
        # The outlets declare the sample rate, so it has to be known before they
        # are created.  Everything read while measuring is published below.
        buffered += self.measure_sample_rate(hid_device)
        eeg_outlet = StreamOutlet(self.get_stream_info())
        diagnostics_outlet = StreamOutlet(self.get_packet_diagnostics_stream_info())
        debug_outlet = StreamOutlet(self.get_debug_stream_info()) if self.emit_debug else None
        loss_reporter = PacketLossReporter('Epoc X')
        rate_monitor = SampleRateMonitor('Epoc X', self.sample_rate)
        print(f"Streaming from {self.describe_hid_device(device)}", file=sys.stderr, flush=True)
        last_live_timestamp: list[float | None] = [None]

        def publish(normalized, *, monitor_rate: bool = True) -> None:
            nonlocal logged_packets
            decrypted = self.decrypt_data(normalized)
            if not self.is_eeg_packet(decrypted):
                return
            # The packet log keeps every EEG report, repeats included, as the
            # raw record of what the headset sent.
            if log_writer:
                self.log_decrypted_packet(log_writer, decrypted)
                logged_packets += 1
                if logged_packets == 1:
                    print("Received first valid decrypted packet.", file=sys.stderr, flush=True)
                if logged_packets % self.LOG_FLUSH_INTERVAL == 0:
                    log_handle.flush()
            # A byte-identical repeat of the previous report is an extra copy,
            # not a new sample: count it, but do not publish it.
            if self._repeats.is_repeat(decrypted):
                loss_reporter.report(self._skip_repeated_packet(decrypted))
                return
            # Packets consumed during startup are historical samples.  They
            # are still published, but must not seed the live-arrival rate
            # monitor or its stopwatch; doing so counts the startup buffer as
            # if it arrived instantaneously and produces false rates such as
            # 195 Hz for a 128 Hz headset.
            if monitor_rate:
                rate_monitor.observe()
            timestamp = local_clock()
            # Arrival spacing counts whole counter cycles lost in a dropout.
            # Startup packets were read earlier, so they have no arrival time.
            # The counter runs over one second, so its modulus is samples per second.
            elapsed_periods = None
            if monitor_rate and last_live_timestamp[0] is not None:
                elapsed_periods = (timestamp - last_live_timestamp[0]) * self.PACKET_COUNTER_MODULUS
            if monitor_rate:
                last_live_timestamp[0] = timestamp
            diagnostics = self._update_packet_diagnostics(decrypted, elapsed_periods)
            loss_reporter.report(diagnostics)
            eeg_outlet.push_sample(
                self.decode_eeg_sample(decrypted),
                timestamp=timestamp,
            )
            diagnostics_outlet.push_sample(
                diagnostics.as_lsl_sample(),
                timestamp=timestamp,
            )

            if debug_outlet:
                debug_outlet.push_sample(
                    self.decode_debug_sample(decrypted),
                    timestamp=timestamp,
                )

        try:
            # Reports consumed while confirming the key are still real samples.
            for normalized in buffered:
                publish(normalized, monitor_rate=False)

            while True:
                encrypted = hid_device.read(self.READ_SIZE, timeout_ms=1000)
                if not encrypted:
                    # A read timeout, not a malformed report: the headset is
                    # connected but idle.
                    continue
                normalized = self.normalize_encrypted_packet(encrypted)
                if normalized is None:
                    packet_len = len(encrypted)
                    if packet_len not in warned_lengths:
                        print(
                            f"Skipping HID packet with length {packet_len}; expected 32 bytes"
                            " or 33 with a leading report ID byte.",
                            file=sys.stderr,
                            flush=True,
                        )
                        warned_lengths.add(packet_len)
                    continue

                publish(normalized)
        finally:
            hid_device.close()
            if log_handle:
                log_handle.flush()
                log_handle.close()

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129) +
                              4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def validate_data(self, data) -> bool:
        return len(data) == 32
