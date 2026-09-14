"""Frozen pre-firmware-0x740 EPOC X reader, used by ``--firmware legacy``.

This is the reader from commit 9944584, the last one before firmware 0x740
support, and it is known to stream from 0x720 headsets.  Do not route it
through the shared discovery, feature-report, key-verification or
rate-measurement code in ``emotiv_epoc_x.py``.  Keep it separate so a change
made for newer firmware cannot break older headsets again.

Device discovery, HID reads and decryption are unchanged from 9944584.  The
additions only work on packets after they are decrypted:

* The outlets declare ``sample_rate`` when one is given, and ``config.SRATE``
  otherwise, as before.
* Every EEG sample is mirrored to the same ``Epoc X Packet Diagnostics`` stream
  as the firmware-0x740 path, and packet-counter gaps are summarized on stderr.
  The EEG and diagnostics samples share one timestamp.
"""

import csv
from pathlib import Path
import sys

import hid
from Crypto.Cipher import AES
from pylsl import StreamInfo, StreamOutlet, local_clock

from emotiv_lsl.emotiv_base import (
    EmotivBase,
    PacketCounterTracker,
    PacketDiagnostics,
    PacketLossReporter,
    make_packet_diagnostics_stream_info,
)
from config import SRATE


class EmotivEpocXLegacy(EmotivBase):
    READ_SIZE = 32
    LOG_FLUSH_INTERVAL = 256
    PROBE_READ_ATTEMPTS = 8
    PROBE_TIMEOUT_MS = 250
    PACKET_DIAGNOSTICS_NAME = "Epoc X Packet Diagnostics"
    # Byte 0 counts once per second of samples.  A 0x720 capture at 128 Hz
    # (data/legacy_packets.csv, 10,032 reports) confirms that it counts 0..127,
    # and that every report is EEG.  256 Hz is assumed to count 0..255, as on
    # firmware 0x740, but has not been captured on this firmware.
    COUNTER_MODULUS_128HZ = 128
    COUNTER_MODULUS_256HZ = 256

    CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
                'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    def __init__(
        self,
        emit_debug: bool = False,
        packet_log_path: str | None = None,
        sample_rate: float | None = None,
    ) -> None:
        self.delimiter = ','
        self.emit_debug = emit_debug
        self.packet_log_path = Path(packet_log_path) if packet_log_path else None
        self.sample_rate = float(sample_rate) if sample_rate else float(SRATE)
        self.packet_tracker = PacketCounterTracker(
            self.COUNTER_MODULUS_256HZ if self.sample_rate > 128 else self.COUNTER_MODULUS_128HZ
        )

        self.cipher = AES.new(self.get_crypto_key(), AES.MODE_ECB)

    def get_hid_devices(self) -> list[dict]:
        devices = []
        for device in hid.enumerate():
            if device.get('manufacturer_string') == 'Emotiv':
                devices.append(device)
        return devices

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
        for device in devices:
            if device.get('usage') == 2:
                return device
        if devices:
            return devices[0]

        raise Exception('Emotiv Epoc X not found')

    def get_crypto_key(self) -> bytearray:
        serial = self.get_hid_device()['serial_number']
        sn = bytearray()
        for i in range(0, len(serial)):
            sn += bytearray([ord(serial[i])])

        return bytearray([sn[-1], sn[-2], sn[-4], sn[-4], sn[-2], sn[-1], sn[-2], sn[-4], sn[-1], sn[-4], sn[-3], sn[-2], sn[-1], sn[-2], sn[-2], sn[-3]])

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

        return info

    def get_debug_stream_info(self) -> StreamInfo:
        info = StreamInfo('Epoc X Debug', 'EmotivDebug', 4, self.sample_rate, 'float32')
        chns = info.desc().append_child("channels")
        for label in ['COUNTER', 'DATA_MODE', 'BYTE16', 'BYTE17']:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "unknown")
            ch.append_child_value("type", "Debug")
        return info

    def decrypt_data(self, data) -> bytearray:
        data = [el ^ 0x55 for el in data]
        return self.cipher.decrypt(bytearray(data))

    def normalize_encrypted_packet(self, data) -> list[int] | None:
        if len(data) == self.READ_SIZE:
            return data
        if len(data) == self.READ_SIZE + 1 and data[0] == 0:
            return data[1:]
        return None

    def get_streaming_hid_device_info(self):
        devices = self.get_hid_devices()
        if not devices:
            raise Exception('Emotiv Epoc X not found')

        self.print_hid_devices(devices)

        for index, device in enumerate(devices):
            hid_device = hid.device()
            hid_device.open_path(device['path'])
            observed_lengths = set()

            try:
                for _ in range(self.PROBE_READ_ATTEMPTS):
                    packet = hid_device.read(self.READ_SIZE, timeout_ms=self.PROBE_TIMEOUT_MS)
                    if not packet:
                        continue

                    observed_lengths.add(len(packet))
                    normalized = self.normalize_encrypted_packet(packet)
                    if normalized is not None:
                        print(
                            f"Using Emotiv HID device [{index}] with packet length {len(packet)}.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return device

                if observed_lengths:
                    print(
                        f"Emotiv HID device [{index}] produced unsupported packet lengths:"
                        f" {sorted(observed_lengths)}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"Emotiv HID device [{index}] produced no data during probing.",
                        file=sys.stderr,
                        flush=True,
                    )
            finally:
                hid_device.close()

        raise RuntimeError('No Emotiv HID interface produced EEG-sized packets during probing.')

    def decode_data(self, data) -> list:
        data = self.decrypt_data(data)
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

    def track_packet(self, counter: int) -> PacketDiagnostics:
        """Update packet-loss accounting; never let it interrupt the EEG stream."""
        if counter >= self.packet_tracker.modulus:
            # A counter past 127 means the headset is at 256 Hz but the stream
            # declares a lower rate, which rescales every frequency downstream.
            print(
                f"Epoc X: WARNING - packet counter reached {counter}, so the headset is "
                f"streaming at 256 Hz but the LSL stream declares {self.sample_rate:g} Hz. "
                "Restart with --sample-rate 256. Packet diagnostics now use a 0..255 counter.",
                file=sys.stderr,
                flush=True,
            )
            self.packet_tracker = PacketCounterTracker(self.COUNTER_MODULUS_256HZ)
        return self.packet_tracker.update(counter)

    def main_loop(self):
        eeg_outlet = StreamOutlet(self.get_stream_info())
        diagnostics_outlet = StreamOutlet(make_packet_diagnostics_stream_info(
            self.PACKET_DIAGNOSTICS_NAME,
            self.sample_rate,
            self.packet_tracker.modulus,
        ))
        loss_reporter = PacketLossReporter('Epoc X')
        debug_outlet = StreamOutlet(self.get_debug_stream_info()) if self.emit_debug else None

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

        device = self.get_streaming_hid_device_info()
        hid_device = hid.device()
        hid_device.open_path(device['path'])
        print(f"Streaming from {self.describe_hid_device(device)}", file=sys.stderr, flush=True)

        try:
            while True:
                encrypted = hid_device.read(self.READ_SIZE, timeout_ms=1000)
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

                decrypted = self.decrypt_data(normalized)
                diagnostics = self.track_packet(decrypted[0])
                timestamp = local_clock()
                eeg_outlet.push_sample(self.decode_eeg_sample(decrypted), timestamp)
                diagnostics_outlet.push_sample(diagnostics.as_lsl_sample(), timestamp)
                loss_reporter.report(diagnostics)

                if debug_outlet:
                    debug_outlet.push_sample(self.decode_debug_sample(decrypted))
                if log_writer:
                    self.log_decrypted_packet(log_writer, decrypted)
                    logged_packets += 1
                    if logged_packets == 1:
                        print("Received first valid decrypted packet.", file=sys.stderr, flush=True)
                    if logged_packets % self.LOG_FLUSH_INTERVAL == 0:
                        log_handle.flush()
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
