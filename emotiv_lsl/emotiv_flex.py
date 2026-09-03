"""Direct HID reader for the original EPOC Flex (Flex 1.0) controller.

Flex 1.0 is not the same protocol as Flex 2.0.  The wireless dongle exposes
an ``EEG Signals`` HID collection with 32-byte reports.  The reports are
AES-128-ECB encrypted and the decrypted payload contains 32 signed seven-bit
delta samples.  This module intentionally does not touch the Cortex session;
it can therefore run beside the quality-only Cortex bridge.

The packet order is always the wire labels (LA..LQ and RA..RQ). Flex 1.0
allows the user to map those wires to arbitrary 10-20 positions, so callers
can replace the LSL labels while retaining the wire name in channel metadata.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import hid
from Crypto.Cipher import AES
from pylsl import StreamInfo, StreamOutlet

from emotiv_lsl.emotiv_base import EmotivBase


class EmotivFlex(EmotivBase):
    """Read the original (Flex 1.0) wireless dongle's EEG HID reports."""

    READ_SIZE = 32
    SAMPLE_RATE = 128
    LSB_UV = 0.51
    ADC_BITS = 14
    ADC_MAX = (1 << ADC_BITS) - 1
    ADC_MIDPOINT = 1 << (ADC_BITS - 1)
    EEG_USAGE = 2
    EEG_PRODUCT_MARKER = "eeg signals"

    # The controller uses A-H and J-Q on each side (I is the reference slot).
    CH_NAMES = (
        "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH",
        "LJ", "LK", "LL", "LM", "LN", "LO", "LP", "LQ",
        "RA", "RB", "RC", "RD", "RE", "RF", "RG", "RH",
        "RJ", "RK", "RL", "RM", "RN", "RO", "RP", "RQ",
    )

    def __init__(
        self,
        remove_dc: bool = False,
        channel_labels: Sequence[str] | None = None,
        montage_name: str | None = None,
        references: dict[str, str] | None = None,
        serial_number: str | None = None,
    ) -> None:
        self.cipher: AES | None = None
        self.remove_dc = remove_dc
        if channel_labels is None:
            channel_labels = self.CH_NAMES
        if len(channel_labels) != len(self.CH_NAMES):
            raise ValueError(
                f"Flex channel mapping must contain {len(self.CH_NAMES)} labels; "
                f"got {len(channel_labels)}"
            )
        normalized_labels = tuple(str(label).strip() for label in channel_labels)
        if any(not label for label in normalized_labels):
            raise ValueError("Flex channel labels must not be empty")
        self.channel_labels = normalized_labels
        self.montage_name = montage_name or "wire labels"
        self.references = {
            str(key): str(value)
            for key, value in (references or {}).items()
            if str(value).strip()
        }
        self.serial_number = serial_number.strip() if serial_number else None
        self._adc = [self.ADC_MIDPOINT] * len(self.CH_NAMES)
        self._last_counter: int | None = None
        self.packet_gaps = 0

    @staticmethod
    def _is_eeg_device(device: dict) -> bool:
        product = str(device.get("product_string") or "").lower()
        return (
            device.get("usage") == EmotivFlex.EEG_USAGE
            and EmotivFlex.EEG_PRODUCT_MARKER in product
        )

    def get_hid_devices(self) -> list[dict]:
        return [device for device in hid.enumerate() if self._is_eeg_device(device)]

    def describe_hid_device(self, device: dict) -> str:
        return (
            f"path={device.get('path')} "
            f"product={device.get('product_string')} "
            f"serial={device.get('serial_number')} "
            f"usage_page={device.get('usage_page')} "
            f"usage={device.get('usage')} "
            f"interface_number={device.get('interface_number')}"
        )

    def get_hid_device(self) -> dict:
        devices = self.get_hid_devices()
        if self.serial_number:
            devices = [
                device
                for device in devices
                if str(device.get("serial_number") or "") == self.serial_number
            ]
        if not devices:
            if self.serial_number:
                raise RuntimeError(
                    "Original EPOC Flex EEG HID collection with serial "
                    f"{self.serial_number!r} was not found; check the dongle "
                    "serial or omit --serial when only one Emotiv receiver is connected."
                )
            raise RuntimeError(
                "Original EPOC Flex EEG HID collection not found; "
                "connect the Flex 1.0 dongle and headset first."
            )
        if len(devices) > 1:
            descriptions = "; ".join(self.describe_hid_device(device) for device in devices)
            raise RuntimeError(
                "More than one Flex-compatible EEG HID collection was found. "
                "Disconnect other Emotiv receivers or pass --serial. "
                f"Candidates: {descriptions}"
            )
        return devices[0]

    @staticmethod
    def get_crypto_key(device: dict) -> bytes:
        """Build the Flex 1.0 AES key from the dongle serial.

        The Flex 1.0 Cortex implementation consumes four bytes from a derived
        serial array and places them into the fixed 16-byte key pattern below.
        In the supplied Flex 1.0 dongle those bytes are the first four ASCII
        bytes of the reversed HID serial. The key is per device, but the
        construction is model/protocol-specific and must not be shared with
        EPOC X or Flex 2.0.
        """

        serial = str(device.get("serial_number") or "").strip()
        if len(serial) < 4:
            raise RuntimeError(
                "Flex 1.0 dongle did not provide a usable serial number"
            )
        seed = serial[::-1].encode("ascii", errors="strict")[:4]
        a, b, c, d = seed
        return bytes((
            a, 0x00, b, 0x15,
            c, 0x00, d, 0x0C,
            c, 0x00, b, 0x44,
            a, 0x00, b, 0x58,
        ))

    @staticmethod
    def normalize_packet(data: list[int] | bytes | bytearray) -> bytes | None:
        """Accept hidapi's 32-byte report or a 33-byte report-ID variant."""

        packet = bytes(data)
        if len(packet) == EmotivFlex.READ_SIZE:
            return packet
        if len(packet) == EmotivFlex.READ_SIZE + 1 and packet[0] == 0:
            return packet[1:]
        return None

    @staticmethod
    def unpack_signed_deltas(packet: bytes) -> list[int]:
        """Extract Cortex's 32 signed seven-bit values from one plaintext.

        Bytes 0 and 1 are packet metadata.  Bytes 2..29 contain 224 bits,
        exactly 32 values of seven bits each, in MSB-first order.  The wire
        values are unsigned; two's-complement conversion gives -64..63.
        """

        if len(packet) != EmotivFlex.READ_SIZE:
            raise ValueError("Flex 1.0 plaintext packets must be 32 bytes")
        payload = packet[2:30]
        values: list[int] = []
        for index in range(32):
            value = 0
            for bit in range(7):
                bit_position = index * 7 + bit
                value = (value << 1) | (
                    (payload[bit_position // 8] >> (7 - bit_position % 8)) & 1
                )
            values.append(value if value < 64 else value - 128)
        return values

    def configure_cipher(self, device: dict) -> None:
        self.cipher = AES.new(self.get_crypto_key(device), AES.MODE_ECB)

    def decrypt_data(self, data: bytes | bytearray | list[int]) -> bytes:
        if self.cipher is None:
            raise RuntimeError("Flex 1.0 HID cipher has not been configured")
        packet = self.normalize_packet(data)
        if packet is None:
            raise ValueError("invalid Flex 1.0 HID report length")
        return self.cipher.decrypt(packet)

    def decode_data(self, data: bytes | bytearray | list[int]) -> list[float]:
        plaintext = self.decrypt_data(data)
        counter = plaintext[0] & 0x7F
        if self._last_counter is not None and counter != ((self._last_counter + 1) & 0x7F):
            self.packet_gaps += 1
            # Missing deltas cannot be reconstructed.  Start the next run at
            # the ADC midpoint rather than allowing the missing samples to
            # create an unbounded offset.
            self._adc = [self.ADC_MIDPOINT] * len(self.CH_NAMES)
        self._last_counter = counter

        for index, delta in enumerate(self.unpack_signed_deltas(plaintext)):
            self._adc[index] = max(0, min(self.ADC_MAX, self._adc[index] + delta))

        if self.remove_dc:
            return [
                (value - self.ADC_MIDPOINT) * self.LSB_UV
                for value in self._adc
            ]
        return [value * self.LSB_UV for value in self._adc]

    def validate_data(self, data) -> bool:
        return self.normalize_packet(data) is not None

    def get_stream_info(self) -> StreamInfo:
        info = StreamInfo(
            "Epoc Flex 1.0",
            "EEG",
            len(self.CH_NAMES),
            self.SAMPLE_RATE,
            "float32",
        )
        channels = info.desc().append_child("channels")
        for wire_label, label in zip(self.CH_NAMES, self.channel_labels):
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("wire", wire_label)
            channel.append_child_value("location", label)
            channel.append_child_value("unit", "microvolts")
            channel.append_child_value("type", "EEG")
            channel.append_child_value("scaling_factor", str(self.LSB_UV))

        cap = info.desc().append_child("cap")
        cap.append_child_value("name", "EPOC Flex 1.0")
        cap.append_child_value("labelscheme", "user-configurable 10-20")
        cap.append_child_value("montage", self.montage_name)
        for reference, location in sorted(self.references.items()):
            cap.append_child_value(reference.lower(), location)
        return info

    def main_loop(self) -> None:
        outlet = StreamOutlet(self.get_stream_info())
        device = self.get_hid_device()
        hid_device = hid.device()
        hid_device.open_path(device["path"])
        self.configure_cipher(device)
        print(
            f"Streaming original EPOC Flex from {self.describe_hid_device(device)}",
            file=sys.stderr,
            flush=True,
        )

        try:
            while True:
                packet = hid_device.read(self.READ_SIZE, timeout_ms=1000)
                if not self.validate_data(packet):
                    continue
                outlet.push_sample(self.decode_data(packet))
        finally:
            hid_device.close()
