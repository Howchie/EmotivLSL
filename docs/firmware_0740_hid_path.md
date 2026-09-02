# EPOC X firmware `0x740` direct-HID path

This document records the no-Cortex-license path being investigated for EPOC X
headsets on firmware `0x740`. It is intentionally separate from the Cortex
bridge: Cortex can decode this firmware, but raw `eeg` access requires a paid
EEG scope and is not a solution for an unlicensed setup.

## What changed

The legacy decoder in `emotiv_lsl/emotiv_epoc_x.py` assumes that every 32-byte
EEG report is transformed with:

1. XOR every byte with `0x55`.
2. AES-128-ECB decrypt with a key derived from the HID serial suffix.
3. Convert the decrypted byte pairs at offsets `2..15` and `18..31` into the
   14 channels, then apply the existing channel swaps.

That path works for older firmware (the issue report compares `0x710` with
`0x740`). Firmware `0x740` returns plausible-length reports, but the old
transform produces uniform/random-looking values.

The current Cortex service contains an explicit firmware gate at `>= 0x73f`,
so this is a protocol change rather than a bad serial number or a changed EEG
sample layout.

## Intended firmware-740 decoder

The successful USBPcap trace shows that the key material is available through a
HID feature report on the `EEG Signals` collection. The relevant control
transaction is a `GET_REPORT` (feature, report ID 0) on interface 1. In the
capture, the returned bytes were:

```text
20 30 06 ff 07 40 e5 02 01 ce 2b 01 00 01 10 00 10
```

The `06 ff` marker is followed by the big-endian firmware number (`07 40` =
`0x740`) and the four-byte session seed (`e5 02 01 ce`). This is preferable to
waiting for a transient input packet: a reader can query the feature report
after opening the HID collection, including when Launcher has already connected
the headset.

The reverse-engineered path is:

1. Query feature report ID 0 on the `EEG Signals` HID collection and locate the
   `06 ff` marker. Read the next two bytes as the firmware number and the next
   four bytes as the session seed.
2. Format the four seed bytes as eight uppercase hexadecimal characters and build
   this ASCII key material (where `FW` is four uppercase hex digits, e.g.
   `0740`):

   ```text
   Vohcha7e + seed_hex[0:4] + Ut3phaej + seed_hex[4:8] + FW
   ```

3. Compute `SHA-256` of that ASCII string. The 32-byte digest is the
   AES-256-ECB key for the EEG reports.
4. XOR every byte of each normal EEG report with `0x14`, then AES-256-ECB
   decrypt the resulting 32 bytes (two 16-byte blocks).
5. Feed the decrypted 32-byte result through the existing EPOC X channel
   conversion and channel-order swaps. The report remains 14 channels; the
   change is the session setup and cryptographic transform.

Pseudocode:

```python
feature = bytes.fromhex("203006ff0740e50201ce2b010001100010")
offset = feature.index(b"\x06\xff")
firmware = int.from_bytes(feature[offset + 2:offset + 4], "big")
seed_hex = feature[offset + 4:offset + 8].hex().upper()
key_material = (
    b"Vohcha7e" + seed_hex[:4].encode("ascii") +
    b"Ut3phaej" + seed_hex[4:8].encode("ascii") +
    f"{firmware:04X}".encode("ascii")
)
aes_key = hashlib.sha256(key_material).digest()
plain = AES.new(aes_key, AES.MODE_ECB).decrypt(
    bytes(byte ^ 0x14 for byte in encrypted_report)
)
```

This path was validated against the supplied `data/capture.pcap` trace: the
derived key produces correctly varying channel values and the decrypted packet
counter advances through 5,188 captured EEG reports (including the initial
baseline period). The implementation now
selects this path automatically for feature-report firmware `>= 0x73f` and
retains the serial-derived AES-128 path for older firmware.

## Runtime relationship with Cortex quality streams

The standalone capture script is only a reverse-engineering tool. The launcher
queries the feature report after opening the HID collection, derives the key,
and retains it for the current headset connection. A new headset power cycle or
reconnect should repeat the feature query so any new seed is picked up.

This runs alongside the existing Cortex quality path. Leave EMOTIV Launcher and
the Cortex service running for `dev`/`eq`; the direct HID reader does not need
Cortex's `eeg` scope and does not replace the quality subscriptions. There is no
startup race with a one-shot input packet: the feature report can be queried
after the headset is already connected. If the feature query fails, the reader
falls back to the legacy path for compatibility with older firmware and logs
that choice.

## Capture notes

`examples/capture_hid_startup.py` polls for Emotiv HID interfaces before the
headset is powered, records both receiver and EEG interfaces, and preserves
raw report lengths and bytes. Run it before powering the headset:

```bash
python -m pipenv run python examples/capture_hid_startup.py --seconds 60 >hid_capture.log 2>&1
```

The CSV is written separately; `2>&1` preserves the recorder's diagnostics in
the log as well. Without it, a shell's `>` redirect captures only stdout while
the interface-discovery messages remain on the console (stderr).

The CSV remains useful for steady-state reports, but the successful
`data/capture.pcap` trace is the artifact that exposed the feature report. Keep
raw report bytes unchanged; only the serial-number column needs redaction before
sharing the CSV.

The first hidapi-only capture opened both interfaces but produced only 32-byte
input reports from `EEG Signals` (`usage=2`); the receiver (`usage=16`) was
silent. USBPcap preserved the missing control transaction and confirmed that it
is a feature transfer on the EEG collection, not a receiver input report.

The recorder does not call Cortex, request access, or require an EEG license.
It is therefore compatible with leaving EMOTIV Launcher running when that is
needed to keep the Cortex service alive. Start the recorder while the headset
is off; then let Launcher/Cortex notice and connect the headset so its normal
initialization exchange occurs while the recorder is listening. If the
operating system reports an access-denied error, stop only applications that
are actively subscribing to EEG (such as EmotivPRO or a second raw-HID
launcher), then retry; do not power the headset on before the recorder is
ready.

## Downgrade status

The [public EPOC X updater](https://emotiv.gitbook.io/updating-firmware-1/epoc-x)
documents upgrading the USB and Bluetooth chips, not rollback. The service exposes
recovery/update operations, but no public,
verified `0x710`/`0x720` image or anti-rollback-safe procedure was found. A
desktop-app downgrade does not downgrade the headset firmware. The safe route
for rollback is Emotiv Support supplying the signed images and exact two-chip
procedure; random DFU files or updating only one chip can brick or desynchronise
the headset.
