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

The current service uses a connection-time type-7 control packet to establish a
per-session key. The packet is not present in the 200 steady-state EEG reports
attached to issue #17, which is why those reports cannot be decoded by
themselves.

The reverse-engineered path is:

1. Observe the type-7 startup/control report on the HID connection. Cortex
   parses its hex payload; the first four decoded payload bytes are the session
   seed.
2. Format those four bytes as eight uppercase hexadecimal characters and build
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
seed_hex = control_payload[:4].hex().upper()
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

The seed extraction and report framing still need to be validated against a
real power-on capture. Do not replace the old decoder globally until that test
passes; the launcher may need to select the old or new path by firmware.

## Runtime relationship with Cortex quality streams

The standalone capture script is only a reverse-engineering tool. The finished
launcher will not require the user to run it on every acquisition. The HID
reader will watch for the type-7 control report when the headset connects,
derive the key, and retain that key for the current headset connection. A new
headset power cycle or reconnect is expected to establish a new session key;
the reader should therefore remain open across normal operation and reconnect
handling should repeat the same handshake step.

This can run alongside the existing Cortex quality path. Leave EMOTIV Launcher
and the Cortex service running for `dev`/`eq`; the direct HID reader does not
need Cortex's `eeg` scope and does not replace the quality subscriptions. The
only startup-order constraint is that the HID reader must be attached before a
new connection's control report is emitted. If a reader is restarted after the
headset is already connected and has missed that report, it should request a
reconnect/power cycle (or use a cached key) rather than attempting to decrypt
the EEG stream with the legacy key.

## Capture and implementation sequence

`examples/capture_hid_startup.py` polls for Emotiv HID interfaces before the
headset is powered, records both receiver and EEG interfaces, and preserves
raw report lengths and bytes. Run it before powering the headset:

```bash
python -m pipenv run python examples/capture_hid_startup.py --seconds 20
```

The next implementation step is to identify the type-7 report in that CSV,
extract its seed, add a firmware-aware cipher/handshake layer, and run the
decoded output through the existing channel conversion. Keep the capture's
`report_hex` unchanged; only the serial-number column needs redaction before
sharing.

The recorder does not call Cortex, request access, or require an EEG license.
It is therefore compatible with leaving EMOTIV Launcher running when that is
needed to keep the Cortex service alive. Start the recorder while the headset
is off; Launcher and the recorder can then observe the interfaces as the
headset connects. If the operating system reports an access-denied error, stop
only applications that are actively subscribing to EEG (such as EmotivPRO or a
second raw-HID launcher), then retry; do not power the headset on before the
recorder is ready.

## Downgrade status

The public EPOC X updater documents upgrading the USB and Bluetooth chips, not
rollback. The service exposes recovery/update operations, but no public,
verified `0x710`/`0x720` image or anti-rollback-safe procedure was found. A
desktop-app downgrade does not downgrade the headset firmware. The safe route
for rollback is Emotiv Support supplying the signed images and exact two-chip
procedure; random DFU files or updating only one chip can brick or desynchronise
the headset.
