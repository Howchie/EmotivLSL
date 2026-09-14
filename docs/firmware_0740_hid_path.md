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
baseline period).

## Supporting both firmware generations from one code path

The feature report chooses the *preferred* key, not the only one. On each
connection the reader builds every candidate it can - the serial-derived
AES-128/`0x55` pair for pre-0x740 firmware and, when the feature report answers
with the `06 ff` marker, the SHA-256/AES-256/`0x14` pair - and then confirms the
choice against the packet stream: under the correct key byte 0 of the decrypted
report advances by one per report, and under the wrong key it is effectively
random. Up to twelve reports are buffered for that check and then published, so
confirming the key costs no samples.

This matters because the feature query is not a reliable firmware oracle. Older
firmware may not answer it, may answer without the marker, or may answer with
something that parses as a high firmware number. Any of those previously routed
an older headset down the `0x740` path or aborted the run; the counter check now
corrects the choice instead. `--firmware 0740` pins the feature-report key.

`--firmware legacy` (`run_epochX_legacy.bat` on Windows) does not use any of
this. It hands the session to `emotiv_lsl/emotiv_epoc_x_legacy.py`, a frozen
copy of the reader from before 0x740 support (commit `9944584`). That reader
matches interfaces on `manufacturer_string == 'Emotiv'` and probes them in
enumeration order. It closes the probed handle and reopens it, takes the key
from the serial number, and publishes every 32-byte report. It never queries
the feature report, verifies the key, measures the rate or filters packets.
The only change from the original is that its outlets declare
`--sample-rate` when one is given, and `config.SRATE` otherwise. The original
reader raised an error if no dongle was found or the headset sent nothing during
its probe. `EmotivEpocX.wait_for_legacy_reader` now catches those two errors
and retries every two seconds until the headset streams. The outlets are only
created once it does. In `main_all.py`, if the EEG reader thread dies, the
whole launcher closes and exits with code 1.

Automatic detection does not currently stream from a 0x720 headset. The
reader opens the stream, but it contains no samples and the rate measurement
falls back to 128 Hz. Use `--firmware legacy` for 0x720 headsets until that is
fixed.

Interface discovery is firmware-independent for the same reason:

* Emotiv interfaces are matched on a case-insensitive manufacturer string *or*
  the receiver's vendor id `0x1234`, because Windows does not always report a
  manufacturer string for these collections.
* Candidates are probed in likelihood order (`EEG Signals` product string, then
  `usage == 2`, then interface number) instead of enumeration order.
* An interface that cannot be opened - Windows claims some HID collections, and
  another application may hold one - is logged and skipped rather than raising.
* If nothing streams during probing the best candidate is used anyway, because a
  connected-but-idle headset is not a discovery failure.
* The probed handle stays open into the streaming loop, so the collection is no
  longer closed and reopened between probing and streaming.
* `python main.py --list-hid` prints every HID interface the operating system
  reports, which is the first thing to check when a headset is not recognized.

## Sample rate

EPOC X runs at either 128 Hz or 256 Hz depending on how the headset is
configured, and nothing in the HID report explicitly labels the rate: the
packet counter increments once per report and wraps after one second of samples
(at 127 for 128 Hz, or at 255 for 256 Hz). The reader therefore times incoming
reports for two seconds after the cipher is confirmed, snaps the result to the
nearer supported rate, and declares that on every outlet. The
reports consumed while measuring are published, not dropped, and the stream
metadata records `sample_rate_source` as `measured` or `configured`.

This matters because a wrong nominal rate is silent and doubles or halves every
frequency downstream. The pilot recording `Pilot_EpochX_001.xdf` was captured
with the rate hardcoded to 128 Hz while the headset streamed at 256 Hz
(217,599 samples over 849.5 s = 256.1 Hz), so its alpha band sits at 4-6.5 Hz
rather than 8-13 Hz and its apparent bandwidth ends at 24 Hz rather than 47 Hz.
Nothing is wrong with the samples themselves - only the declared rate - so such
a recording is corrected by relabelling it.

`--sample-rate HZ` pins the rate when measuring is not wanted; an idle headset
or an implausible measurement falls back to `config.SRATE` and says so.

An LSL outlet's nominal rate is fixed when the outlet is created and cannot be
changed afterwards, which is why the rate is settled before the outlets are
built. Two consequences follow. First, the EEG outlet now appears a few seconds
into startup rather than immediately, and only once the headset is actually
streaming; a recorder that resolves streams once, at launch, should be started
after the reader or told to refresh. Passing `--sample-rate` skips the
measurement and removes that delay. Second, because a pinned rate cannot be
checked in advance, `SampleRateMonitor` re-checks it against real arrivals over
the first few seconds of streaming and warns loudly on a mismatch. That check
runs whether the rate was measured or pinned, so a wrong rate is never silent.

`config.SRATE` is no longer the source of truth for a live stream - it is only
the fallback when the rate cannot be measured, and the default that
`--sample-rate` overrides. Consumers should read `nominal_srate()` from the
stream rather than importing the constant.

## Packet diagnostics

Each decoded report is also mirrored to an always-on `Epoc X Packet Diagnostics`
LSL stream. Its counter fields use the one-second range for the configured EEG
rate (0..127 at 128 Hz or 0..255 at 256 Hz) and identify skipped or duplicated reports;
the captured baseline packet used when a session restarts is reported with
`RESET_FLAG` instead of being counted as packet loss. That baseline report is
`00 10` followed by the byte pair `00 80` on every channel - the ADC midpoint,
not a run of `0x80` bytes - and in `data/capture.pcap` it appears once, at the
`185 -> 0` counter transition. Recognizing it keeps that restart from being
reported as 70 missing reports; the capture otherwise contains no packet loss at
all.

Counter discontinuities are additionally summarized on stderr - the first one
immediately, then at most once every ten seconds - so a run started from a
launcher script shows packet loss in the console. The existing optional
`Epoc X Debug` stream remains available via `--emit-debug`.

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
