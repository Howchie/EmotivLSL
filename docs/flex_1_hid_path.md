# Original EPOC Flex (Flex 1.0) HID path

This document is deliberately separate from the EPOC X firmware notes and from
the Flex 2.0 protocol. The captures supplied for this investigation are from an
original Flex 1.0 controller:

* `data/flex1.pcap` is the USB-cable side. It contains the `FLEX` device and
  status/heartbeat traffic, but no continuous EEG report stream.
* `data/flex2.pcap` is the Flex 1.0 proprietary-radio dongle (the filename is
  user-supplied and does **not** mean Flex 2.0). Its `EEG Signals` collection
  carries the continuous EEG reports.

## Confirmed report path

The dongle enumerates an Emotiv HID collection with these useful properties:

* product: `EEG Signals`
* usage: `2`
* interface: `1`
* input endpoint: `0x82`
* input report: 32 bytes
* observed report interval: approximately 8 ms (the Flex 1.0 nominal rate is
  128 samples/s)

The other dongle endpoints contain control/status traffic. The direct reader
selects only the `EEG Signals` collection, so Flex motion reports do not get
interleaved with the EEG sample vector.

## Decryption

The Flex 1.0 Cortex binary has a separate `EPOCFlexDataDecryption` class (and a
separate `Flex2DataDecryption` class). Its Flex 1.0 key-setup routine consumes
four bytes from a serial-derived array and uses the following AES-128-ECB key
layout. In the supplied Flex 1.0 dongle, those bytes are the first four ASCII
bytes of the reversed HID serial:

```
[a, 00, b, 15, c, 00, d, 0c,
 c, 00, b, 44, a, 00, b, 58]
```

For the supplied dongle serial `UD202208120064B2`, this produces:

```
32 00 42 15 34 00 36 0c 34 00 42 44 32 00 42 58
```

That key decrypts all 1,681 complete EEG reports in `flex2.pcap` without the
EPOC X XOR step. The derivation is strongly supported by the Cortex binary and
the capture, but should be checked against a second Flex 1.0 dongle before it
is treated as a universal hardware guarantee.

## Decrypted packet layout

The decrypted 32-byte report is:

```
byte 0       7-bit packet counter (wraps at 128)
byte 1       Flex packet/status metadata
bytes 2..29  224 bits = 32 values × 7 bits, MSB first
bytes 30..31 packet/status metadata
```

Cortex calls its common bit unpacker with `bits=7`, `count=32`, and a 28-byte
input at offset 2. The unpacked values are unsigned on the wire and are
**offset binary**: 64 encodes a zero delta, so the signed delta is `raw - 64`,
spanning -64 through +63. This exactly matches the Flex 1.0 specification's
maximum slew of 32.64 µV/sample at 0.51 µV/LSB (`32.64 / 0.51 = 64`).

This was originally implemented as a two's-complement conversion, which is
wrong and was corrected after checking a worn-cap recording
(`Pilot_Flex_001.xdf`, 28.8 min, 32 channels). The raw seven-bit values form a
clean unimodal distribution centred on 64 - the signature of offset binary.
Two's complement splits that peak at 64 and maps the *most common* values (a
small delta) onto the extremes +-64: 71% of decoded deltas landed in the outer
quarter of the range and only 2% within +-8 of zero, every one of the 32
channels acquired a uniform +1.2 count/sample bias, and 28 of 32 channels drove
the accumulator into the 14-bit rail. Reading the same bits as offset binary
restores a delta distribution peaked at zero (46% within +-8), drops the
broadband EEG estimate from 107 µV RMS to 50 µV RMS after a 0.5 Hz high-pass,
and recovers the analogue chain's ~43 Hz roll-off (power above 45 Hz falls from
0.26% to 0.004% of the total), none of which is visible under the old rule. The acquisition code therefore maintains a
32-element ADC state and adds one signed delta per report. It initializes the
state at the 14-bit midpoint (8192); the absolute DC offset is arbitrary for
this AC-coupled device. `--remove-dc` emits midpoint-subtracted values.

### DC restore

The protocol never transmits an absolute level, so the accumulator has no
anchor: packet loss, electrode drift and floating wires all leave a permanent
offset, and a pure integrator has no mechanism to shed any of them. On the
pilot recording a pure integrator leaves 96.6% of samples outside the valid
14-bit range.

The state is therefore reconstructed with a leak rather than a pure integral:

```text
state = state * exp(-2*pi*fc/128) + delta
```

`fc` defaults to 0.16 Hz (`--dc-restore-hz`, 0 disables it), which is the
Flex 1.0 passband's documented high-pass corner; the matching 43 Hz roll-off is
already present in the analogue chain and is visible in the decoded data. With
the leak in place the accumulator stays bounded (99th percentile 814 counts,
about 415 µV) and cannot reach the rail, where clamping would otherwise discard
deltas asymmetrically and corrupt the channel. The cost is content below
~0.16 Hz, which this device never transmitted in the first place.

Flex 1.0 has 32 configurable sensor wires plus CMS/DRL references. The packet
order is fixed as `LA..LQ`, then `RA..RQ`; CMS and DRL are references and are not
included in the EEG sample vector. The direct stream loads
`epoch_flex_electrodes.json`, which is intentionally compatible with the flat
JSON emitted by Emotiv Launcher (including its CMS/DRL entries). It preserves
the packet order, replaces the LSL labels with the configured locations, and
stores the original wire label in channel metadata. Missing entries fall back
to their wire labels, so a partial configuration cannot shift or drop data.

## Running the experimental direct stream

The production EPOC X launcher remains unchanged. The Flex 1.0 path is exposed
separately while the packet model is being validated:

```
python -m pipenv run python main_flex.py
```

Use `--remove-dc` if the downstream consumer expects midpoint-subtracted EEG:

```
python -m pipenv run python main_flex.py --remove-dc
```

The headset/dongle should already be connected before starting this process,
which is the normal steady-state workflow. It does not need to observe the
initial pairing handshake. EMOTIV Launcher can remain running in parallel so
the Cortex API bridge can provide `dev`/`eq` quality streams; this direct HID
reader does not request the licensed Cortex `eeg` stream.

When using `main_all_flex.py`, the same montage is also supplied as the
`mappings` object in Cortex's `controlDevice connect` request. The original
Flex path requires that object when Cortex connects a discovered headset.

If a USB report is lost, the 7-bit deltas between the missing and next report
cannot be reconstructed. The reader counts the counter discontinuity and
publishes it on the `Epoc Flex 1.0 Packet Diagnostics` LSL stream. By default it
continues the ADC accumulator: one omitted report can omit only one bounded
delta per channel, whereas restarting every channel at the midpoint creates a
large artificial step. The old midpoint-reset behavior remains available with
`--reset-on-gap` for compatibility, but should normally be left disabled.

The offset a gap leaves behind is not permanent: the DC restore above decays it
with the same 0.16 Hz time constant, so loss no longer accumulates without
bound. Note that the midpoint reset was incidentally acting as a crude DC
restore, which is why disabling it without the leak in place let the
accumulator run away.

### Packet-gap discontinuities and analysis

The missing delta cannot be recovered, so the sample after a gap may carry a
small persistent offset. This is an acquisition discontinuity, not a change in
the CMS/DRL hardware reference, and `--remove-dc` does not detect or repair it.
Do not treat the affected samples as a genuine EEG transient. For analysis, use
the diagnostics stream to mark the gap and reject a short window around it; only
interpolate if that is appropriate for the specific downstream method. The
diagnostics stream is 128 Hz and has the channels `COUNTER`,
`EXPECTED_COUNTER`, `GAP_FLAG`, `MISSING_REPORTS`, `CUMULATIVE_MISSING`,
`RESET_FLAG`, `CUMULATIVE_GAPS`, and `CUMULATIVE_RESETS`, timestamped with the
corresponding EEG sample. The same events are summarized on stderr - the first
one immediately, then at most once every ten seconds - so packet loss is visible
in the console without an LSL consumer attached.

Replaying `data/flex2.pcap` (1,681 reports, 8 counter discontinuities) through
the reader shows the effect directly: carrying the accumulator keeps the largest
step between consecutive samples at 32.64 uV, which is the protocol's maximum
slew, while the legacy midpoint reset produces steps of up to 418 uV.

## Firmware downgrade status

No safe, public Flex 1.0 firmware downgrade image or documented downgrade
procedure has been identified. The archived files available from EMOTIV are
Cortex/Launcher application packages, not a verified controller firmware image.
The direct Flex 1.0 HID path is therefore the practical route for data access;
do not attempt to flash an EPOC X or Flex 2.0 image onto this controller.

## References

* [EPOC Flex 1.0 technical specifications](https://emotiv.gitbook.io/epoc-flex-user-manual/epoc-flex/technical-spec)
* [Flex sensor mapping and wire labels](https://emotiv.gitbook.io/cortex-api/headset/controldevice)
* [Cortex raw EEG stream documentation](https://emotiv.gitbook.io/emotivpro-v3/data-streams/raw-eeg)
