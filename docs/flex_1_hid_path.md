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
input at offset 2. The unpacked values are unsigned on the wire. Two's
complement conversion gives signed deltas from -64 through +63. This exactly
matches the Flex 1.0 specification's maximum slew of 32.64 µV/sample at
0.51 µV/LSB (`32.64 / 0.51 = 64`). The acquisition code therefore maintains a
32-element ADC state and adds one signed delta per report. It initializes the
state at the 14-bit midpoint (8192); the absolute DC offset is arbitrary for
this AC-coupled device. `--remove-dc` emits midpoint-subtracted values.

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

If a USB report is lost, the 7-bit deltas between the missing and next report
cannot be reconstructed. The reader counts the counter discontinuity and
restarts the ADC state at the midpoint to avoid carrying a false accumulated
offset. This is preferable to mixing motion/status packets into the EEG index.

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
