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
**offset binary**: 63 encodes a zero delta, so the signed delta is `raw - 63`,
spanning -63 through +64. The +64 end matches the Flex 1.0 specification's
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

### The zero is 63, not 64

The reader first took 64 as the zero. The two Flex oddball recordings of
2026-09-15 (15 and 18 min) show why that is one count off:

* Decoded as `raw - 64`, the deltas average −1.00 counts/sample. Every channel
  lies between −0.95 and −1.12, and the bias is steady across 10 s windows.
* The analogue chain is AC-coupled, so the true mean delta is zero. Taken
  literally, the bias is a 65 µV/s ramp: about 70 mV per recording, far beyond
  the 14-bit range.
* The 0.16 Hz leak the reader applied then turns the bias into a fixed offset
  of −1 / (1 − leak) counts, or −65 µV. The published signals sit at
  −61 to −63 µV.
* Decoded as `raw - 63`, a pure integral of the longest loss-free stretches
  (70–140 s) has no net drift: median −0.2 to +1.6 µV/s.

Recordings made before the fix have no `delta_zero` entry in the stream's
`cap` metadata. Their received deltas are exactly recoverable by undoing the
leak, so they can be re-decoded offline.

### No DC restore by default

The protocol never transmits an absolute level, so the accumulator has no
anchor. The reader used to reconstruct it with a leak,
`state = state * exp(-2*pi*fc/128) + delta`, with `fc` = 0.16 Hz, and clamp it
to the 14-bit range. Both were there because a pure integrator ran away: on the
pilot recording it left 96.6% of samples outside the 14-bit range. That runaway
was the off-by-one zero above, not a property of the device. At one count per
sample, a pure integral starting at the midpoint leaves the 14-bit range after
64 s. For the 28.8 min pilot that predicts 96.3% of samples out of range.

With the zero corrected, the leak is off by default (`--dc-restore-hz 0`).
0.16 Hz is the Flex 1.0 passband's documented high-pass corner, but that corner
is already in the analogue chain: the pure integral's spectrum rises as 1/f²
down to ~0.3 Hz and flattens below ~0.2 Hz. A leak at the same corner would add
a second high-pass on top, which distorts slow components such as the P3. The
state is no longer clamped either. It is an integral from an arbitrary start,
not an ADC reading, and the offsets left by lost reports make it wander. In the
oddball recordings it wandered 2–6 mV peak-to-peak over a session. Consumers
that display or threshold the raw stream must high-pass it themselves.
`--dc-restore-hz` still enables the leak for anyone who wants a bounded
signal.

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

### The display stream

The accumulator has no anchor, so the published signal is an integral from an
arbitrary start and drifts by millivolts over a session: a 50 uV EEG feature is
under 1% of an autoscaled trace, which is why nothing appears to happen in a live
viewer even when the data is good. The reader therefore publishes a second
outlet, `Epoc Flex 1.0 Display`, carrying the same samples through a one-pole
high pass at `--display-hz` (default 0.5 Hz, 0 to suppress the outlet). It is the
leaky accumulator the reader used to apply to the archive, moved off the archive
and onto a viewer-only copy, and costs one multiply-add per channel per sample.

Record and analyse `Epoc Flex 1.0`. The display copy is a second high pass on top
of the analogue chain's 0.16 Hz one; measured on the oddball runs, that costs the
slow late components 10-20% of their amplitude and adds an undershoot after every
large deflection. `analysis/emotiv/loading.py` refuses the display stream by name
so a recording of the wrong outlet fails loudly rather than quietly.

If a USB report is lost, the 7-bit deltas between the missing and next report
cannot be reconstructed. The reader counts the counter discontinuity and
publishes it on the `Epoc Flex 1.0 Packet Diagnostics` LSL stream. By default it
continues the ADC accumulator: one omitted report can omit only one bounded
delta per channel, whereas restarting every channel at the midpoint creates a
large artificial step. The old midpoint-reset behavior remains available with
`--reset-on-gap` for compatibility, but should normally be left disabled.

### Packet-gap discontinuities and analysis

The missing deltas cannot be recovered, so everything after a gap carries a
persistent offset equal to the signal's change across the gap. With no leak it
is a constant step; an analysis high-pass turns it into a decaying tail. This
is an acquisition discontinuity, not a change in
the CMS/DRL hardware reference, and `--remove-dc` does not detect or repair it.
Do not treat the affected samples as a genuine EEG transient. For analysis, use
the diagnostics stream to mark the gap and reject a short window around it; only
interpolate if that is appropriate for the specific downstream method. The
diagnostics stream is 128 Hz and has the channels `COUNTER`,
`EXPECTED_COUNTER`, `GAP_FLAG`, `MISSING_REPORTS`, `CUMULATIVE_MISSING`,
`RESET_FLAG`, `CUMULATIVE_GAPS`, `CUMULATIVE_RESETS`, and `FILLED`, one row per
EEG sample with the same timestamp. The same events are summarized on stderr - the first
one immediately, then at most once every ten seconds - so packet loss is visible
in the console without an LSL consumer attached.

Estimating the change across a gap does not beat holding the level. This was
tested on the oddball recordings by deleting blocks of deltas from loss-free
stretches and measuring the offset each method leaves behind. RMS, µV, 29
channels:

| gap (samples)                                   | 1   | 8     | 16    |
|-------------------------------------------------|-----|-------|-------|
| hold (the reader)                               | 7.9 | 17.7  | 26.6  |
| pre-gap linear trend, 16 samples                | 8.1 | 20.9  | 33.9  |
| joint quadratic, 16 samples each side           | 11.6| 20.5  | 31.0  |
| minimum-curvature bridge, second differences    | 7.3 | 49.3  | 93.1  |
| best linear estimate (kriging, both sides)      | 1.1 | 15.8  | 24.4  |

Almost every real loss is 8 samples, and 62.5 ms of EEG is too long to
extrapolate. Even the best linear estimate gains only ~10% there. Only
1-sample losses, which are rare, would benefit.

### Repeated reports

The Flex dongle sometimes delivers the previous report a second time. There
are two such copies in `data/flex2.pcap`, and both are identical in all 32
bytes: counter, metadata bytes 1 and 30–31, and payload. A 60 s bench
recording through the Flex's own dongle had 8. In every case the counter and
all 32 channel deltas matched the report before it.

The reader drops a report that is byte-identical to the previous one. Publishing
it would add a fake sample and, because the payload is deltas, apply the same
delta to every channel a second time. Every later sample would be offset by up
to one full slew step (32.64 µV). In the `flex2.pcap`
replay, dropping the two repeats changes the next sample by up to 33 µV. A dropped
repeat counts as one discontinuity with nothing missing. Its `GAP_FLAG` is set
on the next published sample, and `--reset-on-gap` ignores it.

Repeats must be matched on the whole report, not the payload alone. Real
consecutive reports can have identical payloads. In `flex2.pcap`, reports with
counters 17 and 18 both have every channel at the maximum slew (+63 or −64), as
floating or saturated wires produce. Their counters and metadata bytes differ,
so the whole-report comparison keeps both.

On Flex, a repeat also delays every later report by one sample period. See
"Timestamps and lost samples" below.

### Timestamps and lost samples

**Why arrival time is wrong.** Timestamping each sample when its report
arrives, as the reader used to, puts it tens of milliseconds late and by a
varying amount. The Flex dongle sends reports on its own steady schedule. When
a report from the headset is late, the dongle resends the previous one, and
every later report then arrives one sample period behind. Periodically the
dongle catches up by discarding a block of 8 samples, often with almost no gap
in arrival time.

A 4-minute bench recording with deliberate obstructions shows this. With
the headset's rate held fixed, every change in delay was a whole number of
sample periods, within 0.14 of a period:

* each repeat added exactly one period;
* the losses were almost all exactly 8 or 16 samples;
* several 8-sample losses brought the delay straight back to its floor.

In the first 78 s, arrival times lagged the headset's sample clock by a median
of 23 ms (99th percentile 56 ms, maximum 69 ms). On the PC clock, the headset
rate was 127.987 Hz (−103 ppm), consistent to ±4 ppm across three separate
stretches. The metadata bytes carry no usable clock either. Bytes 30–31
cycle through 8 status slots, one per sample in each radio block. One slot
counts steadily, but it restarts after a dropout, so it cannot measure one.

**Counter clock.** The reader therefore timestamps each sample by its place in
the headset's sample sequence (`CounterClock` in `emotiv_base.py`):

* Because the delay moves in whole periods, arrivals keep the same phase within
  a period. A phase-locked loop on the arrival error, wrapped into one period,
  tracks the headset's period and phase. The phase estimate uses every sample,
  not just the rare minimum-delay ones.
* The lowest whole-period delay seen is taken as zero backlog. Until the
  dongle's first catch-up after starting, timestamps can be late by the backlog
  the session began with (15–31 ms in the two recordings).
* After a dropout of more than half a second, the dongle first dumps queued old
  reports about 1 ms apart. The clock holds samples for 1 s, takes the phase and
  floor from those arrivals, and applies them backwards. Publishing is delayed
  by that second once per long dropout.
* Timestamps always increase.

The bench recordings were replayed through the reader, rebuilt from their
counters and deltas with their original arrival times. Against an offline fit
over the whole recording, timestamps agreed to within ±0.1 ms (98% of samples)
once the delay floor had been seen, including after all four long dropouts.
The old arrival timestamps were off by a median of 31 ms.

**Filling lost samples.** Like Cortex's `INTERPOLATED` column, the reader
publishes a stand-in sample for every lost report and flags it with
`FILLED = 1`. Sample number then matches headset time, so tools that ignore
timestamps and assume a fixed rate stay aligned. A lost report's delta is
unknown, so a filled sample applies a zero delta: every channel holds its level.
The report after the loss carries `GAP_FLAG`
and `MISSING_REPORTS`, then applies its own delta. Losses longer than 60 s
(`FILL_LIMIT_SECONDS`) are not filled.

**Long dropouts.** The 7-bit counter cannot see whole cycles lost in a dropout.
The diagnostics used to undercount them: the 4-minute recording reported 581
missing reports instead of about 4,900. For a gap longer than half a cycle,
whole cycles are now added from arrival time, and `RESET_FLAG` marks the count
as an estimate. The estimate cannot be exact. For the four long dropouts in the
recording, the counter and arrival time disagreed by 4–44 samples. Either the
counter does not advance steadily through a dropout, or the dongle comes back
with a different delay. Reject a window around every `RESET_FLAG` in analysis.

**XDF loading.** Both Flex streams declare
`<synchronization><can_drop_samples>true</can_drop_samples></synchronization>`.
pyxdf (1.17+) then keeps their timestamps rather than refitting them over sample
number. Other loaders that do not honour the flag still see evenly
spaced, gap-free samples, because lost samples are filled. The EPOC X streams
declare the same flag.

### Replay check

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
