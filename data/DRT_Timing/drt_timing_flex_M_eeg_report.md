# Flex DRT timing XDF exploratory report

File: `drt_timing_flex_M.xdf`  
Stream: `Epoc Flex 1.0`  
Recording duration: 36.48 s (4,670 EEG samples; 31 DRT trials)

## Acquisition integrity

The EEG stream is present at 127.99 Hz and the packet-counter clock is
linear. There are no filled samples, missing reports, or counter resets. Two
byte-identical repeated reports were dropped. The XDF therefore contains a
valid transport stream rather than an LSL/clock dropout.

## Signal integrity

The stream metadata reports `delta_zero=63`, `dc_restore_hz=0.0`, and 0.51
µV/LSB. Across the 32 channels, 86.1% of decoded sample-to-sample deltas are
at the allowed endpoints (-63 or +64 counts). The decoder range is otherwise
valid (-63..64), but this endpoint fraction is incompatible with ordinary EEG
and produces a large alternating/ramping waveform. No software ADC clamp is
being applied in the current reader.

The Flex Cortex quality streams in this recording report `overall=0` and most
sensor scores are 0. The likely `M` wires are `LM -> PO9` and `RM -> PO10` in
the saved montage; neither is a strong event channel (PO9 and PO10 have only
about 1.5 and 1.3 baseline-SD event-locked SNR, respectively).

## DRT response

The pooled all-channel waveform does not show the clean, high-SNR edge seen in
the EPOC X recording. The existing edge detector returns an onset near 85 ms,
but its 95% interval is approximately 65--128 ms; the offset interval is even
wider. This is not a usable Flex chain-latency estimate.

## Interpretation

The recording proves that LSL is receiving samples and that timestamps are
usable. It does not show a clean electrical pickup. The endpoint pattern is
also a strong decoder-variant warning: replaying the stored deltas with the
alternate seven-bit two's-complement interpretation reduces endpoint deltas
from 86.3% to 0.3% and gives a normal-sized reconstructed waveform. That does
not prove the alternate interpretation is correct, because the raw encrypted
HID packets are not in the XDF, but it means this should not be dismissed as
ordinary display scaling. The current offset-binary rule was validated against
one Flex 1.0 capture/dongle; this receiver may use a different encoding or the
recording process may have used an older reader.

It is not the current reader clipping its reconstructed ADC state. A bad
CMS/DRL/reference connection can still produce endpoint saturation, so verify
that in parallel; if the references really match the EPOC X desk setup, raw HID
capture and the dongle serial are the decisive next check.

Before another timing run, verify both CMS and DRL are physically connected to
the intended reference/return locations and that the quality map is non-black.
Also confirm the direct EEG stream is selected by exact name (`Epoc Flex 1.0`),
not the 2 Hz `Epoc Flex 1.0 EEG Quality` stream. If quality is good but the
delta histogram remains endpoint-dominated, capture the HID serial/raw reports
for decoder validation.
