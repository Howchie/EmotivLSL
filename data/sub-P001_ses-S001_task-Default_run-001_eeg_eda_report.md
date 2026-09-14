# XDF acquisition QC report

**File:** `sub-P001_ses-S001_task-Default_run-001_eeg.xdf`  
**Format:** XDF (Lab Streaming Layer recording)  
**Recording header time:** 2026-09-14T11:35:14+0800  
**File size:** 513,523 bytes

## Streams

| Stream | Nominal rate | Samples | Duration / effective rate |
|---|---:|---:|---:|
| Epoc X | 128 Hz | 3,846 | 30.0003 s / 128.165 Hz |
| Epoc X Packet Diagnostics | 128 Hz | 3,846 | 30.0003 s / 128.165 Hz |
| Epoc X Contact Quality | 2 Hz | 60 | 29.4873 s / 2.001 Hz |
| Epoc X EEG Quality | 2 Hz | 59 | 28.9855 s / 2.001 Hz |
| Epoc X Band Power | 8 Hz | 241 | 29.9849 s / 8.004 Hz |

The EEG and diagnostics timestamps are uniform to floating-point precision and
have identical sample counts. The Cortex streams are separate lower-rate
streams, not samples in the EEG stream.

`pyxdf` reported and truncated one extra EEG-quality sample beyond that stream's
footer count (60 present versus footer count 59). This does not affect the EEG
or packet-diagnostics streams.

## Counter check

The diagnostics metadata declares `counter_modulus=256`, but the recorded
counter values occupy only 0–127. With modulus 256, each normal 127→0 wrap is
reported as 128 missing reports (30 such transitions in this file). Replaying
the same counter values modulo 128 gives 3 duplicate transitions and no skipped
reports. The cumulative missing value starts at 14,976 because startup samples
were counted before recording began; the recording itself adds 3,840 synthetic
“missing” reports from the 30 normal wraps.

## Conclusion

The recording contains a continuous 128 Hz EEG sequence. The large loss count
is a diagnostics-model error, not missing EEG data. The reader must select a
128-value counter modulus for a 128 Hz run and a 256-value modulus for a 256 Hz
run. Startup-buffer samples must also be excluded from the live arrival-rate
monitor.
