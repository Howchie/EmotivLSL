# emotiv-lsl

LSL server for Emotiv EPOC X  
Original code taken from [CyKit](https://github.com/CymatiCorp/CyKit)

### Dependencies

On Windows, install Python 3.11 or newer, then double-click `run.bat`. The
launcher installs Pipenv (if needed), creates the environment, and installs the
exact versions from `Pipfile.lock` before starting the LSL streams. The first
run requires an internet connection; later runs reuse the environment.

For manual setup or non-Windows use:

```
pip install pipenv
python -m pipenv sync --dev
```

### Usage
Disable the motion data in Emotiv app settings  
Connect dongle, turn on the headset, wait for the light from two indicators

On Windows, `run.bat` (or the explicit `run_epochX.bat`) launches the EPOC X
EEG and Cortex LSL streams with the repository's configured developer
credentials.  `run_flex.bat` launches the original EPOC Flex 1.0 path.
`run_epochX_legacy.bat` is the same EPOC X launcher pinned to the pre-0x740
decryption path; it is only needed if automatic firmware detection picks
wrong.

```
# frist terminal
python -m pipenv run python main.py
```

Then you can use any lsl client, for example [bskl](https://github.com/bsl-tools/bsl)

```
bsl_stream_viewer
```

![Alt text](images/bsl_stream_viewer.png)

Or use examples/read_data.py to get raw data

```
# second terminal
python -m pipenv run python examples/read_data.py
```

### Cortex quality streams to LSL

If you have a working Cortex app key, this repo can also bridge the Cortex `dev`
(contact quality) and `eq` (EEG quality) streams into LSL without changing the
raw HID EEG path.

Set your Cortex credentials:

```bash
export EMOTIV_CLIENT_ID=your_client_id
export EMOTIV_CLIENT_SECRET=your_client_secret
```

Then run:

```bash
python -m pipenv run python main_cortex.py --print-samples
```

This opens two additional LSL outlets when Cortex allows them:

* `Epoc X Contact Quality`
* `Epoc X EEG Quality`

You can also enable additional non-motion Cortex streams such as:

* `pow` -> `Epoc X Band Power`
* `met` -> `Epoc X Performance Metrics`
* `com` -> `Epoc X Mental Commands`
* `fac` -> `Epoc X Facial Expressions`

Their Cortex sample rates differ:

* `dev`: 2 Hz
* `eq`: 2 Hz
* `pow`: 8 Hz
* `met`: variable, typically 2 Hz with the right scope/session state, otherwise as low as 0.1 Hz
* `com`: 8 Hz
* `fac`: 32 Hz

The bridge now republishes Cortex samples to LSL using the original Cortex sample
timestamps converted into the local LSL clock domain, so cross-stream alignment
does not rely on arrival timing alone.

Example:

```bash
python -m pipenv run python main_cortex.py --streams dev eq pow met com fac
```

### Firmware 0x740 raw EEG support

EPOC X firmware `0x740` uses a different HID encryption protocol from older
firmware. The direct-HID reader now queries the EEG collection's feature report,
derives the firmware-specific AES-256 key, and publishes the raw EEG stream
without a Cortex EEG license. A licensed Cortex application remains an
alternative:

```bash
python -m pipenv run python main_cortex.py --streams eeg
```

This creates an `Epoc X Cortex EEG` LSL outlet. Its columns are the `eeg` columns
reported by Cortex (`COUNTER`, `INTERPOLATED`, the 14 EPOC X sensors, `RAW_CQ`,
and `MARKER_HARDWARE`); the `MARKERS` object column is omitted because an LSL
`float32` outlet cannot carry JSON objects. The outlet rate is taken from the
headset's Cortex setting (normally 256 Hz for EPOC X).

The Cortex `eeg` stream requires a paid license with the `eeg` scope and an
activated session. The existing default `dev eq` bridge remains unactivated and
continues to work for quality-only access. EmotivPRO's integrated LSL EEG outlet
is another supported fallback when EmotivPRO is licensed.

#### Both EPOC X firmware generations from one launcher

No firmware-specific launcher or headset update is needed. On every connection
the reader builds both keys it can - the pre-0x740 serial-derived AES-128 key
and, when the feature report answers, the 0x740 AES-256 key - and confirms the
choice against the decrypted packet counter, which only advances by one per
report under the correct key. A headset whose feature report is missing,
silent, or misreported therefore still streams.

Interface discovery is also firmware-independent: Emotiv interfaces are matched
on the manufacturer string *or* the receiver's vendor id (Windows does not
always report the former), the most likely EEG collection is probed first, and
an interface that cannot be opened or stays silent no longer aborts the run.

Override the automatic choice only if it picks wrong:

```bash
python -m pipenv run python main.py --firmware legacy   # force the pre-0x740 key
python -m pipenv run python main.py --firmware 0740     # force the feature-report key
```

On Windows, `run_epochX_legacy.bat` is the same launcher as `run_epochX.bat`
with `--firmware legacy` applied. If a headset is not recognized at all, print
what the operating system actually enumerates:

```bash
python -m pipenv run python main.py --list-hid
```


### Inspect a firmware-740 HID feature report

The direct HID decoder can query the feature report after opening the EEG
collection; the steady-state 32-byte EEG dump in issue #17 does not contain the
firmware seed by itself. To inspect the exchange, insert the dongle, leave the
headset powered off, and run the recorder before powering on the headset. It is
fine to leave EMOTIV Launcher running if the Cortex service needs it; the
recorder talks to HID directly and does not use Cortex login.

```bash
python -m pipenv run python examples/capture_hid_startup.py --seconds 60 >hid_capture.log 2>&1
```

Start the recorder first, then power on the headset and leave it running for a
few seconds. It records both Emotiv HID interfaces and preserves the raw report
length and bytes in `data/epocx_startup_hid.csv`; serial numbers can be redacted
afterward, but do not alter the report bytes. A USBPcap trace is needed only if
the feature report itself must be inspected; the launcher now performs the
feature query automatically.

Because the Cortex service may initiate the headset session, keep EMOTIV
Launcher running for this capture and allow it to notice/connect the headset
after the recorder has started. The hidapi CSV may contain only steady-state
`usage=2` rows; the feature report is a separate control transfer.

The recorder writes diagnostics to stderr, so redirect both streams (`>file
2>&1`) if you want the console output saved. If both interfaces report
`Capturing` but the CSV still contains only `usage=2` input reports, that is
normal: the feature query is a control transfer and is not emitted as an input
row. A USBPcap/Wireshark capture of the dongle records both directions if
further protocol inspection is needed.

The complete reverse-engineered design, including the firmware gate,
feature-report seed, SHA-256/AES-256 derivation, and downgrade findings, is
preserved in
[`docs/firmware_0740_hid_path.md`](docs/firmware_0740_hid_path.md).

The capture script is not part of the eventual acquisition workflow. The
decoder collects and caches the feature-report key automatically on each
headset connection while the Cortex service remains running for the quality
streams; no Cortex `eeg` scope is involved.

Notes:

* The HID recorder does not require a Cortex app key, an EEG license, or an
  EmotivPRO subscription.
* Do not start a second copy of this repository's launcher or EmotivPRO's EEG
  stream during capture. EMOTIV Launcher itself may remain running for the
  Cortex service.
* For a complete USBPcap startup trace, turn the headset off after starting the
  capture and power it on again. The production decoder can query the feature
  report even when the headset is already connected.

The separate Cortex bridge uses `wss://localhost:6868`; quality-only runs
(`dev`, `eq`, etc.) use an unactivated `open` session, while a run that includes
`eeg` requests an `active` session and consumes the appropriate EEG license
quota.

### Original EPOC Flex (Flex 1.0)

The original Flex 1.0 controller uses a different HID protocol from both EPOC X
and Flex 2.0. Its direct reader is kept separate while the protocol is being
validated:

```bash
python -m pipenv run python main_flex.py
```

The reader selects the dongle's `EEG Signals` collection, decrypts its 32-byte
AES-128 reports, and reconstructs the 32 compressed EEG channels at 128 Hz. It
does not require a Cortex EEG license and can run alongside EMOTIV Launcher and
the quality-only Cortex bridge. Flex 1.0 sensor placement is configurable. The
default [`epoch_flex_electrodes.json`](epoch_flex_electrodes.json) is a direct
Emotiv Launcher configuration export; edit that file whenever the plugs are
rearranged. The reader preserves the fixed wire order but publishes the
configured locations as the LSL channel labels, while retaining the wire name
in each channel's `wire` metadata field. `main_all_flex.py` also passes the
same mapping object to Cortex when it connects the Flex headset; Flex requires
that object for a discovered-headset connection. A different file can be
supplied with `--mapping PATH`. The full reverse-engineered path and current
caveats are in
[`docs/flex_1_hid_path.md`](docs/flex_1_hid_path.md).

The direct reader also publishes an always-on `Epoc Flex 1.0 Packet Diagnostics`
LSL stream beside the EEG outlet. It contains the 7-bit packet counter,
expected counter, per-sample gap/reset flags, and cumulative missing-report
counts. The Flex ADC accumulator is carried across gaps by default; use
`--reset-on-gap` only to reproduce the legacy midpoint-reset behavior.

To launch Flex EEG and the unlicensed Cortex quality streams together:

```bash
python -m pipenv run python main_all_flex.py \
  --client-id YOUR_ID --client-secret YOUR_SECRET --remove-dc
```

On Windows, double-click `run_flex.bat`. It uses `dev` and `eq` by default and
publishes them with the `Epoc Flex 1.0` prefix so they do not collide with EPOC
X quality outlets (`Epoc Flex 1.0 Contact Quality` and `Epoc Flex 1.0 EEG
Quality`). `main_all_epochX.py`, `main_epochX.py`, and
`run_epochX.bat` are explicit EPOC X entry points; the unsuffixed EPOC X
entry points remain as compatibility aliases.

If more than one Emotiv receiver is connected, select the Flex dongle by its
HID serial:

```bash
python -m pipenv run python main_flex.py --serial UD... --remove-dc
```

The existing `examples/view_contact_quality.py` remains the 14-sensor EPOC X
viewer. `main_all_flex.py` launches the Flex viewer automatically after the
streams start; use `--no-viewer` for a headless LSL-only run. To view the
current EPOC X contact quality stream as a live head map:

```bash
python -m pipenv run python examples/view_contact_quality.py
```

To calibrate the complete Flex reference image (all available 10-20 locations,
not only the locations active in the current montage):

```bash
python -m pipenv run python examples/calibrate_flex_head_image.py
```

The points are saved to `flex_head_image_coords.json` and can be regenerated if
the image or sensor placements change.

The Flex quality viewer downsamples the reference image to a panel width of at
most 420 pixels so both quality panels fit on a 1080p display.  The calibration
values are normalized (0–1), so the overlay locations stay aligned automatically;
the calibration JSON does not need to be edited when the viewer scales the image.

If a computer cannot render the PNG correctly, pass `--no-background` to
`main_all_flex.py`; the viewer will show the full calibrated coordinate layout
and active quality markers on a plain canvas.

This viewer subscribes to both `Epoc X Contact Quality` and `Epoc X EEG Quality`
and renders two live head maps side by side. Each sensor is colored from
black/red through green based on the `0..4` quality scale, while each panel also
shows its overall percentage.

The viewer uses normalized coordinates stored in `head_image_coords.json`, so the
head image only needs to be calibrated once. To create or update that file:

```bash
python -m pipenv run python examples/calibrate_head_image.py
```

Click the sensor centers in this order:

* `AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, `AF4`
* `OVERALL`

To launch raw EEG, the Cortex quality streams, and the dual quality viewer from a
single command:

```bash
python -m pipenv run python main_all.py --client-id YOUR_ID --client-secret YOUR_SECRET
```

To include the additional optional Cortex streams in the unified launcher:

```bash
python -m pipenv run python main_all.py --client-id YOUR_ID --client-secret YOUR_SECRET --streams dev eq pow met com fac
```

This starts:

* `Epoc X` EEG over the raw HID path
* `Epoc X Packet Diagnostics` (counter/gap metadata, timestamped with EEG)
* `Epoc X Contact Quality` from Cortex `dev`
* `Epoc X EEG Quality` from Cortex `eq`
* the live dual-panel viewer

### Config

Change device sampling rate in config.py and emotiv app

### Probe hidden quality fields

This repo publishes the 14 EEG channels and an always-on `Epoc X Packet
Diagnostics` stream. Counter discontinuities are also summarized on the console
(the first one immediately, then at most once every ten seconds) so packet loss
is visible without an LSL consumer attached. The diagnostics stream is timestamped alongside the EEG
samples and includes the decrypted 8-bit packet counter, expected counter,
per-sample gap/reset flags, and cumulative loss/reset counts. A gap flag marks
the first received sample after a counter discontinuity; it does not alter the
EEG values. Use those fields to mark or reject short acquisition windows in
analysis.

The decrypted packet also contains four non-EEG bytes that are not exposed by
default. To inspect whether they carry useful quality information without
breaking existing EEG consumers:

```bash
python -m pipenv run python main.py --emit-debug --log-decrypted data/decrypted_packets.csv
```

This keeps the existing `EEG` outlet and adds a second stream named `Epoc X Debug`
with four channels:

* `COUNTER`
* `DATA_MODE`
* `BYTE16`
* `BYTE17`

Suggested test protocol:

1. Start the EEG and debug outlets.
2. Record both in LabRecorder.
3. Put the headset on with a good fit, then intentionally worsen one sensor at a time.
4. Check whether `BYTE16` or `BYTE17` changes systematically with global fit.
5. If you have temporary EmotivPro access on one machine, compare the debug stream
   against its `Contact-Quality` output once to correlate any hidden fields.

If these bytes do not track contact quality, the next step is to treat true contact
quality as unavailable on the HID path and implement a separate signal-quality proxy
instead of calling it contact quality.

The Flex launcher similarly publishes `Epoc Flex 1.0 Packet Diagnostics` beside
the 32-channel EEG stream. Flex uses a 7-bit wrapping counter and the same
diagnostic channel names. Its ADC accumulator is carried across packet gaps by
default; pass `--reset-on-gap` only when reproducing the legacy midpoint-reset
behavior.

### Examples

Get raw data:

```
python main.py & # start lsl server
python examples/read_data.py # get raw data
[4179.35888671875, 4320.5126953125, 4263.84619140625, 4311.53857421875, 4393.58984375, 4347.56396484375, 4371.41015625, 4549.4873046875, 4511.9228515625, 4434.1025390625, 4378.46142578125, 5053.33349609375, 4283.33349609375, 4228.46142578125] 104573.455064594
[4163.33349609375, 4318.0771484375, 4258.7177734375, 4310.384765625, 4396.41015625, 4350.384765625, 4374.615234375, 4550.76904296875, 4508.0771484375, 4429.4873046875, 4374.4873046875, 5058.205078125, 4274.4873046875, 4222.94873046875] 104573.457074205
[4164.615234375, 4316.02587890625, 4255.384765625, 4312.3076171875, 4398.0771484375, 4350.12841796875, 4376.15380859375, 4552.94873046875, 4513.97412109375, 4430.8974609375, 4375.384765625, 5063.7177734375, 4275.384765625, 4225.0] 104573.464060118
[4177.94873046875, 4321.66650390625, 4261.794921875, 4313.46142578125, 4397.1796875, 4347.05126953125, 4373.58984375, 4551.66650390625, 4521.2822265625, 4436.794921875, 4378.7177734375, 5060.76904296875, 4283.7177734375, 4232.8203125] 104573.472085895
```

Write raw data via mne to .fif:

```
python main.py & # start lsl server
python examples/read_and_export_mne.py # write raw data to fif file
Ready.
Writing emotiv-lsl/data_2023-09-20 18:36:10.775860_raw.fif
Closing emotiv-lsl/data_2023-09-20 18:36:10.775860_raw.fif
```
