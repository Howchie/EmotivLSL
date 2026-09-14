from datetime import datetime

import numpy as np
from mne import Info, create_info
from mne.io.array import RawArray
from pylsl import StreamInlet, resolve_byprop


def get_info(sfreq: float) -> Info:
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
                'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    info = create_info(
        sfreq=sfreq,
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names)
    )

    return info


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    # Take the rate from the stream: EPOC X runs at 128 or 256 Hz and the
    # reader declares whichever it measured.
    sfreq = inlet.info().nominal_srate()
    print(f"stream {inlet.info().name()!r} at {sfreq:g} Hz")

    buffer = []
    while True:
        if len(buffer) >= int(sfreq * 5):  # wait 5 seconds
            break

        sample, _ = inlet.pull_sample()
        sample = [el / 1000000 for el in sample]  # convert to microvolts

        buffer.append(sample)

    info = get_info(sfreq)
    raw = RawArray(np.array(buffer).T, info)

    raw.save("data_{}_raw.fif".format(datetime.now()))


if __name__ == '__main__':
    main()
