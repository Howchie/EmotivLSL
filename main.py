import argparse

from emotiv_lsl.emotiv_base import print_all_hid_interfaces
from emotiv_lsl.emotiv_epoc_x import EmotivEpocX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--emit-debug',
        action='store_true',
        help='publish a second LSL stream with decrypted packet metadata',
    )
    parser.add_argument(
        '--log-decrypted',
        metavar='PATH',
        help='write decrypted 32-byte packets to CSV for reverse engineering',
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=None,
        metavar='HZ',
        help=(
            'declare this EPOC X sample rate instead of measuring it at star'
            'tup (the headset runs at 128 or 256 Hz and the HID report does not say which)'
        ),
    )
    parser.add_argument(
        '--firmware',
        choices=EmotivEpocX.FIRMWARE_MODES,
        default='auto',
        help=(
            'HID decryption path: auto detects and confirms it from the headset '
            '(default), legacy forces the pre-0x740 serial-derived key, 0740 '
            'forces the feature-report key'
        ),
    )
    parser.add_argument(
        '--list-hid',
        action='store_true',
        help='print every HID interface this machine reports, then exit',
    )
    args = parser.parse_args()

    if args.list_hid:
        print_all_hid_interfaces()
        return

    emotiv_epoc_x = EmotivEpocX(
        emit_debug=args.emit_debug,
        packet_log_path=args.log_decrypted,
        firmware_mode=args.firmware,
        sample_rate=args.sample_rate,
    )
    emotiv_epoc_x.main_loop()


if __name__ == "__main__":
    main()
