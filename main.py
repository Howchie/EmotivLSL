import argparse

from emotiv_lsl.emotiv_epoc_x import EmotivEpocX

if __name__ == "__main__":
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
    args = parser.parse_args()

    emotiv_epoc_x = EmotivEpocX(
        emit_debug=args.emit_debug,
        packet_log_path=args.log_decrypted,
    )
    emotiv_epoc_x.main_loop()
