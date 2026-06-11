import argparse
import threading

from emotiv_lsl.cortex_bridge import BridgeConfig, run_bridge
from emotiv_lsl.emotiv_epoc_x import EmotivEpocX
from examples.view_contact_quality import DualQualityViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id")
    parser.add_argument("--client-secret")
    parser.add_argument("--license")
    parser.add_argument("--cortex-url", default="wss://localhost:6868")
    parser.add_argument("--headset-id")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=["dev", "eq"],
        default=["dev", "eq"],
        help="Cortex quality streams to bridge to LSL",
    )
    parser.add_argument("--verify-ssl", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--emit-debug", action="store_true")
    parser.add_argument("--log-decrypted", metavar="PATH")
    args = parser.parse_args()
    if not args.client_id or not args.client_secret:
        parser.error("client credentials are required via --client-id and --client-secret")
    return args


def start_eeg(args: argparse.Namespace) -> None:
    emotiv_epoc_x = EmotivEpocX(
        emit_debug=args.emit_debug,
        packet_log_path=args.log_decrypted,
    )
    emotiv_epoc_x.main_loop()


def start_quality_bridge(args: argparse.Namespace) -> None:
    run_bridge(
        BridgeConfig(
            client_id=args.client_id,
            client_secret=args.client_secret,
            license=args.license,
            cortex_url=args.cortex_url,
            headset_id=args.headset_id,
            streams=tuple(args.streams),
            verify_ssl=args.verify_ssl,
            print_samples=args.print_samples,
        )
    )


def main() -> None:
    args = parse_args()

    eeg_thread = threading.Thread(target=start_eeg, args=(args,), daemon=True, name="hid-eeg")
    eeg_thread.start()

    bridge_thread = threading.Thread(target=start_quality_bridge, args=(args,), daemon=True, name="cortex-quality")
    bridge_thread.start()

    DualQualityViewer().run()


if __name__ == "__main__":
    main()
