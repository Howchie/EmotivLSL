import argparse
import sys
import threading

from emotiv_lsl.cortex_bridge import BridgeConfig, run_bridge
from emotiv_lsl.emotiv_base import print_all_hid_interfaces
from emotiv_lsl.emotiv_epoc_x import EmotivEpocX
from examples.view_contact_quality import DualQualityViewer

EEG_WATCHDOG_MS = 500


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
        choices=["dev", "eq", "pow", "met", "com", "fac"],
        default=["dev", "eq"],
        help="Cortex quality streams to bridge to LSL",
    )
    parser.add_argument("--verify-ssl", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--emit-debug", action="store_true")
    parser.add_argument("--log-decrypted", metavar="PATH")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "declare this EPOC X sample rate instead of measuring it at star"
            "tup (the headset runs at 128 or 256 Hz and the HID report does not say which)"
        ),
    )
    parser.add_argument(
        "--firmware",
        choices=EmotivEpocX.FIRMWARE_MODES,
        default="auto",
        help=(
            "HID decryption path: auto detects and confirms it from the headset "
            "(default), legacy runs the unmodified pre-0x740 reader, 0740 "
            "forces the feature-report key"
        ),
    )
    parser.add_argument(
        "--list-hid",
        action="store_true",
        help="print every HID interface this machine reports, then exit",
    )
    args = parser.parse_args()
    if args.list_hid:
        return args
    if not args.client_id or not args.client_secret:
        parser.error("client credentials are required via --client-id and --client-secret")
    return args


def start_eeg(args: argparse.Namespace) -> None:
    emotiv_epoc_x = EmotivEpocX(
        emit_debug=args.emit_debug,
        packet_log_path=args.log_decrypted,
        firmware_mode=args.firmware,
        sample_rate=args.sample_rate,
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

    if args.list_hid:
        print_all_hid_interfaces()
        return

    eeg_thread = threading.Thread(target=start_eeg, args=(args,), daemon=True, name="hid-eeg")
    eeg_thread.start()

    bridge_thread = threading.Thread(target=start_quality_bridge, args=(args,), daemon=True, name="cortex-quality")
    bridge_thread.start()

    viewer = DualQualityViewer()

    # If the EEG reader dies, close everything rather than leave the quality
    # window and Cortex streams running without EEG. A half-alive instance also
    # keeps its quality streams on the network, where a relaunched viewer can
    # latch onto them. The thread's traceback has already been printed.
    def close_if_eeg_stopped() -> None:
        if eeg_thread.is_alive():
            viewer.root.after(EEG_WATCHDOG_MS, close_if_eeg_stopped)
            return
        print("The EPOC X EEG reader stopped; closing.", file=sys.stderr, flush=True)
        viewer.root.destroy()

    viewer.root.after(EEG_WATCHDOG_MS, close_if_eeg_stopped)
    viewer.run()
    if not eeg_thread.is_alive():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
