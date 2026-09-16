"""Launch the original EPOC Flex 1.0 EEG and Cortex quality streams."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

from emotiv_lsl.cortex_bridge import BridgeConfig, run_bridge
from emotiv_lsl.emotiv_flex import (
    DEFAULT_DC_RESTORE_HZ,
    DEFAULT_DISPLAY_HZ,
    EmotivFlex,
)
from emotiv_lsl.flex_montage import load_flex_montage


DEFAULT_MAPPING_PATH = Path(__file__).resolve().with_name("epoch_flex_electrodes.json")
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().with_name("flex_head_image_coords.json")
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "images" / "10-20.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default=os.getenv("EMOTIV_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("EMOTIV_CLIENT_SECRET"))
    parser.add_argument("--license", default=os.getenv("EMOTIV_LICENSE_ID"))
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
    parser.add_argument(
        "--stream-prefix",
        default="Epoc Flex 1.0",
        help="prefix for the published Cortex quality LSL stream names",
    )
    parser.add_argument(
        "--remove-dc",
        action="store_true",
        help="subtract the 14-bit ADC midpoint before converting EEG to microvolts",
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING_PATH),
        metavar="PATH",
        help=f"Flex wire-to-location JSON mapping (default: {DEFAULT_MAPPING_PATH.name})",
    )
    parser.add_argument(
        "--serial",
        help="select a specific Flex dongle serial when multiple receivers are connected",
    )
    parser.add_argument(
        "--dc-restore-hz",
        type=float,
        default=DEFAULT_DC_RESTORE_HZ,
        metavar="HZ",
        help=(
            "high-pass corner of an optional leak on the ADC accumulator "
            f"(default: {DEFAULT_DC_RESTORE_HZ}, a pure accumulator; the analogue "
            "chain already high-passes at 0.16 Hz, so a leak adds a second high-pass)"
        ),
    )
    parser.add_argument(
        "--reset-on-gap",
        action="store_true",
        help=(
            "reset Flex ADC state to midpoint after a packet gap (legacy behavior; "
            "normally leave disabled)"
        ),
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        metavar="PATH",
        help=f"calibrated Flex quality-image coordinates (default: {DEFAULT_CALIBRATION_PATH.name})",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        metavar="PATH",
        help=f"Flex quality background image (default: {DEFAULT_IMAGE_PATH.name})",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="launch the quality map with calibrated markers but without the PNG background",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run the EEG and Cortex streams without opening the quality-map window",
    )
    parser.add_argument(
        "--display-hz",
        type=float,
        default=DEFAULT_DISPLAY_HZ,
        metavar="HZ",
        help=(
            "high-pass corner of the viewer-only 'Epoc Flex 1.0 Display' stream "
            f"(default: {DEFAULT_DISPLAY_HZ}; 0 publishes no display stream).  The "
            "recorded stream is a bare integral that wanders by millivolts, so a "
            "viewer needs a high-passed copy to show anything.  Record and analyse "
            "'Epoc Flex 1.0', never the display stream"
        ),
    )
    args = parser.parse_args()
    if not args.client_id or not args.client_secret:
        parser.error(
            "client credentials are required via --client-id/--client-secret "
            "or EMOTIV_CLIENT_ID/EMOTIV_CLIENT_SECRET"
        )
    return args


def start_eeg(
    args: argparse.Namespace,
    labels: tuple[str, ...],
    montage_name: str,
    references: dict[str, str],
) -> None:
    EmotivFlex(
        remove_dc=args.remove_dc,
        channel_labels=labels,
        montage_name=montage_name,
        references=references,
        serial_number=args.serial,
        reset_on_gap=args.reset_on_gap,
        dc_restore_hz=args.dc_restore_hz,
        display_hz=args.display_hz,
    ).main_loop()


def start_quality_bridge(args: argparse.Namespace, headset_mappings: dict[str, str]) -> None:
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
            stream_prefix=args.stream_prefix,
            headset_mappings=headset_mappings,
        )
    )


def run_component(
    name: str,
    target,
    stop_event: threading.Event,
) -> None:
    try:
        target()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"{name} failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        stop_event.set()


def wait_for_components(
    eeg_thread: threading.Thread,
    quality_thread: threading.Thread,
    stop_event: threading.Event,
) -> None:
    try:
        while not stop_event.wait(0.5):
            if not eeg_thread.is_alive() and not quality_thread.is_alive():
                break
    except KeyboardInterrupt:
        pass


def main() -> None:
    args = parse_args()
    montage = load_flex_montage(args.mapping)
    print(
        f"Using Flex montage {montage.name!r} from {montage.path}",
        file=sys.stderr,
        flush=True,
    )

    stop_event = threading.Event()
    eeg_thread = threading.Thread(
        target=run_component,
        args=(
            "Flex EEG",
            lambda: start_eeg(
                args,
                montage.labels,
                montage.name,
                montage.references,
            ),
            stop_event,
        ),
        daemon=True,
        name="flex-hid-eeg",
    )
    quality_thread = threading.Thread(
        target=run_component,
        args=(
            "Cortex quality bridge",
            lambda: start_quality_bridge(
                args,
                {**montage.references, **montage.mapping},
            ),
            stop_event,
        ),
        daemon=True,
        name="cortex-quality",
    )
    eeg_thread.start()
    quality_thread.start()

    try:
        if args.no_viewer:
            wait_for_components(eeg_thread, quality_thread, stop_event)
        else:
            from examples.view_flex_quality import FlexQualityViewer, load_calibration

            try:
                viewer = FlexQualityViewer(
                    montage=montage,
                    calibration=load_calibration(Path(args.calibration)),
                    image_path=Path(args.image),
                    stream_prefix=args.stream_prefix,
                    no_background=args.no_background,
                )
            except Exception as exc:
                print(
                    f"Flex quality viewer unavailable ({exc}); continuing with LSL streams only.",
                    file=sys.stderr,
                    flush=True,
                )
                wait_for_components(eeg_thread, quality_thread, stop_event)
            else:
                viewer.run()
    finally:
        stop_event.set()
        print("Flex EEG and Cortex quality streams stopped.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
