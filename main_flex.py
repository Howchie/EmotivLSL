"""Launch the direct EPOC Flex 1.0 HID stream."""

import argparse
from pathlib import Path

from emotiv_lsl.emotiv_flex import DEFAULT_DC_RESTORE_HZ, EmotivFlex
from emotiv_lsl.flex_montage import load_flex_montage


DEFAULT_MAPPING_PATH = Path(__file__).resolve().with_name("epoch_flex_electrodes.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove-dc",
        action="store_true",
        help="subtract the 14-bit ADC midpoint before converting to microvolts",
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_MAPPING_PATH),
        metavar="PATH",
        help=(
            "JSON mapping from Flex wire labels to electrode locations "
            f"(default: {DEFAULT_MAPPING_PATH.name})"
        ),
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
            "high-pass corner of the ADC accumulator's DC restore "
            f"(default: {DEFAULT_DC_RESTORE_HZ}, the Flex passband corner; "
            "0 disables it and restores the unbounded pure accumulator)"
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
    args = parser.parse_args()
    montage = load_flex_montage(args.mapping)
    print(
        f"Using Flex montage {montage.name!r} from {montage.path}",
        flush=True,
    )
    EmotivFlex(
        remove_dc=args.remove_dc,
        channel_labels=montage.labels,
        montage_name=montage.name,
        references=montage.references,
        serial_number=args.serial,
        reset_on_gap=args.reset_on_gap,
        dc_restore_hz=args.dc_restore_hz,
    ).main_loop()
