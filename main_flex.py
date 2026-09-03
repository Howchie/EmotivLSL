"""Launch the direct EPOC Flex 1.0 HID stream."""

import argparse
from pathlib import Path

from emotiv_lsl.emotiv_flex import EmotivFlex
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
    ).main_loop()
