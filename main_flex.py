"""Launch the direct EPOC Flex 1.0 HID stream."""

import argparse

from emotiv_lsl.emotiv_flex import EmotivFlex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove-dc",
        action="store_true",
        help="subtract the 14-bit ADC midpoint before converting to microvolts",
    )
    args = parser.parse_args()
    EmotivFlex(remove_dc=args.remove_dc).main_loop()
