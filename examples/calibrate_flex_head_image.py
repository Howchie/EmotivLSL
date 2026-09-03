"""Calibrate every location on the full Flex 10-20 reference image."""

from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# When this file is run as ``python examples/...py``, Python puts only the
# examples directory on sys.path.  Add the repository root so the package
# import works from both the repository root and another working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotiv_lsl.flex_layout import FLEX_10_20_LOCATIONS


DEFAULT_IMAGE_PATH = ROOT / "images" / "10-20.png"
DEFAULT_CALIBRATION_PATH = ROOT / "flex_head_image_coords.json"
CLICK_SEQUENCE = FLEX_10_20_LOCATIONS + ("OVERALL",)


class FlexHeadImageCalibrator:
    def __init__(self, image_path: Path, calibration_path: Path) -> None:
        self.image_path = image_path
        self.calibration_path = calibration_path
        self.root = tk.Tk()
        self.root.title("Calibrate Flex 10-20 Image")

        self.image = tk.PhotoImage(master=self.root, file=str(self.image_path))
        self.width = self.image.width()
        self.height = self.image.height()

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        viewport_width = min(self.width, max(640, screen_width - 120))
        viewport_height = min(self.height, max(480, screen_height - 180))

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=viewport_width,
            height=viewport_height,
            highlightthickness=0,
            xscrollincrement=20,
            yscrollincrement=20,
            scrollregion=(0, 0, self.width, self.height),
        )
        x_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas.create_image(0, 0, image=self.image, anchor="nw")
        self.canvas.bind("<Button-1>", self.on_click)

        self.status_var = tk.StringVar()
        self.status = tk.Label(self.root, textvariable=self.status_var)
        self.status.pack(pady=8)

        self.index = 0
        self.points: dict[str, list[float]] = {}
        self.update_status()

    def update_status(self) -> None:
        target = CLICK_SEQUENCE[self.index]
        self.status_var.set(
            f"Click {target} ({self.index + 1}/{len(CLICK_SEQUENCE)})"
        )

    def on_click(self, event) -> None:
        key = CLICK_SEQUENCE[self.index]
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        norm_x = canvas_x / self.width
        norm_y = canvas_y / self.height

        self.canvas.create_oval(
            canvas_x - 4,
            canvas_y - 4,
            canvas_x + 4,
            canvas_y + 4,
            fill="red",
            outline="",
        )
        self.canvas.create_text(
            canvas_x + 8,
            canvas_y - 8,
            text=key,
            anchor="w",
            fill="red",
            font=("Helvetica", 9, "bold"),
        )

        self.points[key] = [norm_x, norm_y]
        self.index += 1
        if self.index == len(CLICK_SEQUENCE):
            payload = {
                "image": self.image_path.name,
                "image_width": self.width,
                "image_height": self.height,
                "locations": {
                    location: self.points[location]
                    for location in FLEX_10_20_LOCATIONS
                },
                # Keep the same key used by the existing EPOC X viewer so
                # future shared viewer code can consume either calibration.
                "sensors": {
                    location: self.points[location]
                    for location in FLEX_10_20_LOCATIONS
                },
                "overall_anchor": self.points["OVERALL"],
            }
            self.calibration_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            self.status_var.set(f"Saved calibration to {self.calibration_path}")
            self.canvas.unbind("<Button-1>")
            return

        self.update_status()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Flex calibration image not found: {args.image}")
    FlexHeadImageCalibrator(args.image, args.output).run()


if __name__ == "__main__":
    main()
