"""Display Flex contact/EEG quality on the complete 10-20 reference image.

The viewer deliberately draws every calibrated location.  Only locations in
the current Flex montage receive an active quality marker.  If Tk cannot
decode the background PNG, it falls back to a plain canvas with the same
calibrated coordinates, so the quality display remains usable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from pylsl import StreamInlet, resolve_byprop


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotiv_lsl.flex_layout import FLEX_10_20_LOCATIONS
from emotiv_lsl.flex_montage import FlexMontage, load_flex_montage


DEFAULT_IMAGE_PATH = ROOT / "images" / "10-20.png"
DEFAULT_CALIBRATION_PATH = ROOT / "flex_head_image_coords.json"
DEFAULT_MAPPING_PATH = ROOT / "epoch_flex_electrodes.json"
DEFAULT_STREAM_PREFIX = "Epoc Flex 1.0"

MAX_PANEL_WIDTH = 520
PANEL_GAP = 24
WINDOW_PADDING = 16
HEADER_HEIGHT = 36
INACTIVE_RADIUS = 5
SENSOR_RADIUS = 11
RING_RADIUS = 15

QUALITY_COLORS = {
    0: "#111111",
    1: "#d84a3a",
    2: "#f09a3e",
    3: "#b8d94a",
    4: "#1ebf5b",
}


@dataclass
class PanelImage:
    image: tk.PhotoImage | None
    path: Path | None
    width: int
    height: int
    error: str | None = None


@dataclass
class SensorGlyph:
    ring_id: int
    oval_id: int
    text_id: int


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def score_to_color(score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return QUALITY_COLORS[0]
    rounded = max(0, min(4, int(round(score))))
    return QUALITY_COLORS[rounded]


def overall_to_color(overall: float | None) -> str:
    if overall is None or not math.isfinite(overall):
        return "#8a8a8a"
    if overall >= 80:
        return "#8ad448"
    if overall >= 60:
        return "#c7d94a"
    if overall >= 40:
        return "#f0b23e"
    return "#d84a3a"


def load_calibration(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Flex image calibration not found: {path}. "
            "Run examples/calibrate_flex_head_image.py first."
        ) from exc

    locations = document.get("locations") or document.get("sensors")
    if not isinstance(locations, dict):
        raise ValueError(f"Flex calibration {path} has no locations object")
    missing = [name for name in FLEX_10_20_LOCATIONS if name not in locations]
    if missing:
        raise ValueError(
            f"Flex calibration {path} is missing {len(missing)} location(s): "
            f"{', '.join(missing)}"
        )
    if "overall_anchor" not in document:
        raise ValueError(f"Flex calibration {path} has no overall_anchor")
    return document


def load_panel_image(
    root: tk.Tk,
    image_path: Path,
    calibration: dict,
    no_background: bool,
) -> PanelImage:
    width = int(calibration.get("image_width", 0))
    height = int(calibration.get("image_height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Flex calibration must contain positive image dimensions")
    if no_background:
        return PanelImage(None, None, width, height, "disabled by --no-background")

    try:
        image = tk.PhotoImage(master=root, file=str(image_path))
    except (tk.TclError, OSError) as exc:
        return PanelImage(None, None, width, height, f"could not load {image_path}: {exc}")

    # The coordinates are normalized, so a replacement image can have a
    # different size.  Use the actual image dimensions when it loads.
    return PanelImage(image, image_path, image.width(), image.height())


def channel_labels(info) -> list[str]:
    """Read the labels returned by Cortex in the order used by the LSL inlet."""
    labels: list[str] = []
    try:
        channel = info.desc().child("channels").child("channel")
        while channel is not None and not channel.empty():
            labels.append(channel.child_value("label"))
            channel = channel.next_sibling()
    except (AttributeError, RuntimeError):
        # Some pylsl builds expose the end of the sibling list as None rather
        # than an empty XMLElement. Keep labels already collected in either
        # case.
        pass
    return labels


class QualitySource:
    """Map one Cortex quality inlet's dynamic columns to Flex locations."""

    def __init__(self, inlet: StreamInlet, montage: FlexMontage, kind: str) -> None:
        self.inlet = inlet
        self.montage = montage
        self.kind = kind
        self.labels = channel_labels(inlet.info())
        self.indices = {
            normalize_label(label): index
            for index, label in enumerate(self.labels)
        }
        self.sensor_indices: dict[str, int] = {}
        for wire, location in montage.mapping.items():
            for candidate in (wire, location):
                index = self.indices.get(normalize_label(candidate))
                if index is not None:
                    self.sensor_indices[wire] = index
                    break

        if kind == "dev":
            self.overall_index = self._find_index("overall")
            self.battery_index = self._find_index("battery")
            self.signal_index = self._find_index("signal")
            self.battery_percent_index = self._find_index("batterypercent")
            self.rate_index = None
        else:
            self.overall_index = self._find_index("overall")
            self.battery_index = None
            self.signal_index = None
            self.battery_percent_index = self._find_index("batterypercent")
            self.rate_index = self._find_index("sampleratequality")

    def _find_index(self, label: str) -> int | None:
        return self.indices.get(normalize_label(label))

    def _value(self, sample: list[float], index: int | None) -> float | None:
        if index is None or index >= len(sample):
            return None
        try:
            value = float(sample[index])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def decode(self, sample: list[float]) -> tuple[dict[str, float], dict[str, float | None]]:
        scores: dict[str, float] = {}
        for wire, index in self.sensor_indices.items():
            value = self._value(sample, index)
            if value is not None:
                scores[wire] = value

        metrics: dict[str, float | None] = {
            "overall": self._value(sample, self.overall_index),
            "battery": self._value(sample, self.battery_index),
            "signal": self._value(sample, self.signal_index),
            "batteryPercent": self._value(sample, self.battery_percent_index),
            "sampleRateQuality": self._value(sample, self.rate_index),
        }
        return scores, metrics


class FlexQualityPanel:
    def __init__(
        self,
        canvas: tk.Canvas,
        title: str,
        panel_left: int,
        calibration: dict,
        montage: FlexMontage,
        panel_image: PanelImage,
    ) -> None:
        self.canvas = canvas
        self.title = title
        self.panel_left = panel_left
        self.panel_top = HEADER_HEIGHT
        self.calibration = calibration
        self.montage = montage
        self.panel_image = panel_image
        self.panel_width = panel_image.width
        self.panel_height = panel_image.height
        self.active_locations = {
            location
            for location in montage.mapping.values()
            if location in calibration.get("locations", calibration.get("sensors", {}))
        }
        self.glyphs: dict[str, SensorGlyph] = {}
        self.metric_text_ids: dict[str, int] = {}
        self.overall_text_id: int | None = None
        self.overall_background_id: int | None = None
        self.draw()

    def point(self, location: str) -> tuple[float, float]:
        coordinates = self.calibration.get("locations", self.calibration.get("sensors", {}))[location]
        return (
            float(coordinates[0]) * self.panel_width,
            float(coordinates[1]) * self.panel_height,
        )

    def draw(self) -> None:
        self.canvas.create_text(
            self.panel_left + self.panel_width / 2,
            20,
            text=self.title,
            fill="#23374d",
            font=("Helvetica", 16, "bold"),
        )
        if self.panel_image.image is not None:
            self.canvas.create_image(
                self.panel_left,
                self.panel_top,
                image=self.panel_image.image,
                anchor="nw",
            )

        for location in FLEX_10_20_LOCATIONS:
            x, y = self.point(location)
            if self.panel_image.image is None:
                self.canvas.create_oval(
                    self.panel_left + x - INACTIVE_RADIUS,
                    self.panel_top + y - INACTIVE_RADIUS,
                    self.panel_left + x + INACTIVE_RADIUS,
                    self.panel_top + y + INACTIVE_RADIUS,
                    fill="#f7fafc",
                    outline="#9aa8b5",
                )
                self.canvas.create_text(
                    self.panel_left + x + 7,
                    self.panel_top + y - 7,
                    text=location,
                    anchor="w",
                    fill="#52616f",
                    font=("Helvetica", 8),
                )

        for location in sorted(self.active_locations):
            x, y = self.point(location)
            ring_id = self.canvas.create_oval(
                self.panel_left + x - RING_RADIUS,
                self.panel_top + y - RING_RADIUS,
                self.panel_left + x + RING_RADIUS,
                self.panel_top + y + RING_RADIUS,
                fill="#111111",
                outline="",
            )
            oval_id = self.canvas.create_oval(
                self.panel_left + x - SENSOR_RADIUS,
                self.panel_top + y - SENSOR_RADIUS,
                self.panel_left + x + SENSOR_RADIUS,
                self.panel_top + y + SENSOR_RADIUS,
                fill=QUALITY_COLORS[0],
                outline="",
            )
            text_id = self.canvas.create_text(
                self.panel_left + x,
                self.panel_top + y,
                text=location,
                fill="white",
                font=("Helvetica", 7, "bold"),
            )
            self.glyphs[location] = SensorGlyph(ring_id, oval_id, text_id)

        line_y = self.panel_top + self.panel_height + 20
        metric_labels = (
            ("battery", "Battery"),
            ("signal", "Signal"),
            ("sampleRateQuality", "Rate"),
        )
        for key, label in metric_labels:
            self.metric_text_ids[key] = self.canvas.create_text(
                self.panel_left + 16,
                line_y,
                text=f"{label}: --",
                anchor="w",
                fill="#333333",
                font=("Helvetica", 11, "bold"),
            )
            line_y += 26

        overall = self.calibration["overall_anchor"]
        overall_x = self.panel_left + float(overall[0]) * self.panel_width
        overall_y = self.panel_top + float(overall[1]) * self.panel_height
        self.overall_background_id = self.canvas.create_rectangle(
            overall_x - 58,
            overall_y - 26,
            overall_x + 58,
            overall_y + 26,
            fill="white",
            outline="white",
        )
        self.overall_text_id = self.canvas.create_text(
            overall_x,
            overall_y,
            text="--%",
            fill="#8a8a8a",
            font=("Helvetica", 28, "bold"),
        )

    def update(
        self,
        scores_by_wire: dict[str, float],
        metrics: dict[str, float | None],
    ) -> None:
        by_location: dict[str, list[float]] = {}
        for wire, score in scores_by_wire.items():
            location = self.montage.mapping.get(wire)
            if location in self.glyphs:
                by_location.setdefault(location, []).append(score)

        for location, glyph in self.glyphs.items():
            values = by_location.get(location)
            score = sum(values) / len(values) if values else None
            self.canvas.itemconfigure(glyph.oval_id, fill=score_to_color(score))

        battery_percent = metrics.get("batteryPercent")
        battery = metrics.get("battery")
        signal = metrics.get("signal")
        rate = metrics.get("sampleRateQuality")
        if "battery" in self.metric_text_ids:
            if battery_percent is None:
                self.canvas.itemconfigure(self.metric_text_ids["battery"], text="Battery: --")
            elif battery is None:
                self.canvas.itemconfigure(
                    self.metric_text_ids["battery"],
                    text=f"Battery: {battery_percent:.0f}%",
                )
            else:
                self.canvas.itemconfigure(
                    self.metric_text_ids["battery"],
                    text=f"Battery: {battery_percent:.0f}% ({battery:.0f}/4)",
                )
        if "signal" in self.metric_text_ids:
            self.canvas.itemconfigure(
                self.metric_text_ids["signal"],
                text="Signal: --" if signal is None else f"Signal: {signal:.0f}",
            )
        if "sampleRateQuality" in self.metric_text_ids:
            self.canvas.itemconfigure(
                self.metric_text_ids["sampleRateQuality"],
                text="Rate: --" if rate is None else f"Rate: {rate:.2f}",
            )

        overall = metrics.get("overall")
        if self.overall_text_id is not None:
            self.canvas.itemconfigure(
                self.overall_text_id,
                text="--%" if overall is None else f"{overall:.0f}%",
                fill=overall_to_color(overall),
            )


class FlexQualityViewer:
    def __init__(
        self,
        montage: FlexMontage,
        calibration: dict,
        image_path: Path,
        stream_prefix: str = DEFAULT_STREAM_PREFIX,
        no_background: bool = False,
    ) -> None:
        self.montage = montage
        self.calibration = calibration
        self.stream_prefix = stream_prefix
        self.root = tk.Tk()
        self.root.title("EPOC Flex Contact And EEG Quality")
        self.root.configure(bg="white")
        self.root.geometry("1200x800+80+80")
        self.root.minsize(900, 600)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<Control-w>", lambda _event: self.root.destroy())

        self.panel_image = load_panel_image(
            self.root,
            image_path,
            calibration,
            no_background,
        )
        if self.panel_image.error:
            print(
                f"Flex quality viewer: {self.panel_image.error}; using calibrated markers without a background image.",
                file=sys.stderr,
                flush=True,
            )
        self.panel_width = self.panel_image.width
        self.panel_height = self.panel_image.height

        canvas_width = (WINDOW_PADDING * 2) + (self.panel_width * 2) + PANEL_GAP
        canvas_height = HEADER_HEIGHT + self.panel_height + 110
        canvas_frame = tk.Frame(self.root, bg="white")
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            width=min(canvas_width, 1160),
            height=min(canvas_height, 700),
            bg="white",
            highlightthickness=0,
            xscrollincrement=20,
            yscrollincrement=20,
            scrollregion=(0, 0, canvas_width, canvas_height),
        )
        x_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        if self.panel_image.image is not None:
            self.canvas.image = self.panel_image.image

        self.status_var = tk.StringVar(value="Looking for Flex Cortex quality streams...")
        tk.Label(self.root, textvariable=self.status_var, bg="white", fg="#333333").pack(pady=(0, 8))

        self.cq_panel = FlexQualityPanel(
            self.canvas,
            "Contact Quality",
            WINDOW_PADDING,
            calibration,
            montage,
            self.panel_image,
        )
        self.eq_panel = FlexQualityPanel(
            self.canvas,
            "EEG Quality",
            WINDOW_PADDING + self.panel_width + PANEL_GAP,
            calibration,
            montage,
            self.panel_image,
        )
        self.cq_source: QualitySource | None = None
        self.eq_source: QualitySource | None = None

    def connect(self) -> None:
        cq_name = f"{self.stream_prefix} Contact Quality"
        eq_name = f"{self.stream_prefix} EEG Quality"
        cq_streams = resolve_byprop("name", cq_name, timeout=2)
        eq_streams = resolve_byprop("name", eq_name, timeout=2)
        if not cq_streams or not eq_streams:
            missing = []
            if not cq_streams:
                missing.append(cq_name)
            if not eq_streams:
                missing.append(eq_name)
            self.status_var.set(f"Missing stream(s): {', '.join(missing)}. Retrying...")
            self.root.after(1000, self.connect)
            return

        self.cq_source = QualitySource(StreamInlet(cq_streams[0]), self.montage, "dev")
        self.eq_source = QualitySource(StreamInlet(eq_streams[0]), self.montage, "eq")
        self.status_var.set(
            f"Connected; active Flex locations: {len(self.montage.mapping)}"
        )
        self.poll()

    def poll(self) -> None:
        updated = False
        if self.cq_source:
            while True:
                sample, _ = self.cq_source.inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                scores, metrics = self.cq_source.decode(sample)
                self.cq_panel.update(scores, metrics)
                updated = True
        if self.eq_source:
            while True:
                sample, _ = self.eq_source.inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                scores, metrics = self.eq_source.decode(sample)
                self.eq_panel.update(scores, metrics)
                updated = True
        if not updated and (self.cq_source or self.eq_source):
            self.status_var.set(
                f"Connected; active Flex locations: {len(self.montage.mapping)}"
            )
        self.root.after(100, self.poll)

    def run(self) -> None:
        self.root.after(0, self.connect)
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--stream-prefix", default=DEFAULT_STREAM_PREFIX)
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="show calibrated coordinates on a plain canvas instead of loading the PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    montage = load_flex_montage(args.mapping)
    calibration = load_calibration(args.calibration)
    FlexQualityViewer(
        montage,
        calibration,
        args.image,
        stream_prefix=args.stream_prefix,
        no_background=args.no_background,
    ).run()


if __name__ == "__main__":
    main()
