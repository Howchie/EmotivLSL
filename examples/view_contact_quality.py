import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from pylsl import StreamInlet, resolve_byprop


CQ_STREAM_NAME = "Epoc X Contact Quality"
EQ_STREAM_NAME = "Epoc X EEG Quality"
PANEL_IMAGE_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "head_image.png",
    Path(__file__).resolve().parents[1] / "example_cq.png",
)
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "head_image_coords.json"
PANEL_WIDTH = 462
PANEL_HEIGHT = 510
PANEL_GAP = 18
WINDOW_PADDING = 12
HEADER_HEIGHT = 34
SENSOR_RADIUS = 17
RING_RADIUS = 21
CQ_CHANNEL_ORDER = [
    "Battery",
    "Signal",
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
    "OVERALL",
    "BatteryPercent",
]
EQ_CHANNEL_ORDER = [
    "batteryPercent",
    "overall",
    "sampleRateQuality",
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
SENSOR_NAMES = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
QUALITY_COLORS = {
    0: "#111111",
    1: "#d84a3a",
    2: "#f09a3e",
    3: "#b8d94a",
    4: "#1ebf5b",
}


@dataclass
class SensorGlyph:
    oval_id: int
    text_id: int
    ring_id: int


def score_to_color(score: float) -> str:
    rounded = max(0, min(4, int(round(score))))
    return QUALITY_COLORS[rounded]


def overall_to_color(overall: float) -> str:
    if overall >= 80:
        return "#8ad448"
    if overall >= 60:
        return "#c7d94a"
    if overall >= 40:
        return "#f0b23e"
    return "#d84a3a"


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {CALIBRATION_PATH}. Run examples/calibrate_head_image.py first."
        )
    return json.loads(CALIBRATION_PATH.read_text())


def pick_panel_image() -> Path:
    for path in PANEL_IMAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No panel image found. Expected head_image.png or example_cq.png in the repo root.")


def scale_point(panel_left: int, panel_top: int, norm_x: float, norm_y: float) -> tuple[float, float]:
    return (
        panel_left + (norm_x * PANEL_WIDTH),
        panel_top + (norm_y * PANEL_HEIGHT),
    )


class QualityPanel:
    def __init__(
        self,
        canvas: tk.Canvas,
        title: str,
        panel_left: int,
        channel_order: list[str],
        metric_labels: list[tuple[str, str]],
        calibration: dict,
        panel_image: tk.PhotoImage,
    ) -> None:
        self.canvas = canvas
        self.title = title
        self.panel_left = panel_left
        self.panel_top = HEADER_HEIGHT
        self.channel_index = {name: index for index, name in enumerate(channel_order)}
        self.metric_labels = metric_labels
        self.calibration = calibration
        self.panel_image = panel_image
        self.sensor_glyphs: dict[str, SensorGlyph] = {}
        self.metric_text_ids: dict[str, int] = {}
        self.overall_text_id: int | None = None
        self.overall_background_id: int | None = None
        self.draw()

    def draw(self) -> None:
        self.canvas.create_text(
            self.panel_left + PANEL_WIDTH / 2,
            20,
            text=self.title,
            fill="#23374d",
            font=("Helvetica", 16, "bold"),
        )
        self.canvas.create_image(self.panel_left, self.panel_top, image=self.panel_image, anchor="nw")

        for sensor_name in SENSOR_NAMES:
            norm_x, norm_y = self.calibration["sensors"][sensor_name]
            x, y = scale_point(self.panel_left, self.panel_top, norm_x, norm_y)
            ring_id = self.canvas.create_oval(
                x - RING_RADIUS, y - RING_RADIUS, x + RING_RADIUS, y + RING_RADIUS, fill="#111111", outline=""
            )
            oval_id = self.canvas.create_oval(
                x - SENSOR_RADIUS, y - SENSOR_RADIUS, x + SENSOR_RADIUS, y + SENSOR_RADIUS, fill=QUALITY_COLORS[0], outline=""
            )
            text_id = self.canvas.create_text(x, y, text=sensor_name, fill="white", font=("Helvetica", 10, "bold"))
            self.sensor_glyphs[sensor_name] = SensorGlyph(oval_id=oval_id, text_id=text_id, ring_id=ring_id)

        line_y = self.panel_top + PANEL_HEIGHT + 20
        for key, label in self.metric_labels:
            self.metric_text_ids[key] = self.canvas.create_text(
                self.panel_left + 16,
                line_y,
                text=f"{label}: --",
                anchor="w",
                fill="#333333",
                font=("Helvetica", 11, "bold"),
            )
            line_y += 26

        overall_x, overall_y = self.calibration["overall_anchor"]
        overall_fill_x, overall_fill_y = scale_point(self.panel_left, self.panel_top, overall_x, overall_y)
        self.overall_background_id = self.canvas.create_rectangle(
            overall_fill_x - 52,
            overall_fill_y - 24,
            overall_fill_x + 52,
            overall_fill_y + 24,
            fill="white",
            outline="white",
        )
        self.overall_text_id = self.canvas.create_text(
            overall_fill_x,
            overall_fill_y,
            text="--%",
            fill="#8ad448",
            font=("Helvetica", 28, "bold"),
        )
        self.update_overall_badge("--%", "#8ad448")

    def update_sensor_scores(self, sample: list[float]) -> None:
        for sensor_name in SENSOR_NAMES:
            score = sample[self.channel_index[sensor_name]]
            self.canvas.itemconfigure(self.sensor_glyphs[sensor_name].oval_id, fill=score_to_color(score))

    def update_metrics(self, metrics: dict[str, str], overall: float) -> None:
        for key, value in metrics.items():
            self.canvas.itemconfigure(self.metric_text_ids[key], text=value)
        self.update_overall_badge(f"{overall:.0f}%", overall_to_color(overall))

    def update_overall_badge(self, text: str, color: str) -> None:
        self.canvas.itemconfigure(self.overall_text_id, text=text, fill=color)
        bbox = self.canvas.bbox(self.overall_text_id)
        if not bbox:
            return
        pad_x = 12
        pad_y = 10
        self.canvas.coords(
            self.overall_background_id,
            bbox[0] - pad_x,
            bbox[1] - pad_y,
            bbox[2] + pad_x,
            bbox[3] + pad_y,
        )


class DualQualityViewer:
    def __init__(self) -> None:
        self.calibration = load_calibration()

        self.root = tk.Tk()
        self.root.title("Emotiv Contact And EEG Quality")
        self.root.configure(bg="white")
        self.panel_image = tk.PhotoImage(file=str(pick_panel_image()))

        canvas_width = (WINDOW_PADDING * 2) + (PANEL_WIDTH * 2) + PANEL_GAP
        canvas_height = HEADER_HEIGHT + PANEL_HEIGHT + 72
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        viewport_width = min(canvas_width, max(640, screen_width - 120))
        viewport_height = min(canvas_height, max(480, screen_height - 180))

        canvas_frame = tk.Frame(self.root, bg="white")
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=viewport_width,
            height=viewport_height,
            bg="white",
            highlightthickness=0,
            xscrollincrement=20,
            yscrollincrement=20,
        )
        x_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        y_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set, scrollregion=(0, 0, canvas_width, canvas_height))

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Looking for LSL quality streams...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white", fg="#333333")
        self.status_label.pack(pady=(0, 8))

        self.cq_panel = QualityPanel(
            canvas=self.canvas,
            title="Contact Quality",
            panel_left=WINDOW_PADDING,
            channel_order=CQ_CHANNEL_ORDER,
            metric_labels=[("Battery", "Battery"), ("Signal", "Signal")],
            calibration=self.calibration,
            panel_image=self.panel_image,
        )
        self.eq_panel = QualityPanel(
            canvas=self.canvas,
            title="EEG Quality",
            panel_left=WINDOW_PADDING + PANEL_WIDTH + PANEL_GAP,
            channel_order=EQ_CHANNEL_ORDER,
            metric_labels=[("batteryPercent", "Battery"), ("sampleRateQuality", "Rate")],
            calibration=self.calibration,
            panel_image=self.panel_image,
        )

        self.cq_inlet: StreamInlet | None = None
        self.eq_inlet: StreamInlet | None = None

    def connect(self) -> None:
        self.status_var.set("Resolving LSL quality streams...")
        cq_streams = resolve_byprop("name", CQ_STREAM_NAME, timeout=2)
        eq_streams = resolve_byprop("name", EQ_STREAM_NAME, timeout=2)
        if not cq_streams or not eq_streams:
            missing = []
            if not cq_streams:
                missing.append(CQ_STREAM_NAME)
            if not eq_streams:
                missing.append(EQ_STREAM_NAME)
            self.status_var.set(f"Missing stream(s): {', '.join(missing)}. Retrying...")
            self.root.after(1000, self.connect)
            return

        self.cq_inlet = StreamInlet(cq_streams[0])
        self.eq_inlet = StreamInlet(eq_streams[0])
        self.status_var.set("Connected to both quality streams")
        self.poll()

    def poll(self) -> None:
        updated = False
        if self.cq_inlet:
            while True:
                sample, _ = self.cq_inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                self.update_cq(sample)
                updated = True
        if self.eq_inlet:
            while True:
                sample, _ = self.eq_inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                self.update_eq(sample)
                updated = True
        if not updated:
            self.status_var.set("Connected to both quality streams")
        self.root.after(100, self.poll)

    def update_cq(self, sample: list[float]) -> None:
        self.cq_panel.update_sensor_scores(sample)
        battery = sample[self.cq_panel.channel_index["Battery"]]
        signal = sample[self.cq_panel.channel_index["Signal"]]
        overall = sample[self.cq_panel.channel_index["OVERALL"]]
        battery_percent = sample[self.cq_panel.channel_index["BatteryPercent"]]
        self.cq_panel.update_metrics(
            {
                "Battery": f"Battery: {battery_percent:.0f}% ({battery:.0f}/4)",
                "Signal": f"Signal: {signal:.0f}",
            },
            overall,
        )

    def update_eq(self, sample: list[float]) -> None:
        self.eq_panel.update_sensor_scores(sample)
        battery_percent = sample[self.eq_panel.channel_index["batteryPercent"]]
        sample_rate_quality = sample[self.eq_panel.channel_index["sampleRateQuality"]]
        overall = sample[self.eq_panel.channel_index["overall"]]
        self.eq_panel.update_metrics(
            {
                "batteryPercent": f"Battery: {battery_percent:.0f}%",
                "sampleRateQuality": f"Rate: {sample_rate_quality:.2f}",
            },
            overall,
        )

    def run(self) -> None:
        self.root.after(0, self.connect)
        self.root.mainloop()


def main() -> None:
    DualQualityViewer().run()


if __name__ == "__main__":
    main()
