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
PANEL_WIDTH = 462
PANEL_HEIGHT = 510
PANEL_GAP = 18
WINDOW_PADDING = 12
HEADER_HEIGHT = 34
STATUS_HEIGHT = 32
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
SENSOR_POSITIONS = {
    "AF3": (173, 126),
    "F7": (119, 143),
    "FC5": (119, 197),
    "F3": (173, 198),
    "T7": (84, 267),
    "P7": (132, 393),
    "O1": (175, 460),
    "O2": (257, 460),
    "P8": (301, 393),
    "T8": (347, 267),
    "FC6": (308, 198),
    "F4": (259, 197),
    "F8": (341, 143),
    "AF4": (258, 126),
}
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


class QualityPanel:
    def __init__(
        self,
        canvas: tk.Canvas,
        title: str,
        x_offset: int,
        channel_order: list[str],
        overall_key: str,
        left_metric_lines: list[tuple[str, str]],
    ) -> None:
        self.canvas = canvas
        self.title = title
        self.x_offset = x_offset
        self.channel_index = {name: index for index, name in enumerate(channel_order)}
        self.overall_key = overall_key
        self.left_metric_lines = left_metric_lines
        self.sensor_glyphs: dict[str, SensorGlyph] = {}
        self.metric_text_ids: dict[str, int] = {}
        self.overall_text_id: int | None = None
        self.background_image: tk.PhotoImage | None = None
        self.draw()

    def draw(self) -> None:
        self.canvas.create_text(
            self.x_offset + PANEL_WIDTH // 2,
            20,
            text=self.title,
            fill="#23374d",
            font=("Helvetica", 16, "bold"),
        )

        panel_top = HEADER_HEIGHT
        panel_image_path = next((path for path in PANEL_IMAGE_CANDIDATES if path.exists()), None)
        if panel_image_path:
            self.background_image = tk.PhotoImage(file=str(panel_image_path))
            self.canvas.create_image(self.x_offset, panel_top, image=self.background_image, anchor="nw")
        else:
            self.canvas.create_rectangle(
                self.x_offset,
                panel_top,
                self.x_offset + PANEL_WIDTH,
                panel_top + PANEL_HEIGHT,
                fill="#dce9fb",
                outline="#263648",
                width=2,
            )

        self.canvas.create_rectangle(
            self.x_offset + 320,
            panel_top + 430,
            self.x_offset + PANEL_WIDTH - 6,
            panel_top + PANEL_HEIGHT - 6,
            fill="white",
            outline="white",
        )

        for name, (sensor_x, sensor_y) in SENSOR_POSITIONS.items():
            x = self.x_offset + sensor_x
            y = panel_top + sensor_y
            ring_id = self.canvas.create_oval(x - 21, y - 21, x + 21, y + 21, fill="#111111", outline="")
            oval_id = self.canvas.create_oval(x - 17, y - 17, x + 17, y + 17, fill=QUALITY_COLORS[0], outline="")
            text_id = self.canvas.create_text(x, y, text=name, fill="white", font=("Helvetica", 10, "bold"))
            self.sensor_glyphs[name] = SensorGlyph(oval_id=oval_id, text_id=text_id, ring_id=ring_id)

        line_y = panel_top + PANEL_HEIGHT + 20
        for key, label in self.left_metric_lines:
            self.metric_text_ids[key] = self.canvas.create_text(
                self.x_offset + 16,
                line_y,
                text=f"{label}: --",
                anchor="w",
                fill="#333333",
                font=("Helvetica", 11, "bold"),
            )
            line_y += 26

        self.overall_text_id = self.canvas.create_text(
            self.x_offset + 385,
            panel_top + 442,
            text="--%",
            fill="#8ad448",
            font=("Helvetica", 28, "bold"),
        )

    def update_sensor_scores(self, sample: list[float]) -> None:
        for sensor_name in SENSOR_NAMES:
            score = sample[self.channel_index[sensor_name]]
            glyph = self.sensor_glyphs[sensor_name]
            self.canvas.itemconfigure(glyph.oval_id, fill=score_to_color(score))

    def update_metrics(self, metrics: dict[str, str], overall: float) -> None:
        for key, value in metrics.items():
            self.canvas.itemconfigure(self.metric_text_ids[key], text=value)

        self.canvas.itemconfigure(
            self.overall_text_id,
            text=f"{overall:.0f}%",
            fill=overall_to_color(overall),
        )


class DualQualityViewer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Emotiv Contact And EEG Quality")
        self.root.configure(bg="white")

        canvas_width = (WINDOW_PADDING * 2) + (PANEL_WIDTH * 2) + PANEL_GAP
        canvas_height = HEADER_HEIGHT + PANEL_HEIGHT + 72
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.status_var = tk.StringVar(value="Looking for LSL quality streams...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white", fg="#333333")
        self.status_label.pack(pady=(0, 8))

        self.cq_panel = QualityPanel(
            canvas=self.canvas,
            title="Contact Quality",
            x_offset=WINDOW_PADDING,
            channel_order=CQ_CHANNEL_ORDER,
            overall_key="OVERALL",
            left_metric_lines=[("Battery", "Battery"), ("Signal", "Signal")],
        )
        self.eq_panel = QualityPanel(
            canvas=self.canvas,
            title="EEG Quality",
            x_offset=WINDOW_PADDING + PANEL_WIDTH + PANEL_GAP,
            channel_order=EQ_CHANNEL_ORDER,
            overall_key="overall",
            left_metric_lines=[("batteryPercent", "Battery"), ("sampleRateQuality", "Rate")],
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
