import tkinter as tk
from dataclasses import dataclass

from pylsl import StreamInlet, resolve_byprop


CQ_STREAM_NAME = "Epoc X Contact Quality"
EQ_STREAM_NAME = "Epoc X EEG Quality"
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
    "AF3": (0, -140),
    "F7": (-58, -102),
    "F3": (0, -60),
    "FC5": (-38, -60),
    "T7": (-95, 5),
    "P7": (-42, 122),
    "O1": (2, 206),
    "O2": (98, 206),
    "P8": (138, 122),
    "T8": (190, 5),
    "FC6": (142, -60),
    "F4": (95, -60),
    "F8": (155, -102),
    "AF4": (98, -140),
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
        self.draw()

    def draw(self) -> None:
        self.canvas.create_text(
            self.x_offset + 150,
            34,
            text=self.title,
            fill="#23374d",
            font=("Helvetica", 16, "bold"),
        )

        self.canvas.create_oval(
            self.x_offset + 30,
            60,
            self.x_offset + 270,
            405,
            fill="#dce9fb",
            outline="#263648",
            width=3,
        )
        self.canvas.create_arc(
            self.x_offset + 46,
            86,
            self.x_offset + 122,
            250,
            start=120,
            extent=135,
            style=tk.ARC,
            outline="#263648",
            width=2,
        )
        self.canvas.create_arc(
            self.x_offset + 180,
            86,
            self.x_offset + 256,
            250,
            start=-75,
            extent=135,
            style=tk.ARC,
            outline="#263648",
            width=2,
        )

        center_x = self.x_offset + 88
        center_y = 225
        for name, (dx, dy) in SENSOR_POSITIONS.items():
            x = center_x + dx
            y = center_y + dy
            ring_id = self.canvas.create_oval(x - 21, y - 21, x + 21, y + 21, fill="#111111", outline="")
            oval_id = self.canvas.create_oval(x - 17, y - 17, x + 17, y + 17, fill=QUALITY_COLORS[0], outline="")
            text_id = self.canvas.create_text(x, y, text=name, fill="white", font=("Helvetica", 10, "bold"))
            self.sensor_glyphs[name] = SensorGlyph(oval_id=oval_id, text_id=text_id, ring_id=ring_id)

        line_y = 445
        for key, label in self.left_metric_lines:
            self.metric_text_ids[key] = self.canvas.create_text(
                self.x_offset + 24,
                line_y,
                text=f"{label}: --",
                anchor="w",
                fill="#333333",
                font=("Helvetica", 11, "bold"),
            )
            line_y += 26

        self.overall_text_id = self.canvas.create_text(
            self.x_offset + 230,
            458,
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

        self.canvas = tk.Canvas(self.root, width=680, height=545, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.status_var = tk.StringVar(value="Looking for LSL quality streams...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="white", fg="#333333")
        self.status_label.pack(pady=(0, 8))

        self.cq_panel = QualityPanel(
            canvas=self.canvas,
            title="Contact Quality",
            x_offset=12,
            channel_order=CQ_CHANNEL_ORDER,
            overall_key="OVERALL",
            left_metric_lines=[("Battery", "Battery"), ("Signal", "Signal")],
        )
        self.eq_panel = QualityPanel(
            canvas=self.canvas,
            title="EEG Quality",
            x_offset=344,
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
