import json
import tkinter as tk
from pathlib import Path


PANEL_IMAGE_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "head_image.png",
    Path(__file__).resolve().parents[1] / "example_cq.png",
)
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "head_image_coords.json"
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
CLICK_SEQUENCE = SENSOR_NAMES + ["OVERALL"]


def pick_panel_image() -> Path:
    for path in PANEL_IMAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No panel image found. Expected head_image.png or example_cq.png in the repo root.")


class HeadImageCalibrator:
    def __init__(self) -> None:
        self.image_path = pick_panel_image()
        self.root = tk.Tk()
        self.root.title("Calibrate Head Image")

        self.image = tk.PhotoImage(file=str(self.image_path))
        self.width = self.image.width()
        self.height = self.image.height()

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")
        self.canvas.bind("<Button-1>", self.on_click)

        self.status_var = tk.StringVar()
        self.status = tk.Label(self.root, textvariable=self.status_var)
        self.status.pack(pady=8)

        self.index = 0
        self.points: dict[str, list[float]] = {"sensors": {}}
        self.update_status()

    def update_status(self) -> None:
        target = CLICK_SEQUENCE[self.index]
        self.status_var.set(f"Click {target}")

    def on_click(self, event) -> None:
        key = CLICK_SEQUENCE[self.index]
        norm_x = event.x / self.width
        norm_y = event.y / self.height

        self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="red", outline="")
        self.canvas.create_text(event.x + 10, event.y - 10, text=key, anchor="w", fill="red", font=("Helvetica", 10, "bold"))

        if key == "OVERALL":
            self.points["overall_anchor"] = [norm_x, norm_y]
        else:
            self.points["sensors"][key] = [norm_x, norm_y]

        self.index += 1
        if self.index == len(CLICK_SEQUENCE):
            payload = {
                "image": self.image_path.name,
                "image_width": self.width,
                "image_height": self.height,
                "sensors": self.points["sensors"],
                "overall_anchor": self.points["overall_anchor"],
            }
            CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))
            self.status_var.set(f"Saved calibration to {CALIBRATION_PATH}")
            self.canvas.unbind("<Button-1>")
            return

        self.update_status()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    HeadImageCalibrator().run()


if __name__ == "__main__":
    main()
