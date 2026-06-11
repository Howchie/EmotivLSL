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
        self.points: dict[str, list[float]] = {"sensors": {}}
        self.update_status()

    def update_status(self) -> None:
        target = CLICK_SEQUENCE[self.index]
        self.status_var.set(f"Click {target}")

    def on_click(self, event) -> None:
        key = CLICK_SEQUENCE[self.index]
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        norm_x = canvas_x / self.width
        norm_y = canvas_y / self.height

        self.canvas.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5, fill="red", outline="")
        self.canvas.create_text(
            canvas_x + 10,
            canvas_y - 10,
            text=key,
            anchor="w",
            fill="red",
            font=("Helvetica", 10, "bold"),
        )

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
