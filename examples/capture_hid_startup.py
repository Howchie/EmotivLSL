"""Capture the EPOC X HID startup handshake and steady-state reports.

Run this with the USB dongle inserted and the headset powered off.  The
recorder polls for both Emotiv HID interfaces, so the headset can be powered on
after the program starts.  It deliberately does not decrypt or interpret the
reports; the firmware-0x740 control packet is needed in its original form.
"""

import argparse
import csv
import sys
import threading
import time

import hid


# Short enough to attach promptly when Windows creates the composite HID
# interfaces after the headset powers on, while avoiding a tight enumeration
# loop.
POLL_SECONDS = 0.05
READ_SIZE = 64
READ_TIMEOUT_MS = 500
EMOTIV_VENDOR_ID = 0x1234
EMOTIV_PRODUCT_ID = 0xED02


def is_emotiv(device: dict) -> bool:
    manufacturer = device.get("manufacturer_string")
    if isinstance(manufacturer, str) and manufacturer.casefold() == "emotiv":
        return True
    # Some Windows HID enumeration paths omit the manufacturer string.  The
    # EPOC X receiver and EEG interfaces share this VID/PID pair.
    return (
        device.get("vendor_id") == EMOTIV_VENDOR_ID
        and device.get("product_id") == EMOTIV_PRODUCT_ID
    )


def device_description(device: dict) -> str:
    return (
        f"product={device.get('product_string')} "
        f"usage_page={device.get('usage_page')} usage={device.get('usage')} "
        f"interface={device.get('interface_number')} serial={device.get('serial_number')}"
    )


def capture_device(
    device: dict,
    writer: csv.writer,
    output,
    write_lock: threading.Lock,
    stop: threading.Event,
) -> None:
    handle = hid.device()
    try:
        handle.open_path(device["path"])
        print(f"Capturing {device_description(device)}", file=sys.stderr, flush=True)
        while not stop.is_set():
            try:
                report = handle.read(READ_SIZE, timeout_ms=READ_TIMEOUT_MS)
            except Exception as exc:  # HID backends report disconnects differently.
                print(f"HID read stopped for {device.get('path')!r}: {exc}", file=sys.stderr, flush=True)
                return
            if not report:
                continue

            row = [
                f"{time.time():.9f}",
                device.get("product_string", ""),
                device.get("usage_page", ""),
                device.get("usage", ""),
                device.get("interface_number", ""),
                device.get("serial_number", ""),
                len(report),
                bytes(report).hex(),
            ]
            with write_lock:
                writer.writerow(row)
                # Keep the capture useful even if the headset disconnects or
                # the user stops the process before the requested duration.
                output.flush()

    except Exception as exc:
        print(f"Could not open HID path {device.get('path')!r}: {exc}", file=sys.stderr, flush=True)
    finally:
        handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/epocx_startup_hid.csv", metavar="PATH")
    parser.add_argument("--seconds", type=float, default=20.0, help="capture duration (default: 20 seconds)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")

    output_path = args.output
    if output_path:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open("w", newline="", encoding="utf-8")
    else:
        output = sys.stdout

    writer = csv.writer(output)
    writer.writerow([
        "timestamp_unix",
        "product_string",
        "usage_page",
        "usage",
        "interface_number",
        "serial_number",
        "report_length",
        "report_hex",
    ])
    output.flush()
    if output is not sys.stdout:
        print(f"Writing raw HID capture to {path.resolve()}", file=sys.stderr, flush=True)
    else:
        print("Writing raw HID capture to stdout", file=sys.stderr, flush=True)

    stop = threading.Event()
    write_lock = threading.Lock()
    workers: dict[str, threading.Thread] = {}
    deadline = time.monotonic() + args.seconds
    next_wait_message = 0.0

    try:
        while time.monotonic() < deadline:
            devices = [device for device in hid.enumerate() if is_emotiv(device)]
            now = time.monotonic()
            if not devices and now >= next_wait_message:
                print(
                    "Waiting for an Emotiv HID interface; power on the headset when ready.",
                    file=sys.stderr,
                    flush=True,
                )
                next_wait_message = now + 5.0

            for device in devices:
                path_key = str(device.get("path"))
                worker = workers.get(path_key)
                if worker and worker.is_alive():
                    continue
                if worker:
                    workers.pop(path_key, None)

                print(f"Found Emotiv HID {device_description(device)}", file=sys.stderr, flush=True)

                thread = threading.Thread(
                    target=capture_device,
                    args=(device, writer, output, write_lock, stop),
                    daemon=True,
                    name=f"hid-{device.get('usage')}",
                )
                workers[path_key] = thread
                thread.start()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("Stopping capture.", file=sys.stderr, flush=True)
    finally:
        stop.set()
        for worker in workers.values():
            worker.join(timeout=READ_TIMEOUT_MS / 1000 + 1)
        output.flush()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
