import argparse
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass

from pylsl import StreamInfo, StreamOutlet, local_clock
from websocket import WebSocketTimeoutException, create_connection


DEFAULT_CORTEX_URL = "wss://localhost:6868"
DEFAULT_STREAMS = ("dev", "eq")
POLL_INTERVAL_SECONDS = 1.0


@dataclass
class StreamSpec:
    cortex_name: str
    lsl_name: str
    lsl_type: str
    channel_format: str
    nominal_srate: float


@dataclass
class BridgeConfig:
    client_id: str
    client_secret: str
    license: str | None = None
    cortex_url: str = DEFAULT_CORTEX_URL
    headset_id: str | None = None
    streams: tuple[str, ...] = DEFAULT_STREAMS
    verify_ssl: bool = False
    print_samples: bool = False
    stream_prefix: str = "Epoc X"


STREAM_SPECS = {
    # Raw EEG is delivered by Cortex after it has performed the firmware-
    # specific decryption.  Keep this outlet name distinct from the legacy
    # direct-HID outlet so both paths can be run side by side while testing.
    "eeg": StreamSpec("eeg", "Epoc X Cortex EEG", "EEG", "float32", 0),
    "dev": StreamSpec("dev", "Epoc X Contact Quality", "EmotivCQ", "float32", 2),
    "eq": StreamSpec("eq", "Epoc X EEG Quality", "EmotivEQ", "float32", 2),
    "pow": StreamSpec("pow", "Epoc X Band Power", "EmotivPow", "float32", 8),
    "met": StreamSpec("met", "Epoc X Performance Metrics", "EmotivMet", "float32", 0),
    "com": StreamSpec("com", "Epoc X Mental Commands", "EmotivCom", "string", 8),
    "fac": StreamSpec("fac", "Epoc X Facial Expressions", "EmotivFac", "string", 32),
}


class CortexError(RuntimeError):
    pass


class CortexClient:
    def __init__(self, url: str, verify_ssl: bool) -> None:
        sslopt = None if verify_ssl else {"cert_reqs": ssl.CERT_NONE}
        self.ws = create_connection(url, sslopt=sslopt, timeout=10)
        self.next_id = 1

    def close(self) -> None:
        self.ws.close()

    def call(self, method: str, params: dict | None = None) -> dict | list:
        request_id = self.next_id
        self.next_id += 1
        payload = {
            "id": request_id,
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        self.ws.send(json.dumps(payload))

        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                raise CortexError(f"{method} failed: {error.get('code')} {error.get('message')}")
            return message["result"]

    def recv(self) -> dict:
        return json.loads(self.ws.recv())

    def settimeout(self, seconds: float) -> None:
        self.ws.settimeout(seconds)


def add_channel_metadata(info: StreamInfo, labels: list[str]) -> None:
    chns = info.desc().append_child("channels")
    for label in labels:
        if isinstance(label, list):
            for part in label:
                ch = chns.append_child("channel")
                ch.append_child_value("label", str(part))
                ch.append_child_value("type", "Cortex")
                ch.append_child_value("unit", "score")
            continue

        ch = chns.append_child("channel")
        ch.append_child_value("label", str(label))
        ch.append_child_value("type", "Cortex")
        ch.append_child_value("unit", "score")


def flatten_labels(labels: list) -> list[str]:
    flattened: list[str] = []
    for label in labels:
        if isinstance(label, list):
            flattened.extend(str(part) for part in label)
        else:
            flattened.append(str(label))
    return flattened


def flatten_values(values: list) -> list:
    flattened: list = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(part for part in value)
        else:
            flattened.append(value)
    return flattened


def create_outlet(
    spec: StreamSpec,
    labels: list[str],
    nominal_srate: float | None = None,
    lsl_name: str | None = None,
) -> StreamOutlet:
    rate = spec.nominal_srate if nominal_srate is None else nominal_srate
    info = StreamInfo(
        spec.lsl_name if lsl_name is None else lsl_name,
        spec.lsl_type,
        len(flatten_labels(labels)),
        rate,
        spec.channel_format,
    )
    info.desc().append_child_value("source_stream", spec.cortex_name)
    info.desc().append_child_value("cortex_nominal_srate_hz", str(rate))
    add_channel_metadata(info, labels)
    return StreamOutlet(info)


def stream_lsl_name(spec: StreamSpec, prefix: str) -> str:
    """Return a device-specific LSL name while preserving legacy defaults."""
    if not prefix:
        return spec.lsl_name
    legacy_prefix = "Epoc X"
    if spec.lsl_name.startswith(f"{legacy_prefix} "):
        suffix = spec.lsl_name[len(legacy_prefix):]
        return f"{prefix}{suffix}"
    return spec.lsl_name


def convert_sample(values: list, channel_format: str) -> list:
    flattened = flatten_values(values)
    if channel_format == "string":
        return ["" if value is None else str(value) for value in flattened]
    return [float("nan") if value is None else float(value) for value in flattened]


def convert_stream_sample(stream_name: str, values: list, labels: list[str], channel_format: str) -> list:
    """Convert one Cortex sample while handling EEG's marker-object column.

    Cortex's ``eeg`` stream ends with ``MARKERS``, whose value is an array of
    JSON objects rather than a numeric sample.  LSL's float outlet cannot carry
    that object, so the marker column is intentionally omitted from the EEG
    outlet.  ``MARKER_HARDWARE`` remains available as a numeric channel.
    """
    if stream_name != "eeg":
        return convert_sample(values, channel_format)

    numeric_values = [value for label, value in zip(labels, values) if label != "MARKERS"]
    return convert_sample(numeric_values, channel_format)


def eeg_nominal_rate(headset: dict) -> float:
    """Return Cortex's configured EPOC X EEG rate for LSL metadata."""
    settings = headset.get("settings", {})
    try:
        rate = float(settings.get("eegRate", 256))
    except (TypeError, ValueError):
        rate = 256.0
    return rate if rate > 0 else 256.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", default=os.getenv("EMOTIV_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("EMOTIV_CLIENT_SECRET"))
    parser.add_argument("--license", default=os.getenv("EMOTIV_LICENSE_ID"))
    parser.add_argument("--cortex-url", default=DEFAULT_CORTEX_URL)
    parser.add_argument("--headset-id", help="optional Cortex headset id from queryHeadsets")
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=sorted(STREAM_SPECS),
        default=list(DEFAULT_STREAMS),
        help="Cortex streams to bridge to LSL (eeg requires an activated EEG license)",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="verify the Cortex self-signed certificate instead of skipping verification",
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="print incoming quality samples to stderr as they arrive",
    )
    parser.add_argument(
        "--stream-prefix",
        default="Epoc X",
        help="prefix for the published LSL quality stream names (default: Epoc X)",
    )
    args = parser.parse_args()
    if not args.client_id or not args.client_secret:
        parser.error("client credentials are required via --client-id/--client-secret or EMOTIV_CLIENT_ID/EMOTIV_CLIENT_SECRET")
    return args


def config_from_args(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        client_id=args.client_id,
        client_secret=args.client_secret,
        license=args.license,
        cortex_url=args.cortex_url,
        headset_id=args.headset_id,
        streams=tuple(args.streams),
        verify_ssl=args.verify_ssl,
        print_samples=args.print_samples,
        stream_prefix=args.stream_prefix,
    )


def ensure_access(client: CortexClient, client_id: str, client_secret: str) -> None:
    result = client.call(
        "requestAccess",
        {"clientId": client_id, "clientSecret": client_secret},
    )
    if result.get("accessGranted"):
        return
    raise CortexError(
        "Cortex access is not granted. Open EMOTIV Launcher, approve the application, and rerun."
    )


def authorize(client: CortexClient, client_id: str, client_secret: str, license_id: str | None) -> str:
    params = {"clientId": client_id, "clientSecret": client_secret}
    if license_id:
        params["license"] = license_id
    result = client.call("authorize", params)
    warning = result.get("warning")
    if warning:
        print(f"authorize warning: {warning.get('message')}", file=sys.stderr, flush=True)
    return result["cortexToken"]


def get_logged_in_user(client: CortexClient) -> None:
    result = client.call("getUserLogin")
    if not result:
        raise CortexError("No EmotivID user is logged in via EMOTIV Launcher.")
    user = result[0]
    print(
        f"Logged in as {user['username']} on OS user {user['currentOSUsername']}",
        file=sys.stderr,
        flush=True,
    )


def query_headsets(client: CortexClient) -> list[dict]:
    result = client.call("queryHeadsets")
    return result if isinstance(result, list) else []


def print_headsets(headsets: list[dict]) -> None:
    for index, headset in enumerate(headsets):
        print(
            "INFO| cortex-headset "
            f"'{index}' (id={headset.get('id')} status={headset.get('status')} "
            f"connectedBy={headset.get('connectedBy')} mode={headset.get('settings', {}).get('mode')} "
            f"eegRate={headset.get('settings', {}).get('eegRate')})",
            file=sys.stderr,
            flush=True,
        )


def wait_for_connected_headset(client: CortexClient, headset_id: str | None) -> dict:
    client.call("controlDevice", {"command": "refresh"})
    deadline = time.time() + 15

    while time.time() < deadline:
        headsets = query_headsets(client)
        if headsets:
            print_headsets(headsets)

        if headset_id:
            matches = [headset for headset in headsets if headset.get("id") == headset_id]
        else:
            matches = headsets

        for headset in matches:
            if headset.get("status") == "connected":
                return headset

        for headset in matches:
            if headset.get("status") == "discovered":
                print(f"Connecting headset {headset['id']}...", file=sys.stderr, flush=True)
                client.call("controlDevice", {"command": "connect", "headset": headset["id"]})
                break

        time.sleep(POLL_INTERVAL_SECONDS)

    raise CortexError("No connected headset available in Cortex.")


def open_session(client: CortexClient, token: str, headset_id: str, status: str = "open") -> str:
    result = client.call(
        "createSession",
        {
            "cortexToken": token,
            "headset": headset_id,
            "status": status,
        },
    )
    return result["id"]


def close_session(client: CortexClient, token: str, session_id: str) -> None:
    try:
        client.call(
            "updateSession",
            {
                "cortexToken": token,
                "session": session_id,
                "status": "close",
            },
        )
    except CortexError as exc:
        print(f"close_session warning: {exc}", file=sys.stderr, flush=True)


def subscribe_streams(client: CortexClient, token: str, session_id: str, streams: list[str]) -> dict[str, list[str]]:
    result = client.call(
        "subscribe",
        {
            "cortexToken": token,
            "session": session_id,
            "streams": streams,
        },
    )

    failures = result.get("failure", [])
    if failures:
        messages = ", ".join(
            f"{failure['streamName']}: {failure['code']} {failure['message']}"
            for failure in failures
        )
        raise CortexError(f"subscribe failed: {messages}")

    subscriptions: dict[str, list[str]] = {}
    for success in result.get("success", []):
        subscriptions[success["streamName"]] = success["cols"]
    return subscriptions


def run_bridge(config: BridgeConfig) -> None:
    client = CortexClient(config.cortex_url, verify_ssl=config.verify_ssl)
    session_id = None
    token = None
    # Cortex timestamps are Unix epoch seconds; LSL expects local_clock() timebase.
    cortex_to_lsl_offset = local_clock() - time.time()

    try:
        get_logged_in_user(client)
        ensure_access(client, config.client_id, config.client_secret)
        token = authorize(client, config.client_id, config.client_secret, config.license)

        headset = wait_for_connected_headset(client, config.headset_id)
        print(f"Using headset {headset['id']}", file=sys.stderr, flush=True)

        # Cortex requires an activated (licensed) session for raw EEG.  The
        # existing quality-only default stays unactivated so it remains usable
        # with a free account.
        session_status = "active" if "eeg" in config.streams else "open"
        session_id = open_session(client, token, headset["id"], session_status)
        subscriptions = subscribe_streams(client, token, session_id, list(config.streams))

        outlets: dict[str, StreamOutlet] = {}
        for stream_name, labels in subscriptions.items():
            spec = STREAM_SPECS[stream_name]
            # MARKERS is an array of marker objects and cannot be represented in
            # the numeric EEG outlet.  All other EEG columns are numeric and
            # retain Cortex's documented order.
            publish_labels = [label for label in labels if not (stream_name == "eeg" and label == "MARKERS")]
            rate = eeg_nominal_rate(headset) if stream_name == "eeg" else None
            lsl_name = stream_lsl_name(spec, config.stream_prefix)
            outlets[stream_name] = create_outlet(
                spec,
                publish_labels,
                nominal_srate=rate,
                lsl_name=lsl_name,
            )
            print(
                f"Publishing {stream_name} as LSL '{lsl_name}' with columns {publish_labels!r}",
                file=sys.stderr,
                flush=True,
            )

        client.settimeout(1.0)
        while True:
            try:
                message = client.recv()
            except WebSocketTimeoutException:
                continue

            for stream_name, outlet in outlets.items():
                if stream_name not in message:
                    continue

                values = convert_stream_sample(
                    stream_name,
                    message[stream_name],
                    subscriptions[stream_name],
                    STREAM_SPECS[stream_name].channel_format,
                )
                sample_timestamp = message.get("time")
                lsl_timestamp = sample_timestamp + cortex_to_lsl_offset if sample_timestamp is not None else None
                if lsl_timestamp is None:
                    outlet.push_sample(values)
                else:
                    outlet.push_sample(values, timestamp=lsl_timestamp)
                if config.print_samples:
                    print(f"{stream_name}: {values}", file=sys.stderr, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        if session_id and token:
            close_session(client, token, session_id)
        client.close()


def bridge() -> None:
    run_bridge(config_from_args(parse_args()))


if __name__ == "__main__":
    bridge()
