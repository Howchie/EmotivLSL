"""Load user-maintained electrode names for the original EPOC Flex.

The Flex HID packet contains a fixed vector of wire positions.  A montage is
therefore metadata layered on top of that vector: it changes the LSL labels,
not the order of the decoded samples.  This module accepts both the simple
mapping file used by this project and the flat mapping object exported by
Emotiv tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WIRE_LABELS = (
    "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH",
    "LJ", "LK", "LL", "LM", "LN", "LO", "LP", "LQ",
    "RA", "RB", "RC", "RD", "RE", "RF", "RG", "RH",
    "RJ", "RK", "RL", "RM", "RN", "RO", "RP", "RQ",
)
_WIRE_LABEL_SET = frozenset(WIRE_LABELS)
_MAPPING_KEYS = ("mapping", "flexMappings", "channels")


@dataclass(frozen=True)
class FlexMontage:
    """Resolved labels in packet order, plus the source mapping metadata."""

    labels: tuple[str, ...]
    mapping: dict[str, str]
    references: dict[str, str]
    name: str
    path: Path


def _mapping_object(document: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(document, dict):
        raise ValueError("Flex mapping JSON must contain an object")

    for key in _MAPPING_KEYS:
        candidate = document.get(key)
        if isinstance(candidate, dict):
            name = document.get("name") or document.get("label") or key
            return candidate, str(name)

    # Emotiv's Launcher output is a flat object containing CMS/DRL and the
    # 32 wires, for example {"CMS": "TP9", "LA": "C3", ...}.
    candidate = {
        key: value
        for key, value in document.items()
        if key in _WIRE_LABEL_SET or key in {"CMS", "DRL"}
    }
    if not candidate:
        raise ValueError(
            "Flex mapping JSON must contain a 'mapping' object or wire labels "
            "such as 'LA' and 'RA'"
        )
    name = document.get("name") or document.get("label") or "flat mapping"
    return candidate, str(name)


def load_flex_montage(path: str | Path) -> FlexMontage:
    """Read a Flex montage and return labels in the fixed HID packet order.

    Missing wires are deliberately retained under their stable wire label.
    This permits a partial configuration while ensuring that no channel is
    silently dropped or re-ordered.  Empty values are treated the same way.
    """

    mapping_path = Path(path).expanduser()
    try:
        document = json.loads(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Flex mapping file not found: {mapping_path}. "
            "Copy epoch_flex_electrodes.json and edit it, or pass --mapping PATH."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Flex mapping JSON in {mapping_path}: {exc}") from exc

    raw_mapping, name = _mapping_object(document)
    if name == "flat mapping":
        name = mapping_path.stem
    unknown = sorted(set(raw_mapping) - (_WIRE_LABEL_SET | {"CMS", "DRL"}))
    if unknown:
        raise ValueError(
            f"Unknown Flex wire label(s) in {mapping_path}: {', '.join(unknown)}"
        )

    normalized: dict[str, str] = {}
    for wire in WIRE_LABELS:
        value = raw_mapping.get(wire)
        if value is None or not str(value).strip():
            normalized[wire] = wire
        else:
            normalized[wire] = str(value).strip()

    references: dict[str, str] = {}
    for reference in ("CMS", "DRL"):
        value = raw_mapping.get(reference)
        if value is not None and str(value).strip():
            references[reference] = str(value).strip()

    return FlexMontage(
        labels=tuple(normalized[wire] for wire in WIRE_LABELS),
        mapping=normalized,
        references=references,
        name=name,
        path=mapping_path,
    )
