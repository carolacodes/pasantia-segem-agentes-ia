from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


EMPTY_VALUES = {"", "nan", "none", "null", "nat"}


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().casefold() in EMPTY_VALUES


def clean_text(value: Any) -> str:
    if is_empty(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_label(label: Any) -> str:
    return clean_text(label).casefold()


def normalize_text_value(value: Any) -> str:
    return clean_text(value).casefold()


IDENTIFIER_PREFIXES = {
    "dni": ("dni",),
    "cuit_cuil": ("cuit", "cuil"),
    "cbu": ("cbu",),
    "cvu": ("cvu",),
}

IDENTIFIER_LENGTHS = {
    "dni": {7, 8},
    "cuit_cuil": {11},
    "cbu": {22},
    "cvu": {22},
}

IDENTIFIER_ABBREVIATIONS = {
    "dni": ("dni", r"d\s*\.?\s*n\s*\.?\s*i\s*\.?"),
    "cuit": ("cuit", r"c\s*\.?\s*u\s*\.?\s*i\s*\.?\s*t\s*\.?"),
    "cuil": ("cuil", r"c\s*\.?\s*u\s*\.?\s*i\s*\.?\s*l\s*\.?"),
    "cbu": ("cbu", r"c\s*\.?\s*b\s*\.?\s*u\s*\.?"),
    "cvu": ("cvu", r"c\s*\.?\s*v\s*\.?\s*u\s*\.?"),
}

NUMBER_MARKER_PATTERN = r"(?:n\s*[°ºo]\.?|nro\.?|numero|número|num\.?)"


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _contains_other_identifier_prefix(text: str, label: str) -> bool:
    allowed = set(IDENTIFIER_PREFIXES.get(label, ()))
    for prefixes in IDENTIFIER_PREFIXES.values():
        for prefix in prefixes:
            if prefix not in allowed and re.search(rf"\b{prefix}\b", text, flags=re.IGNORECASE):
                return True
    return False


def _canonical_identifier_text(value: str) -> str:
    text = value
    for canonical, pattern in IDENTIFIER_ABBREVIATIONS.values():
        text = re.sub(pattern, f" {canonical} ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_identifier(label: str, value: Any) -> str:
    label = normalize_label(label)
    text = _canonical_identifier_text(clean_text(value))
    if not text:
        return ""
    if label not in IDENTIFIER_PREFIXES:
        return re.sub(r"[\s.\-_/]", "", text).casefold()
    if _contains_other_identifier_prefix(text, label):
        return ""

    allowed_prefixes = IDENTIFIER_PREFIXES[label]
    allowed_lengths = IDENTIFIER_LENGTHS[label]
    prefix_pattern = "|".join(re.escape(prefix) for prefix in allowed_prefixes)
    candidates: list[str] = []

    for match in re.finditer(
        rf"\b(?:{prefix_pattern})\b\s*(?:{NUMBER_MARKER_PATTERN})?\s*:?\s*([0-9][0-9\s.\-_/]*[0-9])",
        text,
        flags=re.IGNORECASE,
    ):
        digits = _digits_only(match.group(1))
        if len(digits) in allowed_lengths:
            candidates.append(digits)

    without_allowed_prefixes = re.sub(
        rf"\b(?:{prefix_pattern})\b\s*(?:{NUMBER_MARKER_PATTERN})?\s*:?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    if not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", without_allowed_prefixes):
        digits = _digits_only(without_allowed_prefixes)
        if len(digits) in allowed_lengths:
            candidates.append(digits)

    unique_candidates = sorted(set(candidates))
    return unique_candidates[0] if len(unique_candidates) == 1 else ""


def normalize_amount(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("\u00a0", " ").strip()
    text = re.sub(r"(?i)\b(ars|usd|pesos?)\b", "", text)
    text = text.replace("$", "").strip()
    if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", text):
        return ""
    if re.fullmatch(r"\d{2}-\d{8}-\d", text):
        return ""
    text = re.sub(r"\s+", "", text)
    if "-" in text[1:]:
        return ""
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text:
        return ""
    if not re.search(r"\d", text):
        return ""

    if "," in text and "." in text:
        if text.rfind(",") < text.rfind("."):
            return ""
        integer, decimal = text.rsplit(",", 1)
        if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", integer.lstrip("-")):
            return ""
        if not re.fullmatch(r"\d{1,2}", decimal):
            return ""
        text = integer.replace(".", "") + "." + decimal
    elif "," in text:
        integer, decimal = text.rsplit(",", 1)
        if not integer.lstrip("-").isdigit() or not re.fullmatch(r"\d{1,2}", decimal):
            return ""
        text = integer + "." + decimal
    elif "." in text:
        sign = "-" if text.startswith("-") else ""
        unsigned = text[1:] if sign else text
        parts = unsigned.split(".")
        if len(parts) > 1:
            if len(parts) == 2 and len(parts[1]) in {1, 2} and len(parts[0]) > 3:
                text = sign + parts[0] + "." + parts[1]
            elif len(parts[0]) in {1, 2, 3} and all(len(part) == 3 and part.isdigit() for part in parts[1:]) and parts[0].isdigit():
                text = sign + "".join(parts)
            else:
                return ""
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return ""
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return ""
    return format(amount.normalize(), "f")


def normalize_value(label: str, value: Any, numeric_labels: set[str]) -> str:
    label = normalize_label(label)
    if label == "monto":
        return normalize_amount(value)
    if label in numeric_labels:
        return normalize_identifier(label, value)
    return normalize_text_value(value)


def values_equivalent(label: str, left: Any, right: Any, numeric_labels: set[str]) -> bool:
    left_normalized = normalize_value(label, left, numeric_labels)
    right_normalized = normalize_value(label, right, numeric_labels)
    return bool(left_normalized and right_normalized and left_normalized == right_normalized)
