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


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[\s.\-_/]", "", clean_text(value)).casefold()


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
        return normalize_identifier(value)
    return normalize_text_value(value)


def values_equivalent(label: str, left: Any, right: Any, numeric_labels: set[str]) -> bool:
    left_normalized = normalize_value(label, left, numeric_labels)
    right_normalized = normalize_value(label, right, numeric_labels)
    return bool(left_normalized and right_normalized and left_normalized == right_normalized)
