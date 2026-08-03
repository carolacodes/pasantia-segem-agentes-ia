from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from normalization import normalize_value


PATTERNS = {
    "dni": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}\b"),
    "cuit_cuil": re.compile(r"\b\d{2}[-\s]?\d{8}[-\s]?\d\b"),
    "cbu": re.compile(r"\b\d{22}\b"),
    "cvu": re.compile(r"\b\d{22}\b"),
    "monto": re.compile(r"(?:\$|ARS|USD)?\s*-?\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?|-?\d+(?:[,.]\d{2})"),
}


def extract_regex_values(text: str, label: str) -> list[str]:
    pattern = PATTERNS.get(label)
    if not pattern:
        return []
    return [match.group(0).strip() for match in pattern.finditer(text or "")]


def compare_models_vs_regex(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    regex_labels: Iterable[str],
    numeric_labels: set[str],
) -> pd.DataFrame:
    rows = []
    labels = set(regex_labels)
    text_by_doc = gold.groupby("documento")["texto_documento"].first().to_dict()
    all_docs = sorted(set(gold["documento"]) | set(predictions["documento"]))
    models = sorted(predictions["modelo"].unique()) if not predictions.empty else []

    for doc in all_docs:
        text = text_by_doc.get(doc, "")
        for label in sorted(labels):
            gold_values = gold[(gold["documento"] == doc) & (gold["etiqueta"] == label)]["valor"].tolist()
            regex_values = extract_regex_values(text, label)
            normalized_gold = {normalize_value(label, value, numeric_labels) for value in gold_values}
            normalized_regex = {normalize_value(label, value, numeric_labels) for value in regex_values}
            for model in models:
                model_values = predictions[
                    (predictions["documento"] == doc)
                    & (predictions["modelo"] == model)
                    & (predictions["etiqueta"] == label)
                ]["valor"].tolist()
                normalized_model = {normalize_value(label, value, numeric_labels) for value in model_values}
                rows.append(
                    {
                        "documento": doc,
                        "etiqueta": label,
                        "modelo": model,
                        "valor_gold": " | ".join(gold_values),
                        "valor_regex": " | ".join(regex_values),
                        "valor_modelo": " | ".join(model_values),
                        "regex_detecta_gold": bool(normalized_gold & normalized_regex),
                        "modelo_detecta_gold": bool(normalized_gold & normalized_model),
                    }
                )
    return pd.DataFrame(rows)

