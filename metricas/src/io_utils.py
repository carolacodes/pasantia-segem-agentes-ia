from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from normalization import clean_text, is_empty, normalize_label


def detect_delimiter(path: str | Path) -> str:
    raw = Path(path).read_bytes()[:8192]
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        return csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")
    delimiter = detect_delimiter(csv_path)
    return pd.read_csv(csv_path, sep=delimiter, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def choose_document_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        "No se encontro columna identificadora de documento. "
        f"Candidatas configuradas: {candidates}. Columnas disponibles: {list(df.columns)}"
    )


def validate_entity_columns(df: pd.DataFrame, columns: dict[str, Any], require_score: bool = False) -> None:
    required = [columns["label"], columns["value"]]
    if require_score:
        required.append(columns["score"])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas {missing}. Columnas disponibles: {list(df.columns)}")


def validate_spans(df: pd.DataFrame, columns: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    start_col = columns.get("span_start")
    end_col = columns.get("span_end")
    if start_col not in df.columns or end_col not in df.columns:
        return errors
    for idx, row in df[[start_col, end_col]].iterrows():
        start = clean_text(row[start_col])
        end = clean_text(row[end_col])
        if not start and not end:
            continue
        if not re.fullmatch(r"-?\d+", start) or not re.fullmatch(r"-?\d+", end):
            errors.append(f"Fila {idx}: span no numerico ({start!r}, {end!r})")
            continue
        if int(start) < 0 or int(end) < int(start):
            errors.append(f"Fila {idx}: span invalido ({start}, {end})")
    return errors


def canonical_label(label: str, aliases: dict[str, str]) -> str:
    normalized = normalize_label(label)
    return normalize_label(aliases.get(normalized, normalized))


def prepare_entities(df: pd.DataFrame, config: dict[str, Any], source: str, model: str | None = None) -> pd.DataFrame:
    columns = config["columns"]
    validate_entity_columns(df, columns, require_score=False)
    doc_col = choose_document_column(df, columns["document_id_candidates"])
    text_col = columns.get("text")
    score_col = columns.get("score")
    span_start_col = columns.get("span_start")
    span_end_col = columns.get("span_end")

    prepared = pd.DataFrame()
    aliases = config.get("matching", {}).get("label_aliases", {})
    prepared["entidad_id"] = [f"{source}_{idx}" for idx in df.index]
    prepared["documento"] = df[doc_col].map(clean_text)
    prepared["etiqueta_original"] = df[columns["label"]].map(normalize_label)
    prepared["etiqueta"] = prepared["etiqueta_original"].map(lambda label: canonical_label(label, aliases))
    prepared["valor"] = df[columns["value"]].map(clean_text)
    prepared["texto_documento"] = df[text_col].map(clean_text) if text_col in df.columns else ""
    prepared["score_modelo"] = df[score_col].map(clean_text) if score_col in df.columns else ""
    prepared["span_inicio"] = df[span_start_col].map(clean_text) if span_start_col in df.columns else ""
    prepared["span_fin"] = df[span_end_col].map(clean_text) if span_end_col in df.columns else ""
    prepared["fuente"] = source
    prepared["modelo"] = model or source
    prepared = prepared[~prepared["documento"].map(is_empty)]
    prepared = prepared[~prepared["etiqueta"].map(is_empty)]
    prepared = prepared[~prepared["valor"].map(is_empty)]
    return prepared.reset_index(drop=True)


def build_gold_audit(raw_df: pd.DataFrame, config: dict[str, Any], optional_labels: set[str]) -> pd.DataFrame:
    columns = config["columns"]
    doc_col = choose_document_column(raw_df, columns["document_id_candidates"])
    aliases = config.get("matching", {}).get("label_aliases", {})
    rows = []
    for idx, row in raw_df.iterrows():
        documento = clean_text(row.get(doc_col, ""))
        etiqueta_original = normalize_label(row.get(columns["label"], ""))
        etiqueta = canonical_label(etiqueta_original, aliases)
        valor = clean_text(row.get(columns["value"], ""))
        motivos = []
        if is_empty(documento):
            motivos.append("documento_vacio")
        if is_empty(etiqueta):
            motivos.append("etiqueta_vacia")
        if is_empty(valor):
            motivos.append("valor_vacio")
        valida = not motivos
        es_opcional = etiqueta in optional_labels if valida else False
        incluida = valida and not es_opcional
        if valida and es_opcional:
            motivos.append("etiqueta_opcional_no_penaliza_f1_principal")
        rows.append(
            {
                "fila_original": idx,
                "documento": documento,
                "etiqueta": etiqueta,
                "etiqueta_original": etiqueta_original,
                "valor": valor,
                "incluida": incluida,
                "categoria": "obligatoria" if incluida else ("opcional" if es_opcional else "excluida"),
                "motivo_exclusion": " | ".join(motivos),
            }
        )
    return pd.DataFrame(rows)


def infer_model_name(path: str | Path) -> str:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) > 3 and all(part.isdigit() for part in parts[:3]):
        return "_".join(parts[3:])
    return stem
