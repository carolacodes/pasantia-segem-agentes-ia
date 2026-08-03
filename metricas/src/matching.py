from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from normalization import clean_text, normalize_value, values_equivalent


@dataclass(frozen=True)
class MatchConfig:
    threshold: int
    length_tolerance: int
    numeric_labels: set[str]
    optional_labels: set[str]
    fuzzy_labels: set[str]


def _to_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text or not text.lstrip("-").isdigit():
        return None
    return int(text)


def _span(row: dict[str, Any]) -> tuple[int, int] | None:
    start = _to_int(row.get("span_inicio", ""))
    end = _to_int(row.get("span_fin", ""))
    if start is None or end is None or start < 0 or end < start:
        return None
    return start, end


def _same_span(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_span = _span(left)
    right_span = _span(right)
    return left_span is not None and left_span == right_span


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_span = _span(left)
    right_span = _span(right)
    if left_span is None or right_span is None:
        return False
    return max(left_span[0], right_span[0]) < min(left_span[1], right_span[1])


def _detail(
    documento: str,
    modelo: str,
    tipo: str,
    cfg: MatchConfig,
    gold: dict[str, Any] | None,
    pred: dict[str, Any] | None,
    score_rf: float = 0,
    subtipo: str = "",
    pred_original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_label = gold.get("etiqueta", "") if gold else ""
    pred_label = pred.get("etiqueta", "") if pred else ""
    metodo = "rapidfuzz" if tipo == "parcial" else (tipo if tipo else "")
    return {
        "documento": documento,
        "modelo": modelo,
        "gold_id": gold.get("entidad_id", "") if gold else "",
        "pred_id": pred.get("entidad_id", "") if pred else "",
        "etiqueta_gold": gold_label,
        "valor_gold": gold.get("valor", "") if gold else "",
        "etiqueta_predicha": pred_label,
        "etiqueta_predicha_original": pred.get("etiqueta_original", "") if pred else "",
        "valor_predicho": pred.get("valor", "") if pred else "",
        "score_modelo": pred.get("score_modelo", "") if pred else "",
        "score_rapidfuzz": round(float(score_rf), 2) if tipo == "parcial" and score_rf else "",
        "metodo_matching": metodo,
        "span_inicio_gold": gold.get("span_inicio", "") if gold else "",
        "span_fin_gold": gold.get("span_fin", "") if gold else "",
        "span_inicio_predicho": pred.get("span_inicio", "") if pred else "",
        "span_fin_predicho": pred.get("span_fin", "") if pred else "",
        "tipo_resultado": tipo,
        "subtipo_resultado": subtipo,
        "pred_id_original": pred_original.get("entidad_id", "") if pred_original else "",
        "etiqueta_original_conservada": pred_original.get("etiqueta", "") if pred_original else "",
        "valor_original_conservado": pred_original.get("valor", "") if pred_original else "",
        "span_inicio_original": pred_original.get("span_inicio", "") if pred_original else "",
        "span_fin_original": pred_original.get("span_fin", "") if pred_original else "",
        "pred_id_duplicada": pred.get("entidad_id", "") if pred and tipo == "duplicada" else "",
        "valor_duplicado": pred.get("valor", "") if pred and tipo == "duplicada" else "",
        "span_inicio_duplicado": pred.get("span_inicio", "") if pred and tipo == "duplicada" else "",
        "span_fin_duplicado": pred.get("span_fin", "") if pred and tipo == "duplicada" else "",
        "gold_opcional": bool(gold_label in cfg.optional_labels) if gold_label else False,
        "pred_opcional": bool(pred_label in cfg.optional_labels) if pred_label else False,
        "gold_incluida_principal": bool(gold_label and gold_label not in cfg.optional_labels),
        "pred_incluida_principal": bool(pred_label and pred_label not in cfg.optional_labels),
        "etiqueta_opcional": bool((gold_label in cfg.optional_labels) or (pred_label in cfg.optional_labels)),
    }


def _is_partial(gold: dict[str, Any], pred: dict[str, Any], cfg: MatchConfig) -> tuple[bool, float]:
    label = gold["etiqueta"]
    if label != pred["etiqueta"] or label not in cfg.fuzzy_labels:
        return False, 0
    if label in cfg.numeric_labels:
        return False, 0
    left = gold["valor"]
    right = pred["valor"]
    if abs(len(left) - len(right)) > cfg.length_tolerance:
        return False, 0
    score = fuzz.token_sort_ratio(left, right)
    return score >= cfg.threshold, score


def mark_duplicates(predictions: pd.DataFrame, cfg: MatchConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if predictions.empty:
        return predictions.copy(), []
    rows: list[dict[str, Any]] = []
    duplicate_details: list[dict[str, Any]] = []
    kept_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for row in predictions.to_dict("records"):
        normalized = normalize_value(row["etiqueta"], row["valor"], cfg.numeric_labels)
        if not normalized:
            normalized = f"raw:{clean_text(row['valor']).casefold()}"
        key = (
            row["documento"],
            row["etiqueta"],
            normalized,
        )
        kept = kept_by_key.setdefault(key, [])
        duplicate_subtype = ""
        duplicate_original = None
        for previous in kept:
            if _same_span(previous, row):
                duplicate_subtype = "duplicado_mismo_valor_mismo_span"
                duplicate_original = previous
                break
            if _overlap(previous, row):
                duplicate_subtype = "duplicado_overlap_chunks"
                duplicate_original = previous
                break
            if _span(previous) is None or _span(row) is None:
                duplicate_subtype = "duplicado_dudoso_sin_span"
                duplicate_original = previous
                break
        if duplicate_subtype:
            duplicate_details.append(
                _detail(
                    row["documento"],
                    row["modelo"],
                    "duplicada",
                    cfg,
                    None,
                    row,
                    subtipo=duplicate_subtype,
                    pred_original=duplicate_original,
                )
            )
        else:
            kept.append(row)
            rows.append(row)

    return pd.DataFrame(rows), duplicate_details


def _same_value_any_label(gold: dict[str, Any], pred: dict[str, Any], cfg: MatchConfig) -> bool:
    if values_equivalent(gold["etiqueta"], gold["valor"], pred["valor"], cfg.numeric_labels):
        return True
    gold_as_pred = normalize_value(pred["etiqueta"], gold["valor"], cfg.numeric_labels)
    pred_as_pred = normalize_value(pred["etiqueta"], pred["valor"], cfg.numeric_labels)
    return bool(gold_as_pred and pred_as_pred and gold_as_pred == pred_as_pred)


def compare_model(gold: pd.DataFrame, predictions: pd.DataFrame, cfg: MatchConfig, model_name: str) -> pd.DataFrame:
    predictions, duplicate_details = mark_duplicates(predictions, cfg)
    details: list[dict[str, Any]] = list(duplicate_details)

    pred_docs = set(predictions["documento"]) if "documento" in predictions.columns else set()
    all_docs = sorted(set(gold["documento"]) | pred_docs)
    for doc in all_docs:
        gold_doc = gold[gold["documento"] == doc].to_dict("records")
        pred_doc = predictions[predictions["documento"] == doc].to_dict("records")
        used_preds: set[int] = set()

        for gold_row in gold_doc:
            best_idx = None
            best_score = -1.0
            best_type = ""
            best_subtype = ""

            for idx, pred_row in enumerate(pred_doc):
                if idx in used_preds:
                    continue
                same_label = gold_row["etiqueta"] == pred_row["etiqueta"]
                same_value = values_equivalent(gold_row["etiqueta"], gold_row["valor"], pred_row["valor"], cfg.numeric_labels)
                if same_label and same_value and _same_span(gold_row, pred_row):
                    best_idx = idx
                    best_score = 100.0
                    best_type = "exacta_span"
                    break

            if best_idx is None:
                for idx, pred_row in enumerate(pred_doc):
                    if idx in used_preds:
                        continue
                    same_label = gold_row["etiqueta"] == pred_row["etiqueta"]
                    same_value = values_equivalent(gold_row["etiqueta"], gold_row["valor"], pred_row["valor"], cfg.numeric_labels)
                    if same_label and same_value:
                        best_idx = idx
                        best_score = 100.0
                        best_type = "exacta_valor"
                        best_subtype = "span_diferente_o_no_disponible"
                        break

            if best_idx is None:
                for idx, pred_row in enumerate(pred_doc):
                    if idx in used_preds:
                        continue
                    if gold_row["etiqueta"] != pred_row["etiqueta"] and _same_value_any_label(gold_row, pred_row, cfg):
                        best_idx = idx
                        best_score = 100.0
                        best_type = "etiqueta_incorrecta"
                        break

            if best_idx is None:
                for idx, pred_row in enumerate(pred_doc):
                    if idx in used_preds:
                        continue
                    ok, score = _is_partial(gold_row, pred_row, cfg)
                    if ok and score > best_score:
                        best_idx = idx
                        best_score = score
                        best_type = "parcial"

            if best_idx is None:
                details.append(_detail(doc, model_name, "no_encontrada", cfg, gold_row, None))
            else:
                used_preds.add(best_idx)
                pred_row = pred_doc[best_idx]
                details.append(_detail(doc, model_name, best_type, cfg, gold_row, pred_row, best_score, best_subtype))

        for idx, pred_row in enumerate(pred_doc):
            if idx not in used_preds:
                details.append(_detail(doc, model_name, "extra", cfg, None, pred_row))

    return pd.DataFrame(details)
