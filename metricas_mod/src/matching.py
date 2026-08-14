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
    min_span_overlap_ratio: float = 0.30
    tier5_token_set_threshold: int = 60
    tier5_partial_ratio_threshold: int = 70


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
    gold_relacionado: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_label = gold.get("etiqueta", "") if gold else ""
    pred_label = pred.get("etiqueta", "") if pred else ""
    metodo = "rapidfuzz" if tipo == "parcial" else (tipo if tipo else "")
    rel_gold = gold_relacionado or {}
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
        "gold_id_relacionado": rel_gold.get("entidad_id", ""),
        "valor_gold_relacionado": rel_gold.get("valor", ""),
        "span_inicio_gold_relacionado": rel_gold.get("span_inicio", ""),
        "span_fin_gold_relacionado": rel_gold.get("span_fin", ""),
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


def _tier5_textual_evidence(gold: dict[str, Any], pred: dict[str, Any], cfg: MatchConfig) -> tuple[bool, float]:
    gold_value = clean_text(gold["valor"]).casefold()
    pred_value = clean_text(pred["valor"]).casefold()
    token_set = float(fuzz.token_set_ratio(gold_value, pred_value))
    partial = float(fuzz.partial_ratio(gold_value, pred_value))
    return (
        token_set >= cfg.tier5_token_set_threshold
        or partial >= cfg.tier5_partial_ratio_threshold,
        max(token_set, partial),
    )


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
        
        gold_matches: list[tuple[int, str, float, str] | None] = [None] * len(gold_doc)
        used_preds: set[int] = set()

        def get_span_key(g_idx: int, p_idx: int) -> tuple[int, int, int, int, int]:
            g_span = _span(gold_doc[g_idx])
            p_span = _span(pred_doc[p_idx])
            if g_span is None or p_span is None:
                return (1, 0, 0, g_idx, p_idx)
            g_start, g_end = g_span
            p_start, p_end = p_span
            overlap = max(0, min(g_end, p_end) - max(g_start, p_start))
            dist = abs(g_start - p_start)
            return (0, -overlap, dist, g_idx, p_idx)

        # Tier 1 (exacta_span)
        # Find exact value + span + label matches first.
        pass1_pairs = []
        for g_idx, gold_row in enumerate(gold_doc):
            for p_idx, pred_row in enumerate(pred_doc):
                same_label = gold_row["etiqueta"] == pred_row["etiqueta"]
                same_value = values_equivalent(gold_row["etiqueta"], gold_row["valor"], pred_row["valor"], cfg.numeric_labels)
                if same_label and same_value and _same_span(gold_row, pred_row):
                    pass1_pairs.append((g_idx, p_idx))

        for g_idx, p_idx in pass1_pairs:
            if gold_matches[g_idx] is None and p_idx not in used_preds:
                gold_matches[g_idx] = (p_idx, "exacta_span", 100.0, "")
                used_preds.add(p_idx)

        # Tier 2 (exacta_valor)
        # Match remaining entities with identical labels and equivalent values
        pass2_pairs = []
        for g_idx, gold_row in enumerate(gold_doc):
            if gold_matches[g_idx] is not None:
                continue
            for p_idx, pred_row in enumerate(pred_doc):
                if p_idx in used_preds:
                    continue
                same_label = gold_row["etiqueta"] == pred_row["etiqueta"]
                same_value = values_equivalent(gold_row["etiqueta"], gold_row["valor"], pred_row["valor"], cfg.numeric_labels)
                if same_label and same_value:
                    key = get_span_key(g_idx, p_idx)
                    pass2_pairs.append((key, g_idx, p_idx))

        pass2_pairs.sort(key=lambda x: x[0])
        for _, g_idx, p_idx in pass2_pairs:
            if gold_matches[g_idx] is None and p_idx not in used_preds:
                gold_matches[g_idx] = (p_idx, "exacta_valor", 100.0, "span_diferente_o_no_disponible")
                used_preds.add(p_idx)

        # Tier 3 (etiqueta_incorrecta)
        # Match remaining entities with different labels but equivalent values
        pass3_pairs = []
        for g_idx, gold_row in enumerate(gold_doc):
            if gold_matches[g_idx] is not None:
                continue
            for p_idx, pred_row in enumerate(pred_doc):
                if p_idx in used_preds:
                    continue
                diff_label = gold_row["etiqueta"] != pred_row["etiqueta"]
                same_value = _same_value_any_label(gold_row, pred_row, cfg)
                if diff_label and same_value:
                    key = get_span_key(g_idx, p_idx)
                    pass3_pairs.append((key, g_idx, p_idx))

        pass3_pairs.sort(key=lambda x: x[0])
        for _, g_idx, p_idx in pass3_pairs:
            if gold_matches[g_idx] is None and p_idx not in used_preds:
                gold_matches[g_idx] = (p_idx, "etiqueta_incorrecta", 100.0, "")
                used_preds.add(p_idx)

        # Tier 4 (parcial)
        # Match remaining entities with similar values using fuzzy scoring
        pass4_pairs = []
        for g_idx, gold_row in enumerate(gold_doc):
            if gold_matches[g_idx] is not None:
                continue
            for p_idx, pred_row in enumerate(pred_doc):
                if p_idx in used_preds:
                    continue
                ok, score = _is_partial(gold_row, pred_row, cfg)
                if ok:
                    g_span = _span(gold_row)
                    p_span = _span(pred_doc[p_idx])
                    if g_span is None or p_span is None:
                        key = (1, -int(score), 0, 0, g_idx, p_idx)
                    else:
                        g_start, g_end = g_span
                        p_start, p_end = p_span
                        overlap = max(0, min(g_end, p_end) - max(g_start, p_start))
                        dist = abs(g_start - p_start)
                        key = (0, -overlap, -int(score), dist, g_idx, p_idx)
                    pass4_pairs.append((key, g_idx, p_idx, score))

        pass4_pairs.sort(key=lambda x: x[0])
        for _, g_idx, p_idx, score in pass4_pairs:
            if gold_matches[g_idx] is None and p_idx not in used_preds:
                gold_matches[g_idx] = (p_idx, "parcial", score, "")
                used_preds.add(p_idx)

        # Tier 5 (overlap_span)
        # Match remaining entities with same label, sufficient span overlap and textual evidence.
        pass5_pairs = []
        for g_idx, gold_row in enumerate(gold_doc):
            if gold_matches[g_idx] is not None:
                continue
            g_span = _span(gold_row)
            if g_span is None:
                continue
            gold_len = g_span[1] - g_span[0]
            if gold_len <= 0:
                continue
            for p_idx, pred_row in enumerate(pred_doc):
                if p_idx in used_preds:
                    continue
                same_label = gold_row["etiqueta"] == pred_row["etiqueta"]
                if not same_label or not _overlap(gold_row, pred_row):
                    continue
                p_span = _span(pred_row)
                if p_span is None:
                    continue
                overlap_amount = max(0, min(g_span[1], p_span[1]) - max(g_span[0], p_span[0]))
                overlap_ratio = overlap_amount / gold_len
                has_textual_evidence, score = _tier5_textual_evidence(gold_row, pred_row, cfg)
                if overlap_ratio >= cfg.min_span_overlap_ratio and has_textual_evidence:
                    key = (0, -overlap_amount, -int(score), g_idx, p_idx)
                    pass5_pairs.append((key, g_idx, p_idx, score))

        pass5_pairs.sort(key=lambda x: x[0])
        for _, g_idx, p_idx, score in pass5_pairs:
            if gold_matches[g_idx] is None and p_idx not in used_preds:
                gold_matches[g_idx] = (p_idx, "parcial", score, "overlap_span")
                used_preds.add(p_idx)

        # Build final details for this document preserving the original order of gold standard
        for g_idx, match in enumerate(gold_matches):
            if match is None:
                details.append(_detail(doc, model_name, "no_encontrada", cfg, gold_doc[g_idx], None))
            else:
                p_idx, match_type, score, subtype = match
                details.append(_detail(doc, model_name, match_type, cfg, gold_doc[g_idx], pred_doc[p_idx], score, subtype))

        # Append unmatched predictions as extra or extra_fragmento
        for p_idx, pred_row in enumerate(pred_doc):
            if p_idx not in used_preds:
                # Check if this unused prediction overlaps with any matched gold entity in the document
                fragment_of_gold = None
                for g_idx, match in enumerate(gold_matches):
                    if match is not None:
                        gold_row = gold_doc[g_idx]
                        has_textual_evidence, _ = _tier5_textual_evidence(gold_row, pred_row, cfg)
                        if (
                            gold_row["etiqueta"] == pred_row["etiqueta"]
                            and _overlap(gold_row, pred_row)
                            and has_textual_evidence
                        ):
                            fragment_of_gold = gold_row
                            break
                if fragment_of_gold:
                    details.append(
                        _detail(
                            doc,
                            model_name,
                            "extra",
                            cfg,
                            None,
                            pred_row,
                            subtipo="extra_fragmento",
                            gold_relacionado=fragment_of_gold,
                        )
                    )
                else:
                    details.append(_detail(doc, model_name, "extra", cfg, None, pred_row))

    return pd.DataFrame(details)
