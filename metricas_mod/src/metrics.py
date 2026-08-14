from __future__ import annotations

import pandas as pd


RESULT_TYPES = [
    "exacta_span",
    "exacta_valor",
    "parcial",
    "extra",
    "no_encontrada",
    "etiqueta_incorrecta",
    "duplicada",
]
STRICT_CORRECT = {"exacta_span", "exacta_valor"}
RELAXED_CORRECT = {"exacta_span", "exacta_valor", "parcial"}


def _safe_div(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def _empty_metrics(model: str, label: str | None = None) -> dict[str, object]:
    counts = {kind: 0 for kind in RESULT_TYPES}
    return {
        "modelo": model,
        "etiqueta": label or "",
        "total_documentos": 0,
        "total_entidades_gold": 0,
        "total_entidades_predichas": 0,
        "exactas": 0,
        **counts,
        "precision_estricta": 0.0,
        "recall_estricto": 0.0,
        "f1_estricto": 0.0,
        "precision_relajada": 0.0,
        "recall_relajado": 0.0,
        "f1_relajado": 0.0,
        "cobertura": 0.0,
    }


def _row_relevant_for_main(row: pd.Series) -> bool:
    return bool(row.get("gold_incluida_principal", False)) or bool(row.get("pred_incluida_principal", False))


def _metrics_from_counts(
    df: pd.DataFrame,
    model: str,
    label: str | None,
    gold_mask: pd.Series,
    pred_mask: pd.Series,
    count_mask: pd.Series | dict[str, pd.Series],
    correct_mask: pd.Series,
) -> dict[str, object]:
    if isinstance(count_mask, dict):
        count_masks = count_mask
        row_count_mask = pd.Series(False, index=df.index)
        for mask in count_masks.values():
            row_count_mask = row_count_mask | mask
    else:
        count_masks = {kind: count_mask for kind in RESULT_TYPES}
        row_count_mask = count_mask
    counts = {kind: int((count_masks.get(kind, row_count_mask) & (df["tipo_resultado"] == kind)).sum()) for kind in RESULT_TYPES}
    exactas = counts["exacta_span"] + counts["exacta_valor"]
    strict_correct = int((correct_mask & df["tipo_resultado"].isin(STRICT_CORRECT)).sum())
    relaxed_correct = int((correct_mask & df["tipo_resultado"].isin(RELAXED_CORRECT)).sum())
    gold_total = int(gold_mask.sum())
    pred_total = int(pred_mask.sum())
    precision_strict = _safe_div(strict_correct, pred_total)
    recall_strict = _safe_div(strict_correct, gold_total)
    precision_relaxed = _safe_div(relaxed_correct, pred_total)
    recall_relaxed = _safe_div(relaxed_correct, gold_total)
    return {
        "modelo": model,
        "etiqueta": label or "",
        "total_documentos": int(df.loc[row_count_mask, "documento"].nunique()),
        "total_entidades_gold": gold_total,
        "total_entidades_predichas": pred_total,
        "exactas": exactas,
        **counts,
        "precision_estricta": precision_strict,
        "recall_estricto": recall_strict,
        "f1_estricto": _safe_div(2 * precision_strict * recall_strict, precision_strict + recall_strict),
        "precision_relajada": precision_relaxed,
        "recall_relajado": recall_relaxed,
        "f1_relajado": _safe_div(2 * precision_relaxed * recall_relaxed, precision_relaxed + recall_relaxed),
        "cobertura": _safe_div(
            int((gold_mask & df["tipo_resultado"].isin(RELAXED_CORRECT | {"etiqueta_incorrecta"})).sum()),
            gold_total,
        ),
    }


def metrics_by_model(detail: pd.DataFrame, include_optional: bool = False) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for model, group in detail.groupby("modelo"):
        if include_optional:
            gold_mask = group["gold_id"].astype(str).ne("")
            pred_mask = group["pred_id"].astype(str).ne("")
            count_mask = gold_mask | pred_mask
        else:
            gold_mask = group["gold_id"].astype(str).ne("") & group["gold_incluida_principal"].astype(bool)
            pred_mask = group["pred_id"].astype(str).ne("") & group["pred_incluida_principal"].astype(bool)
            count_mask = group.apply(_row_relevant_for_main, axis=1)
        correct_mask = gold_mask & pred_mask
        rows.append(_metrics_from_counts(group, model, None, gold_mask, pred_mask, count_mask, correct_mask))
    return pd.DataFrame(rows)


def _scope_count_masks(group: pd.DataFrame, gold_scope_mask: pd.Series, pred_scope_mask: pd.Series) -> dict[str, pd.Series]:
    has_gold = group["gold_id"].astype(str).ne("")
    has_pred = group["pred_id"].astype(str).ne("")
    by_gold = has_gold & gold_scope_mask
    by_pred_without_gold = (~has_gold) & has_pred & pred_scope_mask
    return {
        "exacta_span": by_gold,
        "exacta_valor": by_gold,
        "parcial": by_gold,
        "no_encontrada": by_gold,
        "etiqueta_incorrecta": by_gold,
        "extra": by_pred_without_gold,
        "duplicada": by_pred_without_gold,
    }


def metrics_by_model_principal_scope(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for model, group in detail.groupby("modelo"):
        gold_scope = group["gold_incluida_principal"].astype(bool)
        pred_scope = group["pred_incluida_principal"].astype(bool)
        gold_mask = group["gold_id"].astype(str).ne("") & gold_scope
        pred_mask = group["pred_id"].astype(str).ne("") & pred_scope
        correct_mask = gold_mask & pred_mask
        rows.append(_metrics_from_counts(group, model, None, gold_mask, pred_mask, _scope_count_masks(group, gold_scope, pred_scope), correct_mask))
    return pd.DataFrame(rows)


def metrics_by_model_optional(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for model, group in detail.groupby("modelo"):
        has_gold = group["gold_id"].astype(str).ne("")
        has_pred = group["pred_id"].astype(str).ne("")
        gold_scope = group["gold_opcional"].astype(bool)
        pred_scope = group["pred_opcional"].astype(bool)
        gold_mask = has_gold & gold_scope
        pred_mask = has_pred & pred_scope
        correct_mask = gold_mask & pred_mask
        rows.append(_metrics_from_counts(group, model, None, gold_mask, pred_mask, _scope_count_masks(group, gold_scope, pred_scope), correct_mask))
    return pd.DataFrame(rows)


def metrics_by_model_total(detail: pd.DataFrame) -> pd.DataFrame:
    return metrics_by_model(detail, include_optional=True)


def metrics_by_label(detail: pd.DataFrame, include_optional: bool = False, optional_labels: set[str] | None = None) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    optional_labels = optional_labels or set()
    labels = sorted(set(detail["etiqueta_gold"]) | set(detail["etiqueta_predicha"]))
    labels = [label for label in labels if label]
    if not include_optional:
        labels = [label for label in labels if label not in optional_labels]

    rows = []
    for model, group in detail.groupby("modelo"):
        for label in labels:
            gold_mask = group["gold_id"].astype(str).ne("") & (group["etiqueta_gold"] == label)
            pred_mask = group["pred_id"].astype(str).ne("") & (group["etiqueta_predicha"] == label)
            if not include_optional and label in optional_labels:
                continue
            count_mask = gold_mask | pred_mask
            if not count_mask.any():
                continue
            correct_mask = gold_mask & pred_mask & (group["etiqueta_predicha"] == label)
            rows.append(_metrics_from_counts(group, model, label, gold_mask, pred_mask, count_mask, correct_mask))
    return pd.DataFrame(rows)


def optional_metrics(detail: pd.DataFrame, optional_labels: set[str]) -> pd.DataFrame:
    metrics = metrics_by_label(detail, include_optional=True, optional_labels=optional_labels)
    if metrics.empty:
        return metrics
    return metrics[metrics["etiqueta"].isin(optional_labels)].reset_index(drop=True)
