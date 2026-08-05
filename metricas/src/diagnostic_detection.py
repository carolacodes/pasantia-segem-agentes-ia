from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from normalization import clean_text


TITLE_PATTERN = re.compile(
    r"\b(?:doctora|doctor|dra|dr|jueza|juez|abogada|abogado|abg|licenciada|licenciado|lic|sra|sr)\.?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DiagnosticConfig:
    enabled: bool = True
    token_sort_threshold: int = 75
    token_set_threshold: int = 80
    partial_ratio_threshold: int = 80
    require_same_document: bool = True
    require_same_label: bool = True
    allow_title_difference: bool = True
    allow_containment: bool = True
    allow_span_overlap: bool = True
    min_containment_chars: int = 6
    min_containment_tokens: int = 2
    min_containment_ratio: float = 0.60
    identifier_labels: tuple[str, ...] = ("dni", "cuit_cuil", "cbu", "cvu")


def diagnostic_config_from_dict(raw: dict[str, Any] | None) -> DiagnosticConfig:
    raw = raw or {}
    return DiagnosticConfig(
        enabled=bool(raw.get("enabled", True)),
        token_sort_threshold=int(raw.get("token_sort_threshold", 75)),
        token_set_threshold=int(raw.get("token_set_threshold", 80)),
        partial_ratio_threshold=int(raw.get("partial_ratio_threshold", 80)),
        require_same_document=bool(raw.get("require_same_document", True)),
        require_same_label=bool(raw.get("require_same_label", True)),
        allow_title_difference=bool(raw.get("allow_title_difference", True)),
        allow_containment=bool(raw.get("allow_containment", True)),
        allow_span_overlap=bool(raw.get("allow_span_overlap", True)),
        min_containment_chars=int(raw.get("min_containment_chars", 6)),
        min_containment_tokens=int(raw.get("min_containment_tokens", 2)),
        min_containment_ratio=float(raw.get("min_containment_ratio", 0.60)),
        identifier_labels=tuple(raw.get("identifier_labels", ("dni", "cuit_cuil", "cbu", "cvu"))),
    )


def _to_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text or not text.lstrip("-").isdigit():
        return None
    return int(text)


def _span(row: dict[str, Any], prefix: str) -> tuple[int, int] | None:
    start = _to_int(row.get(f"span_inicio_{prefix}", ""))
    end = _to_int(row.get(f"span_fin_{prefix}", ""))
    if start is None or end is None or start < 0 or end < start:
        return None
    return start, end


def _span_overlap(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    gold_span = _span(gold, "gold")
    pred_span = _span(pred, "predicho")
    if gold_span is None or pred_span is None:
        return False
    return max(gold_span[0], pred_span[0]) < min(gold_span[1], pred_span[1])


def _norm(value: Any) -> str:
    return clean_text(value).casefold()


def _without_titles(value: Any) -> str:
    text = TITLE_PATTERN.sub(" ", clean_text(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _has_title_difference(gold_value: Any, pred_value: Any) -> bool:
    gold_raw = _norm(gold_value)
    pred_raw = _norm(pred_value)
    gold_stripped = _without_titles(gold_value)
    pred_stripped = _without_titles(pred_value)
    return bool(gold_stripped and pred_stripped and gold_raw != pred_raw and gold_stripped == pred_stripped)


def _tokens(value: Any) -> list[str]:
    return re.findall(r"\w+", _norm(value))


def _identifier_digits(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"(?i)\b(?:dni|dn!|cuit|cuil|cbu|cvu)\b", " ", text)
    return re.sub(r"\D", "", text)


def _containment_stats(gold_value: Any, pred_value: Any) -> tuple[bool, float, int]:
    gold_text = _norm(gold_value)
    pred_text = _norm(pred_value)
    if not gold_text or not pred_text:
        return False, 0.0, 0
    shorter, longer = sorted([gold_text, pred_text], key=len)
    contained = shorter in longer
    gold_tokens = set(_tokens(gold_value))
    pred_tokens = set(_tokens(pred_value))
    common_tokens = len(gold_tokens & pred_tokens)
    ratio = len(shorter) / len(longer) if longer else 0.0
    return contained, ratio, common_tokens


def _valid_containment(gold_value: Any, pred_value: Any, cfg: DiagnosticConfig, token_set: float, partial: float) -> tuple[bool, bool, float, int]:
    contained, ratio, common_tokens = _containment_stats(gold_value, pred_value)
    if not contained:
        return False, False, ratio, common_tokens
    shorter_len = min(len(_norm(gold_value)), len(_norm(pred_value)))
    enough_length = shorter_len >= cfg.min_containment_chars
    enough_tokens = common_tokens >= cfg.min_containment_tokens
    enough_ratio = ratio >= cfg.min_containment_ratio
    enough_similarity = token_set >= cfg.token_set_threshold or partial >= cfg.partial_ratio_threshold
    valid = enough_similarity and (enough_length and (enough_tokens or enough_ratio))
    return valid, True, ratio, common_tokens


def _near_threshold(score: float, threshold: int, window: int = 5) -> bool:
    return threshold - window <= score < threshold


def _single_fragment_match(common_tokens: int, containment_ratio: float, title_difference: bool, identifier_match: bool) -> bool:
    return bool(common_tokens <= 1 and containment_ratio < 0.60 and not title_difference and not identifier_match)


def _candidate_row(gold: dict[str, Any], pred: dict[str, Any], cfg: DiagnosticConfig) -> dict[str, Any] | None:
    if cfg.require_same_document and gold.get("documento") != pred.get("documento"):
        return None
    same_label = gold.get("etiqueta_gold") == pred.get("etiqueta_predicha")
    if cfg.require_same_label and not same_label:
        return None

    gold_value = gold.get("valor_gold", "")
    pred_value = pred.get("valor_predicho", "")
    token_sort = float(fuzz.token_sort_ratio(gold_value, pred_value))
    token_set = float(fuzz.token_set_ratio(gold_value, pred_value))
    partial = float(fuzz.partial_ratio(gold_value, pred_value))
    length_diff = abs(len(clean_text(gold_value)) - len(clean_text(pred_value)))
    gold_label = str(gold.get("etiqueta_gold", ""))
    pred_label = str(pred.get("etiqueta_predicha", ""))
    identifier_gold = _identifier_digits(gold_value) if gold_label in cfg.identifier_labels else ""
    identifier_pred = _identifier_digits(pred_value) if pred_label in cfg.identifier_labels else ""
    identifier_match = bool(identifier_gold and identifier_pred and identifier_gold == identifier_pred)
    if gold_label in cfg.identifier_labels or pred_label in cfg.identifier_labels:
        if not identifier_match:
            return None
        containment = False
        raw_containment = False
        containment_ratio = 0.0
        common_tokens = 0
    else:
        containment, raw_containment, containment_ratio, common_tokens = (
            _valid_containment(gold_value, pred_value, cfg, token_set, partial)
            if cfg.allow_containment
            else (False, False, 0.0, 0)
        )
    overlap = cfg.allow_span_overlap and _span_overlap(gold, pred)
    title_difference = cfg.allow_title_difference and _has_title_difference(gold_value, pred_value)

    reasons = []
    strong_signals = []
    medium_signals = []
    weak_signals = []
    if overlap:
        reasons.append("overlap_span")
        weak_signals.append("overlap_span")
    if containment:
        reasons.append("contencion_textual")
        medium_signals.append("contencion_textual")
    elif raw_containment:
        reasons.append("contencion_textual_debil")
        weak_signals.append("contencion_textual_debil")
    if title_difference:
        reasons.append("diferencia_por_titulo")
        strong_signals.append("diferencia_por_titulo")
    if identifier_match:
        reasons.append("identificador_normalizado_exacto")
        strong_signals.append("identificador_normalizado_exacto")
    if token_sort >= cfg.token_sort_threshold:
        reasons.append("token_sort_ratio")
        medium_signals.append("token_sort_ratio")
    if token_set >= cfg.token_set_threshold:
        reasons.append("token_set_ratio")
        medium_signals.append("token_set_ratio")
    if partial >= cfg.partial_ratio_threshold:
        reasons.append("partial_ratio")
        medium_signals.append("partial_ratio")
    if _near_threshold(token_sort, cfg.token_sort_threshold) or _near_threshold(token_set, cfg.token_set_threshold) or _near_threshold(partial, cfg.partial_ratio_threshold):
        reasons.append("similitud_cercana_umbral")
        weak_signals.append("similitud_cercana_umbral")
    if not reasons:
        return None

    single_fragment = _single_fragment_match(common_tokens, containment_ratio, title_difference, identifier_match)
    if strong_signals and (identifier_match or title_difference or len(strong_signals) + len(medium_signals) >= 2):
        tipo = "detectada_adicional_alta"
        nivel = "alta"
        regla = strong_signals[0]
    elif not single_fragment and (
        len(medium_signals) >= 2 or (containment and (token_set >= cfg.token_set_threshold or partial >= cfg.partial_ratio_threshold))
    ):
        tipo = "detectada_adicional_media"
        nivel = "media"
        regla = medium_signals[0] if medium_signals else "contencion_textual"
    elif not single_fragment and medium_signals and overlap:
        tipo = "detectada_adicional_media"
        nivel = "media"
        regla = medium_signals[0]
    else:
        tipo = "candidata_revision"
        nivel = "revision"
        regla = weak_signals[0] if weak_signals else (medium_signals[0] if medium_signals else reasons[0])
        if set(weak_signals).issubset({"overlap_span", "contencion_textual_debil"}) and not medium_signals and not strong_signals:
            return None

    score = max(token_sort, token_set, partial)
    return {
        "documento": gold.get("documento", ""),
        "modelo": gold.get("modelo", pred.get("modelo", "")),
        "gold_id": gold.get("gold_id", ""),
        "pred_id": pred.get("pred_id", ""),
        "etiqueta_gold": gold.get("etiqueta_gold", ""),
        "valor_gold": gold_value,
        "span_inicio_gold": gold.get("span_inicio_gold", ""),
        "span_fin_gold": gold.get("span_fin_gold", ""),
        "etiqueta_predicha": pred.get("etiqueta_predicha", ""),
        "valor_predicho": pred_value,
        "span_inicio_predicho": pred.get("span_inicio_predicho", ""),
        "span_fin_predicho": pred.get("span_fin_predicho", ""),
        "score_modelo": pred.get("score_modelo", ""),
        "token_sort_ratio": round(token_sort, 2),
        "token_set_ratio": round(token_set, 2),
        "partial_ratio": round(partial, 2),
        "diferencia_longitud": length_diff,
        "hay_contencion": containment,
        "hay_overlap_span": overlap,
        "diferencia_por_titulo": title_difference,
        "nivel_confianza": nivel,
        "regla_principal": regla,
        "cantidad_senales": len(set(strong_signals + medium_signals + weak_signals)),
        "identificador_normalizado_gold": identifier_gold,
        "identificador_normalizado_predicho": identifier_pred,
        "porcentaje_contencion": round(containment_ratio, 4),
        "tokens_coincidentes": common_tokens,
        "motivo_deteccion": " | ".join(reasons),
        "tipo_diagnostico": tipo,
        "estado_gold_diagnostico": "no_encontrada_con_candidato",
        "estado_pred_diagnostico": "extra_asociable",
        "gold_incluida_principal": bool(gold.get("gold_incluida_principal", False)),
        "pred_incluida_principal": bool(pred.get("pred_incluida_principal", False)),
        "_rank_score": score,
        "_same_label": same_label,
    }


def _empty_diagnostic_row(source: dict[str, Any], tipo: str) -> dict[str, Any]:
    is_gold = tipo == "no_encontrada_sin_candidato"
    return {
        "documento": source.get("documento", ""),
        "modelo": source.get("modelo", ""),
        "gold_id": source.get("gold_id", "") if is_gold else "",
        "pred_id": "" if is_gold else source.get("pred_id", ""),
        "etiqueta_gold": source.get("etiqueta_gold", "") if is_gold else "",
        "valor_gold": source.get("valor_gold", "") if is_gold else "",
        "span_inicio_gold": source.get("span_inicio_gold", "") if is_gold else "",
        "span_fin_gold": source.get("span_fin_gold", "") if is_gold else "",
        "etiqueta_predicha": "" if is_gold else source.get("etiqueta_predicha", ""),
        "valor_predicho": "" if is_gold else source.get("valor_predicho", ""),
        "span_inicio_predicho": "" if is_gold else source.get("span_inicio_predicho", ""),
        "span_fin_predicho": "" if is_gold else source.get("span_fin_predicho", ""),
        "score_modelo": "" if is_gold else source.get("score_modelo", ""),
        "token_sort_ratio": "",
        "token_set_ratio": "",
        "partial_ratio": "",
        "diferencia_longitud": "",
        "hay_contencion": False,
        "hay_overlap_span": False,
        "diferencia_por_titulo": False,
        "nivel_confianza": "",
        "regla_principal": "",
        "cantidad_senales": 0,
        "identificador_normalizado_gold": "",
        "identificador_normalizado_predicho": "",
        "porcentaje_contencion": "",
        "tokens_coincidentes": "",
        "motivo_deteccion": "",
        "tipo_diagnostico": tipo,
        "estado_gold_diagnostico": tipo if is_gold else "",
        "estado_pred_diagnostico": "" if is_gold else "extra_real",
        "gold_incluida_principal": bool(source.get("gold_incluida_principal", False)) if is_gold else False,
        "pred_incluida_principal": False if is_gold else bool(source.get("pred_incluida_principal", False)),
    }


def evaluate_diagnostic_detection(detail: pd.DataFrame, cfg: DiagnosticConfig) -> pd.DataFrame:
    if detail.empty or not cfg.enabled:
        return pd.DataFrame()
    required_columns = {"tipo_resultado", "gold_id", "pred_id", "modelo", "documento"}
    if not required_columns.issubset(detail.columns):
        return pd.DataFrame()

    gold_rows = detail[detail["tipo_resultado"].eq("no_encontrada") & detail["gold_id"].astype(str).ne("")]
    pred_rows = detail[detail["tipo_resultado"].eq("extra") & detail["pred_id"].astype(str).ne("")]
    candidates = []
    for gold in gold_rows.to_dict("records"):
        same_model_preds = pred_rows[pred_rows["modelo"] == gold.get("modelo", "")]
        for pred in same_model_preds.to_dict("records"):
            candidate = _candidate_row(gold, pred, cfg)
            if candidate:
                candidates.append(candidate)

    matched_rows = []
    used_gold: set[tuple[str, str]] = set()
    used_pred: set[tuple[str, str]] = set()
    candidates = sorted(
        candidates,
        key=lambda row: (
            bool(row["_same_label"]),
            row["nivel_confianza"] == "alta",
            row["nivel_confianza"] == "media",
            float(row["_rank_score"]),
            bool(row["hay_overlap_span"]),
            -int(row["diferencia_longitud"]),
        ),
        reverse=True,
    )
    for candidate in candidates:
        gold_key = (str(candidate["modelo"]), str(candidate["gold_id"]))
        pred_key = (str(candidate["modelo"]), str(candidate["pred_id"]))
        if gold_key in used_gold or pred_key in used_pred:
            continue
        used_gold.add(gold_key)
        used_pred.add(pred_key)
        matched_rows.append({key: value for key, value in candidate.items() if not key.startswith("_")})

    rows = list(matched_rows)
    for gold in gold_rows.to_dict("records"):
        gold_key = (str(gold.get("modelo", "")), str(gold.get("gold_id", "")))
        if gold_key not in used_gold:
            rows.append(_empty_diagnostic_row(gold, "no_encontrada_sin_candidato"))
    for pred in pred_rows.to_dict("records"):
        pred_key = (str(pred.get("modelo", "")), str(pred.get("pred_id", "")))
        if pred_key not in used_pred:
            rows.append(_empty_diagnostic_row(pred, "extra_real"))

    return pd.DataFrame(rows)


def _scope_gold_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    has_gold = df["gold_id"].astype(str).ne("")
    if scope == "principal":
        return has_gold & df["gold_incluida_principal"].astype(bool)
    if scope == "opcional":
        return has_gold & ~df["gold_incluida_principal"].astype(bool)
    if scope == "total":
        return has_gold
    raise ValueError(f"Universo diagnostico no soportado: {scope}")


def _scope_diag_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    has_gold = df["gold_id"].astype(str).ne("")
    has_pred = df["pred_id"].astype(str).ne("")
    if scope == "principal":
        return (has_gold & df["gold_incluida_principal"].astype(bool)) | (~has_gold & has_pred & df["pred_incluida_principal"].astype(bool))
    if scope == "opcional":
        return (has_gold & ~df["gold_incluida_principal"].astype(bool)) | (~has_gold & has_pred & ~df["pred_incluida_principal"].astype(bool))
    if scope == "total":
        return has_gold | has_pred
    raise ValueError(f"Universo diagnostico no soportado: {scope}")


def _scope_pred_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    has_pred = df["pred_id"].astype(str).ne("")
    if scope == "principal":
        return has_pred & df["pred_incluida_principal"].astype(bool)
    if scope == "opcional":
        return has_pred & ~df["pred_incluida_principal"].astype(bool)
    if scope == "total":
        return has_pred
    raise ValueError(f"Universo diagnostico no soportado: {scope}")


def filter_diagnostics_by_scope(diagnostics: pd.DataFrame, scope: str) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    return diagnostics[_scope_diag_mask(diagnostics, scope)].reset_index(drop=True)


def summarize_diagnostic_detection(detail: pd.DataFrame, diagnostics: pd.DataFrame, metrics_model: pd.DataFrame | None = None, scope: str = "principal") -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    suffix = {"principal": "principales", "opcional": "opcionales", "total": "total"}[scope]
    rows = []
    for model, group in detail.groupby("modelo"):
        gold_mask = _scope_gold_mask(group, scope)
        official_detected = int((gold_mask & group["tipo_resultado"].isin(["exacta_span", "exacta_valor", "parcial"])).sum())
        official_missing = int((gold_mask & group["tipo_resultado"].eq("no_encontrada")).sum())
        total_gold = int(gold_mask.sum())
        diag_model = diagnostics[diagnostics["modelo"] == model] if not diagnostics.empty else pd.DataFrame()
        diag_model = filter_diagnostics_by_scope(diag_model, scope) if not diag_model.empty else diag_model
        alta = int((diag_model["tipo_diagnostico"] == "detectada_adicional_alta").sum()) if not diag_model.empty else 0
        media = int((diag_model["tipo_diagnostico"] == "detectada_adicional_media").sum()) if not diag_model.empty else 0
        revision = int((diag_model["tipo_diagnostico"] == "candidata_revision").sum()) if not diag_model.empty else 0
        sin_candidato = int((diag_model["tipo_diagnostico"] == "no_encontrada_sin_candidato").sum()) if not diag_model.empty else 0
        total_wide = official_detected + alta + media
        extras_official = int((group["tipo_resultado"].eq("extra") & _scope_pred_mask(group, scope)).sum())
        extras_asociables = alta + media + revision
        extras_reales = int((diag_model["tipo_diagnostico"] == "extra_real").sum()) if not diag_model.empty else 0
        rows.append(
            {
                "modelo": model,
                f"total_gold_{scope}": total_gold,
                f"detectadas_oficiales_{suffix}": official_detected,
                f"no_encontradas_oficiales_{suffix}": official_missing,
                "detectadas_adicionales_alta": alta,
                "detectadas_adicionales_media": media,
                "candidatas_revision": revision,
                "total_detectadas_amplias_confiables": total_wide,
                "porcentaje_detectado_amplio_confiable": round(total_wide / total_gold, 4) if total_gold else 0.0,
                f"no_encontradas_sin_candidato_{suffix}": sin_candidato,
                "extras_oficiales": extras_official,
                "extras_asociables": extras_asociables,
                "extras_reales": extras_reales,
                "invariante_no_encontradas_ok": official_missing == alta + media + revision + sin_candidato,
                "invariante_extras_ok": extras_official == extras_asociables + extras_reales,
            }
        )
    return pd.DataFrame(rows)


def diagnostic_invariants(detail: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ["principal", "opcional", "total"]:
        summary = summarize_diagnostic_detection(detail, diagnostics, scope=scope)
        if summary.empty:
            continue
        suffix = {"principal": "principales", "opcional": "opcionales", "total": "total"}[scope]
        missing_col = f"no_encontradas_oficiales_{suffix}"
        no_candidate_col = f"no_encontradas_sin_candidato_{suffix}"
        for _, row in summary.iterrows():
            diagnostic_missing = (
                int(row["detectadas_adicionales_alta"])
                + int(row["detectadas_adicionales_media"])
                + int(row["candidatas_revision"])
                + int(row[no_candidate_col])
            )
            diagnostic_extras = int(row["extras_asociables"]) + int(row["extras_reales"])
            rows.append(
                {
                    "modelo": row["modelo"],
                    "universo": scope,
                    "no_encontradas_oficiales": int(row[missing_col]),
                    "no_encontradas_diagnosticas_clasificadas": diagnostic_missing,
                    "invariante_no_encontradas_ok": int(row[missing_col]) == diagnostic_missing,
                    "extras_oficiales": int(row["extras_oficiales"]),
                    "extras_diagnosticas_clasificadas": diagnostic_extras,
                    "invariante_extras_ok": int(row["extras_oficiales"]) == diagnostic_extras,
                }
            )
    return pd.DataFrame(rows)


def diagnostic_reports(diagnostics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if diagnostics.empty:
        return {
            "no_encontradas_con_candidato.csv": diagnostics,
            "candidatas_revision.csv": diagnostics,
            "no_encontradas_sin_candidato.csv": diagnostics,
            "extras_asociables.csv": diagnostics,
            "extras_reales.csv": diagnostics,
        }
    associable = diagnostics[diagnostics["tipo_diagnostico"].isin(["detectada_adicional_alta", "detectada_adicional_media", "candidata_revision"])]
    return {
        "no_encontradas_con_candidato.csv": associable,
        "candidatas_revision.csv": diagnostics[diagnostics["tipo_diagnostico"] == "candidata_revision"],
        "no_encontradas_sin_candidato.csv": diagnostics[diagnostics["tipo_diagnostico"] == "no_encontrada_sin_candidato"],
        "extras_asociables.csv": associable,
        "extras_reales.csv": diagnostics[diagnostics["tipo_diagnostico"] == "extra_real"],
    }
