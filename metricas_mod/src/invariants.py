from __future__ import annotations

import pandas as pd


def audit_invariants(detail: pd.DataFrame, gold: pd.DataFrame, predictions: pd.DataFrame, metrics_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(detail["modelo"].dropna().unique().tolist()) if not detail.empty else []
    gold_ids = set(gold["entidad_id"].astype(str))
    pred_ids_by_model = {
        model: set(group["entidad_id"].astype(str))
        for model, group in predictions.groupby("modelo")
    } if not predictions.empty else {}

    gold_totals = metrics_model.set_index("modelo")["total_entidades_gold"].to_dict() if not metrics_model.empty else {}
    unique_gold_totals = sorted(set(gold_totals.values()))
    rows.append(
        {
            "invariante": "total_gold_obligatorio_identico_entre_modelos",
            "modelo": "",
            "estado": "ok" if len(unique_gold_totals) <= 1 else "error",
            "detalle": f"totales={gold_totals}",
        }
    )

    valid_gold_results = {"exacta_span", "exacta_valor", "parcial", "etiqueta_incorrecta", "no_encontrada"}
    valid_pred_results = {"exacta_span", "exacta_valor", "parcial", "etiqueta_incorrecta", "extra", "duplicada"}

    for model in models:
        model_detail = detail[detail["modelo"] == model]
        gold_counts = model_detail[model_detail["gold_id"].astype(str).ne("")].groupby("gold_id").size().to_dict()
        missing_gold = sorted(gold_ids - set(gold_counts))
        repeated_gold = sorted(gold_id for gold_id, count in gold_counts.items() if count != 1)
        invalid_gold_status = sorted(
            model_detail[
                model_detail["gold_id"].astype(str).ne("")
                & ~model_detail["tipo_resultado"].isin(valid_gold_results)
            ]["gold_id"].astype(str).unique().tolist()
        )
        rows.append(
            {
                "invariante": "cada_gold_clasificado_exactamente_una_vez",
                "modelo": model,
                "estado": "ok" if not missing_gold and not repeated_gold and not invalid_gold_status else "error",
                "detalle": f"faltantes={len(missing_gold)} repetidos={len(repeated_gold)} estado_invalido={len(invalid_gold_status)}",
            }
        )

        expected_pred_ids = pred_ids_by_model.get(model, set())
        pred_counts = model_detail[model_detail["pred_id"].astype(str).ne("")].groupby("pred_id").size().to_dict()
        missing_pred = sorted(expected_pred_ids - set(pred_counts))
        repeated_pred = sorted(pred_id for pred_id, count in pred_counts.items() if count != 1)
        invalid_pred_status = sorted(
            model_detail[
                model_detail["pred_id"].astype(str).ne("")
                & ~model_detail["tipo_resultado"].isin(valid_pred_results)
            ]["pred_id"].astype(str).unique().tolist()
        )
        rows.append(
            {
                "invariante": "cada_prediccion_clasificada_exactamente_una_vez",
                "modelo": model,
                "estado": "ok" if not missing_pred and not repeated_pred and not invalid_pred_status else "error",
                "detalle": f"faltantes={len(missing_pred)} repetidos={len(repeated_pred)} estado_invalido={len(invalid_pred_status)}",
            }
        )

    return pd.DataFrame(rows)
