from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from matching import MatchConfig, compare_model
from metrics import metrics_by_label, metrics_by_model
from normalization import normalize_amount, values_equivalent
from reports import dataframe_to_html, write_dashboard


CFG = MatchConfig(
    threshold=85,
    length_tolerance=3,
    numeric_labels={"dni", "cuit_cuil", "cbu", "cvu", "monto"},
    optional_labels={"alias"},
    fuzzy_labels={"persona", "persona_juridica", "alias"},
)


def gold(rows: list[dict[str, object]]) -> pd.DataFrame:
    data = []
    for idx, row in enumerate(rows):
        data.append(
            {
                "entidad_id": f"gold_{idx}",
                "documento": row.get("documento", "doc1"),
                "etiqueta": row["etiqueta"],
                "etiqueta_original": row["etiqueta"],
                "valor": row["valor"],
                "span_inicio": str(row.get("span_inicio", "")),
                "span_fin": str(row.get("span_fin", "")),
                "score_modelo": "",
                "modelo": "gold",
            }
        )
    return pd.DataFrame(data)


def preds(model: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    data = []
    for idx, row in enumerate(rows):
        data.append(
            {
                "entidad_id": f"{model}_{idx}",
                "documento": row.get("documento", "doc1"),
                "etiqueta": row["etiqueta"],
                "etiqueta_original": row.get("etiqueta_original", row["etiqueta"]),
                "valor": row["valor"],
                "span_inicio": str(row.get("span_inicio", "")),
                "span_fin": str(row.get("span_fin", "")),
                "score_modelo": str(row.get("score_modelo", "")),
                "modelo": model,
            }
        )
    columns = [
        "entidad_id",
        "documento",
        "etiqueta",
        "etiqueta_original",
        "valor",
        "span_inicio",
        "span_fin",
        "score_modelo",
        "modelo",
    ]
    return pd.DataFrame(data, columns=columns)


class MetricLogicTests(unittest.TestCase):
    def test_same_gold_total_for_three_models(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "persona", "valor": "Maria Gomez"},
                {"etiqueta": "dni", "valor": "12345678"},
            ]
        )
        details = []
        for model, rows in {
            "a": [{"etiqueta": "persona", "valor": "Juan Perez"}],
            "b": [{"etiqueta": "dni", "valor": "12345678"}],
            "c": [],
        }.items():
            details.append(compare_model(g, preds(model, rows), CFG, model))
        detail = pd.concat(details, ignore_index=True)
        metrics = metrics_by_model(detail)
        self.assertEqual(set(metrics["total_entidades_gold"]), {3})

    def test_argentine_amount_formats_are_valid(self) -> None:
        self.assertEqual(normalize_amount("$ 1.350.000"), "1350000")
        self.assertEqual(normalize_amount("$ 1.350.000,50"), "1350000.5")
        self.assertEqual(normalize_amount("$ 438.867"), "438867")
        self.assertEqual(normalize_amount("$ 170.000"), "170000")
        self.assertEqual(normalize_amount("108.332,50"), "108332.5")
        self.assertEqual(normalize_amount("1.500.000"), "1500000")

    def test_invalid_amounts_are_empty(self) -> None:
        self.assertEqual(normalize_amount("AITOR hope"), "")
        self.assertEqual(normalize_amount("30-70308853-4"), "")
        self.assertEqual(normalize_amount("20-36003333-4"), "")

    def test_empty_normalizations_are_never_equivalent(self) -> None:
        self.assertFalse(values_equivalent("monto", "AITOR hope", "30-70308853-4", CFG.numeric_labels))

    def test_required_gold_with_optional_prediction_does_not_disappear(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "Juan Perez"}])
        p = preds("m", [{"etiqueta": "alias", "valor": "Juan Perez"}])
        detail = compare_model(g, p, CFG, "m")
        row = detail.iloc[0]
        self.assertEqual(row["tipo_resultado"], "etiqueta_incorrecta")
        metrics = metrics_by_label(detail, optional_labels=CFG.optional_labels)
        persona = metrics[metrics["etiqueta"] == "persona"].iloc[0]
        self.assertEqual(persona["total_entidades_gold"], 1)
        self.assertEqual(persona["total_entidades_predichas"], 0)
        self.assertEqual(persona["recall_estricto"], 0)

    def test_amount_and_person_invalid_normalization_are_not_matched(self) -> None:
        g = gold([{"etiqueta": "monto", "valor": "17.196.196"}])
        p = preds("m", [{"etiqueta": "persona", "valor": "AITOR hope"}])
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual(set(detail["tipo_resultado"]), {"no_encontrada", "extra"})
        self.assertNotIn("etiqueta_incorrecta", set(detail["tipo_resultado"]))

    def test_wrong_label_is_fn_for_gold_and_fp_for_predicted_label(self) -> None:
        g = gold([{"etiqueta": "cuit_cuil", "valor": "20-36003333-4"}])
        p = preds("m", [{"etiqueta": "dni", "valor": "20-36003333-4"}])
        detail = compare_model(g, p, CFG, "m")
        metrics = metrics_by_label(detail, include_optional=True)
        cuit = metrics[metrics["etiqueta"] == "cuit_cuil"].iloc[0]
        dni = metrics[metrics["etiqueta"] == "dni"].iloc[0]
        self.assertEqual(cuit["total_entidades_gold"], 1)
        self.assertEqual(cuit["total_entidades_predichas"], 0)
        self.assertEqual(dni["total_entidades_gold"], 0)
        self.assertEqual(dni["total_entidades_predichas"], 1)

    def test_same_value_different_label_is_wrong_label(self) -> None:
        g = gold([{"etiqueta": "cuit_cuil", "valor": "20-36003333-4"}])
        p = preds("m", [{"etiqueta": "dni", "valor": "20-36003333-4"}])
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual(detail.iloc[0]["tipo_resultado"], "etiqueta_incorrecta")

    def test_same_entity_different_spans_is_not_automatic_duplicate(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 100, "span_fin": 110},
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 800, "span_fin": 810},
            ]
        )
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 100, "span_fin": 110},
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 800, "span_fin": 810},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        self.assertNotIn("duplicada", set(detail["tipo_resultado"]))
        self.assertEqual((detail["tipo_resultado"] == "exacta_span").sum(), 2)

    def test_same_entity_same_span_is_duplicate(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 100, "span_fin": 110}])
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 100, "span_fin": 110},
                {"etiqueta": "persona", "valor": "Juan Perez", "span_inicio": 100, "span_fin": 110},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual((detail["tipo_resultado"] == "duplicada").sum(), 1)
        self.assertEqual((detail["tipo_resultado"] == "exacta_span").sum(), 1)

    def test_identifier_one_digit_difference_is_not_partial(self) -> None:
        g = gold([{"etiqueta": "dni", "valor": "36.003.333"}])
        p = preds("m", [{"etiqueta": "dni", "valor": "36.003.338"}])
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual(set(detail["tipo_resultado"]), {"no_encontrada", "extra"})

    def test_rapidfuzz_score_only_for_partial(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "persona", "valor": "Maria Gomez"},
            ]
        )
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "persona", "valor": "Maria Gomes"},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        exact = detail[detail["tipo_resultado"] == "exacta_valor"].iloc[0]
        partial = detail[detail["tipo_resultado"] == "parcial"].iloc[0]
        self.assertEqual(exact["score_rapidfuzz"], "")
        self.assertNotEqual(partial["score_rapidfuzz"], "")
        self.assertEqual(partial["metodo_matching"], "rapidfuzz")

    def test_optional_absent_does_not_change_main_f1(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "alias", "valor": "JP"},
            ]
        )
        p = preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}])
        detail = compare_model(g, p, CFG, "m")
        metrics = metrics_by_model(detail)
        self.assertEqual(metrics.iloc[0]["total_entidades_gold"], 1)
        self.assertEqual(metrics.iloc[0]["f1_estricto"], 1.0)

    def test_each_gold_and_prediction_classified_once(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "Juan Perez"}])
        p = preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}, {"etiqueta": "persona", "valor": "Maria Gomez"}])
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual(detail[detail["gold_id"].astype(str).ne("")].groupby("gold_id").size().tolist(), [1])
        self.assertEqual(sorted(detail[detail["pred_id"].astype(str).ne("")].groupby("pred_id").size().tolist()), [1, 1])

    def test_metrics_by_label_gold_total_does_not_depend_on_model_prediction_count(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "A"}, {"etiqueta": "persona", "valor": "B"}])
        d1 = compare_model(g, preds("a", [{"etiqueta": "persona", "valor": "A"}]), CFG, "a")
        d2 = compare_model(g, preds("b", []), CFG, "b")
        metrics = metrics_by_label(pd.concat([d1, d2], ignore_index=True), optional_labels=CFG.optional_labels)
        persona = metrics[metrics["etiqueta"] == "persona"]
        self.assertEqual(set(persona["total_entidades_gold"]), {2})

    def test_dataframe_to_html_shows_all_rows_by_default(self) -> None:
        df = pd.DataFrame({"modelo": [f"m{i}" for i in range(21)], "etiqueta": ["persona"] * 21})
        html = dataframe_to_html(df)
        self.assertEqual(html.count("<tr"), 22)

    def test_dashboard_shows_all_metric_rows(self) -> None:
        metrics = pd.DataFrame(
            {
                "modelo": [f"m{i}" for i in range(21)],
                "etiqueta": ["persona"] * 21,
                "f1_relajado": [0.1] * 21,
                "f1_estricto": [0.1] * 21,
            }
        )
        detail = pd.DataFrame(columns=["modelo", "tipo_resultado"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dashboard.html"
            write_dashboard(path, metrics, metrics, detail, [])
            html = path.read_text(encoding="utf-8")
        self.assertIn("m20", html)


if __name__ == "__main__":
    unittest.main()
