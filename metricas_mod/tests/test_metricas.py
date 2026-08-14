from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_test_cache"))

from diagnostic_detection import DiagnosticConfig, diagnostic_invariants, evaluate_diagnostic_detection, summarize_diagnostic_detection, summarize_wide_model_detection
from config import get_doc_type_config, load_config
from invariants import audit_invariants
from matching import MatchConfig, compare_model
from metrics import RESULT_TYPES, metrics_by_label, metrics_by_model, metrics_by_model_optional, metrics_by_model_principal_scope, metrics_by_model_total
from normalization import normalize_amount, normalize_value, values_equivalent
from pdf_report import write_dashboard_pdf, verify_dashboard_pdf
from reports import dataframe_to_html, error_summary_by_scope, write_dashboard
from run_evaluacion import MAX_RUN_ID_LENGTH, build_match_config, build_run_id, unique_run_dir, unique_run_name


CFG = MatchConfig(
    threshold=85,
    length_tolerance=3,
    numeric_labels={"dni", "cuit_cuil", "cbu", "cvu", "monto"},
    optional_labels={"alias"},
    fuzzy_labels={"persona", "persona_juridica", "alias"},
)

DIAG_CFG = DiagnosticConfig()


def cfg_with(**overrides: object) -> MatchConfig:
    values = {
        "threshold": CFG.threshold,
        "length_tolerance": CFG.length_tolerance,
        "numeric_labels": CFG.numeric_labels,
        "optional_labels": CFG.optional_labels,
        "fuzzy_labels": CFG.fuzzy_labels,
        "min_span_overlap_ratio": CFG.min_span_overlap_ratio,
        "tier5_token_set_threshold": CFG.tier5_token_set_threshold,
        "tier5_partial_ratio_threshold": CFG.tier5_partial_ratio_threshold,
    }
    values.update(overrides)
    return MatchConfig(**values)


def gold(rows: list[dict[str, object]]) -> pd.DataFrame:
    data = []
    for idx, row in enumerate(rows):
        data.append(
            {
                "entidad_id": str(idx),
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


def diagnostic_pair_detail(label: str, gold_value: str, pred_value: str, model: str = "m") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "documento": "doc1",
                "modelo": model,
                "gold_id": "0",
                "pred_id": "",
                "etiqueta_gold": label,
                "valor_gold": gold_value,
                "etiqueta_predicha": "",
                "valor_predicho": "",
                "score_modelo": "",
                "score_rapidfuzz": "",
                "metodo_matching": "no_encontrada",
                "span_inicio_gold": "",
                "span_fin_gold": "",
                "span_inicio_predicho": "",
                "span_fin_predicho": "",
                "tipo_resultado": "no_encontrada",
                "subtipo_resultado": "",
                "gold_id_relacionado": "",
                "gold_opcional": False,
                "pred_opcional": False,
                "gold_incluida_principal": True,
                "pred_incluida_principal": False,
                "etiqueta_opcional": False,
            },
            {
                "documento": "doc1",
                "modelo": model,
                "gold_id": "",
                "pred_id": f"{model}_0",
                "etiqueta_gold": "",
                "valor_gold": "",
                "etiqueta_predicha": label,
                "valor_predicho": pred_value,
                "score_modelo": "",
                "score_rapidfuzz": "",
                "metodo_matching": "extra",
                "span_inicio_gold": "",
                "span_fin_gold": "",
                "span_inicio_predicho": "",
                "span_fin_predicho": "",
                "tipo_resultado": "extra",
                "subtipo_resultado": "",
                "gold_id_relacionado": "",
                "gold_opcional": False,
                "pred_opcional": False,
                "gold_incluida_principal": False,
                "pred_incluida_principal": True,
                "etiqueta_opcional": False,
            },
        ]
    )


class MetricLogicTests(unittest.TestCase):
    def test_run_id_is_short_and_does_not_include_result_file_names(self) -> None:
        run_id = build_run_id(
            "conjunta",
            [
                "20260805_descripcion_muy_larga_80_384_64_gliner2.csv",
                "otra_fecha_y_descripcion_80_384_64_large_v25.csv",
            ],
            85,
            3,
        )
        self.assertLessEqual(len(run_id), MAX_RUN_ID_LENGTH)
        self.assertIn("thr85_len3_conjunta", run_id)
        self.assertNotIn("gliner2", run_id)
        self.assertNotIn("large_v25", run_id)
        self.assertNotIn("20260805_descripcion", run_id)

    def test_run_id_uses_corrida_when_run_name_is_missing(self) -> None:
        run_id = build_run_id(None, ["80_384_64_pii_v1.csv"], 85, 3)
        self.assertLessEqual(len(run_id), MAX_RUN_ID_LENGTH)
        self.assertTrue(run_id.endswith("_thr85_len3_corrida"))
        self.assertNotIn("pii_v1", run_id)

    def test_unique_run_dir_adds_numeric_suffix_without_exceeding_safe_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            run_id = "a" * MAX_RUN_ID_LENGTH
            (base_dir / run_id).mkdir()
            candidate = unique_run_dir(base_dir, run_id)
        self.assertTrue(candidate.name.endswith("_2"))
        self.assertLessEqual(len(candidate.name), MAX_RUN_ID_LENGTH)

    def test_unique_run_name_checks_outputs_and_graphs_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_base = Path(tmp) / "outputs"
            graph_base = Path(tmp) / "graficos"
            output_base.mkdir()
            graph_base.mkdir()
            (graph_base / "20260805_130000_thr85_len3_conjunta").mkdir()
            run_name = unique_run_name(output_base, graph_base, "20260805_130000_thr85_len3_conjunta")
        self.assertEqual(run_name, "20260805_130000_thr85_len3_conjunta_2")

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
        self.assertEqual(normalize_amount("ARS 1.500.000"), "1500000")
        self.assertEqual(normalize_amount("1 500 000"), "1500000")
        self.assertEqual(normalize_amount("ARS 1.500.000,50"), "1500000.5")
        self.assertEqual(normalize_amount("$438.867,50"), "438867.5")
        self.assertEqual(normalize_amount("438 867"), "438867")
        self.assertEqual(normalize_amount("Pesos 438.867"), "438867")

    def test_invalid_amounts_are_empty(self) -> None:
        self.assertEqual(normalize_amount("AITOR hope"), "")
        self.assertEqual(normalize_amount("30-70308853-4"), "")
        self.assertEqual(normalize_amount("20-36003333-4"), "")

    def test_empty_normalizations_are_never_equivalent(self) -> None:
        self.assertFalse(values_equivalent("monto", "AITOR hope", "30-70308853-4", CFG.numeric_labels))

    def test_dni_prefix_and_separators_are_equivalent(self) -> None:
        self.assertEqual(normalize_value("dni", "DNI 17.196.196", CFG.numeric_labels), "17196196")
        self.assertEqual(normalize_value("dni", "DNI Nº 17.196.196", CFG.numeric_labels), "17196196")
        self.assertEqual(normalize_value("dni", "DNI N° 17.196.196", CFG.numeric_labels), "17196196")
        self.assertEqual(normalize_value("dni", "D.N.I. 17.196.196", CFG.numeric_labels), "17196196")
        self.assertEqual(normalize_value("dni", "D.N.I Nº: 17 196 196", CFG.numeric_labels), "17196196")
        self.assertTrue(values_equivalent("dni", "17.196.196", "DNI: 17 196 196", CFG.numeric_labels))
        self.assertTrue(values_equivalent("dni", "17.196.196", "17-196-196", CFG.numeric_labels))

    def test_dni_one_digit_difference_is_still_different(self) -> None:
        self.assertFalse(values_equivalent("dni", "17.196.196", "DNI 17.196.198", CFG.numeric_labels))

    def test_cuit_cuil_prefixes_and_separators_are_equivalent(self) -> None:
        self.assertEqual(normalize_value("cuit_cuil", "CUIL 27-29231660-2", CFG.numeric_labels), "27292316602")
        self.assertEqual(normalize_value("cuit_cuil", "CUIT Nº 30-70308853-4", CFG.numeric_labels), "30703088534")
        self.assertEqual(normalize_value("cuit_cuil", "C.U.I.T. 30 70308853 4", CFG.numeric_labels), "30703088534")
        self.assertTrue(values_equivalent("cuit_cuil", "27-29231660-2", "CUIT: 27 29231660 2", CFG.numeric_labels))

    def test_cbu_and_cvu_prefixes_and_separators_are_equivalent(self) -> None:
        cbu = "0140025027713653768085"
        cvu = "0000003100012345678901"
        self.assertEqual(normalize_value("cbu", f"CBU Nº: {cbu[:4]}.{cbu[4:8]}-{cbu[8:]}", CFG.numeric_labels), cbu)
        self.assertEqual(normalize_value("cvu", f"C.V.U. N° {cvu[:4]} {cvu[4:8]} {cvu[8:]}", CFG.numeric_labels), cvu)
        self.assertTrue(values_equivalent("cbu", cbu, "CBU 0140-0250-27713653768085", CFG.numeric_labels))
        self.assertTrue(values_equivalent("cvu", cvu, "CVU 0000-0031-00012345678901", CFG.numeric_labels))

    def test_mixed_identifiers_are_ambiguous_not_concatenated(self) -> None:
        mixed = "DNI 20.258.157 / CUIL 23-20258157-9"
        self.assertEqual(normalize_value("dni", mixed, CFG.numeric_labels), "")
        self.assertEqual(normalize_value("cuit_cuil", mixed, CFG.numeric_labels), "")
        self.assertFalse(values_equivalent("dni", "20.258.157", mixed, CFG.numeric_labels))
        self.assertFalse(values_equivalent("cuit_cuil", "23-20258157-9", mixed, CFG.numeric_labels))

    def test_multiple_identifiers_do_not_concatenate_digits(self) -> None:
        self.assertEqual(normalize_value("dni", "DNI 12.345.678 DNI 87.654.321", CFG.numeric_labels), "")

    def test_original_values_are_preserved_in_matching_reports(self) -> None:
        g = gold([{"etiqueta": "dni", "valor": "17.196.196", "span_inicio": 10, "span_fin": 20}])
        p = preds("m", [{"etiqueta": "dni", "valor": "DNI 17.196.196", "span_inicio": 10, "span_fin": 20}])
        detail = compare_model(g, p, CFG, "m")
        row = detail.iloc[0]
        self.assertEqual(row["tipo_resultado"], "exacta_span")
        self.assertEqual(row["valor_gold"], "17.196.196")
        self.assertEqual(row["valor_predicho"], "DNI 17.196.196")

    def test_normalization_enters_existing_exact_value_tiers(self) -> None:
        g = gold([{"etiqueta": "dni", "valor": "17.196.196", "span_inicio": 10, "span_fin": 20}])
        exact_span = compare_model(
            g,
            preds("m", [{"etiqueta": "dni", "valor": "DNI 17.196.196", "span_inicio": 10, "span_fin": 20}]),
            CFG,
            "m",
        )
        exact_value = compare_model(
            g,
            preds("m", [{"etiqueta": "dni", "valor": "DNI 17.196.196", "span_inicio": 30, "span_fin": 45}]),
            CFG,
            "m",
        )
        self.assertEqual(exact_span.iloc[0]["tipo_resultado"], "exacta_span")
        self.assertEqual(exact_value.iloc[0]["tipo_resultado"], "exacta_valor")

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

    def test_greedy_span_mismatch_issue(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "LEVY, VERONICA LILIANA", "span_inicio": 242, "span_fin": 264},
                {"etiqueta": "persona", "valor": "LEVY VERONICA LILIANA", "span_inicio": 575, "span_fin": 596},
            ]
        )
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "LEVY", "span_inicio": 242, "span_fin": 246},
                {"etiqueta": "persona", "valor": "VERONICA LILIANA", "span_inicio": 248, "span_fin": 264},
                {"etiqueta": "persona", "valor": "LEVY VERONICA LILIANA", "span_inicio": 575, "span_fin": 596},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        
        # Check that gold_2 (LEVY VERONICA LILIANA at 575-596) matched model_3
        gold2_match = detail[detail["gold_id"] == "1"].iloc[0]
        self.assertEqual(gold2_match["pred_id"], "m_2")
        self.assertEqual(gold2_match["tipo_resultado"], "exacta_span")

        # Check that gold_1 is matched as parcial in Tier 5 (overlap_span) with m_0 or m_1
        gold1_match = detail[detail["gold_id"] == "0"].iloc[0]
        self.assertEqual(gold1_match["tipo_resultado"], "parcial")
        self.assertEqual(gold1_match["subtipo_resultado"], "overlap_span")

        # Check that the remaining fragment prediction is classified as extra_fragmento
        extra_preds = detail[detail["tipo_resultado"] == "extra"]
        self.assertEqual(len(extra_preds), 1)
        self.assertEqual(extra_preds.iloc[0]["subtipo_resultado"], "extra_fragmento")

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

    def test_model_metrics_main_optional_and_total_scopes(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "alias", "valor": "JP"},
            ]
        )
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "alias", "valor": "JP"},
            ],
        )
        detail = compare_model(g, p, CFG, "m")

        main_before = metrics_by_model(detail)
        optional = metrics_by_model_optional(detail)
        total = metrics_by_model_total(detail)
        main_after = metrics_by_model(detail)

        self.assertEqual(main_before.to_dict("records"), main_after.to_dict("records"))
        self.assertEqual(main_before.iloc[0]["total_entidades_gold"], 1)
        self.assertEqual(main_before.iloc[0]["total_entidades_predichas"], 1)
        self.assertEqual(optional.iloc[0]["total_entidades_gold"], 1)
        self.assertEqual(optional.iloc[0]["total_entidades_predichas"], 1)
        self.assertEqual(total.iloc[0]["total_entidades_gold"], 2)
        self.assertEqual(total.iloc[0]["total_entidades_predichas"], 2)
        self.assertEqual(total.iloc[0]["total_entidades_gold"], main_before.iloc[0]["total_entidades_gold"] + optional.iloc[0]["total_entidades_gold"])
        self.assertEqual(total.iloc[0]["total_entidades_predichas"], main_before.iloc[0]["total_entidades_predichas"] + optional.iloc[0]["total_entidades_predichas"])

    def test_model_metrics_scopes_are_independent_per_model(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "Juan Perez"},
                {"etiqueta": "alias", "valor": "JP"},
            ]
        )
        d1 = compare_model(
            g,
            preds("a", [{"etiqueta": "persona", "valor": "Juan Perez"}, {"etiqueta": "alias", "valor": "JP"}]),
            CFG,
            "a",
        )
        d2 = compare_model(g, preds("b", [{"etiqueta": "alias", "valor": "JP"}]), CFG, "b")
        detail = pd.concat([d1, d2], ignore_index=True)

        main = metrics_by_model(detail).set_index("modelo")
        optional = metrics_by_model_optional(detail).set_index("modelo")
        total = metrics_by_model_total(detail).set_index("modelo")

        self.assertEqual(main.loc["a", "exactas"], 1)
        self.assertEqual(main.loc["b", "no_encontrada"], 1)
        self.assertEqual(optional.loc["a", "exactas"], 1)
        self.assertEqual(optional.loc["b", "exactas"], 1)
        self.assertEqual(total.loc["a", "exactas"], 2)
        self.assertEqual(total.loc["b", "exactas"], 1)

    def test_wrong_label_counts_follow_gold_scope_and_extras_follow_prediction_scope(self) -> None:
        g = gold(
            [
                {"etiqueta": "persona", "valor": "VALOR PRINCIPAL"},
                {"etiqueta": "alias", "valor": "VALOR OPCIONAL"},
            ]
        )
        p = preds(
            "m",
            [
                {"etiqueta": "alias", "valor": "VALOR PRINCIPAL"},
                {"etiqueta": "persona", "valor": "VALOR OPCIONAL"},
                {"etiqueta": "persona", "valor": "EXTRA PRINCIPAL"},
                {"etiqueta": "alias", "valor": "EXTRA OPCIONAL"},
            ],
        )
        detail = compare_model(g, p, CFG, "m")

        principal = metrics_by_model_principal_scope(detail).iloc[0]
        optional = metrics_by_model_optional(detail).iloc[0]
        total = metrics_by_model_total(detail).iloc[0]
        legacy_principal = metrics_by_model(detail).iloc[0]

        self.assertEqual(principal["etiqueta_incorrecta"], 1)
        self.assertEqual(optional["etiqueta_incorrecta"], 1)
        self.assertEqual(principal["extra"], 1)
        self.assertEqual(optional["extra"], 1)
        self.assertEqual(total["etiqueta_incorrecta"], 2)
        self.assertEqual(total["extra"], 2)
        for result_type in RESULT_TYPES:
            self.assertEqual(total[result_type], principal[result_type] + optional[result_type])
        self.assertEqual(total["total_entidades_gold"], principal["total_entidades_gold"] + optional["total_entidades_gold"])
        self.assertEqual(total["total_entidades_predichas"], principal["total_entidades_predichas"] + optional["total_entidades_predichas"])
        for row in [principal, optional, total]:
            expected = round((row["exactas"] + row["parcial"] + row["etiqueta_incorrecta"]) / row["total_entidades_gold"], 4)
            self.assertEqual(row["cobertura"], expected)
        for column in ["precision_estricta", "recall_estricto", "f1_estricto", "precision_relajada", "recall_relajado", "f1_relajado"]:
            self.assertEqual(principal[column], legacy_principal[column])

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

    def test_diagnostic_additional_does_not_change_official_metrics(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Soledad Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Maria Soledad Perez"}]),
            CFG,
            "m",
        )
        before = metrics_by_model(detail)
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        after = metrics_by_model(detail)
        self.assertEqual(diagnostics["tipo_diagnostico"].tolist(), ["detectada_adicional_alta"])
        self.assertEqual(
            before[["precision_relajada", "recall_relajado", "f1_relajado"]].to_dict("records"),
            after[["precision_relajada", "recall_relajado", "f1_relajado"]].to_dict("records"),
        )

    def test_wide_model_summary_counts_reliable_detection_once(self) -> None:
        detail = pd.DataFrame(
            [
                {"documento": "doc1", "modelo": "m", "gold_id": "g0", "pred_id": "p0", "tipo_resultado": "exacta_span", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g1", "pred_id": "p1", "tipo_resultado": "exacta_valor", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g2", "pred_id": "p2", "tipo_resultado": "parcial", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g3", "pred_id": "", "tipo_resultado": "no_encontrada", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g4", "pred_id": "", "tipo_resultado": "no_encontrada", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g5", "pred_id": "", "tipo_resultado": "no_encontrada", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "g6", "pred_id": "", "tipo_resultado": "no_encontrada", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "", "pred_id": "p7", "tipo_resultado": "extra", "subtipo_resultado": ""},
                {"documento": "doc1", "modelo": "m", "gold_id": "", "pred_id": "p8", "tipo_resultado": "extra", "subtipo_resultado": "extra_fragmento"},
            ]
        )
        diagnostics = pd.DataFrame(
            [
                {"documento": "doc1", "modelo": "m", "gold_id": "g3", "pred_id": "p3", "tipo_diagnostico": "detectada_adicional_alta"},
                {"documento": "doc1", "modelo": "m", "gold_id": "g3", "pred_id": "p9", "tipo_diagnostico": "detectada_adicional_alta"},
                {"documento": "doc1", "modelo": "m", "gold_id": "g4", "pred_id": "p4", "tipo_diagnostico": "detectada_adicional_media"},
                {"documento": "doc1", "modelo": "m", "gold_id": "g5", "pred_id": "p5", "tipo_diagnostico": "candidata_revision"},
                {"documento": "doc1", "modelo": "m", "gold_id": "g6", "pred_id": "", "tipo_diagnostico": "no_encontrada_sin_candidato"},
                {"documento": "doc1", "modelo": "m", "gold_id": "", "pred_id": "p7", "tipo_diagnostico": "extra_real"},
            ]
        )
        before = detail.copy(deep=True)
        summary = summarize_wide_model_detection(detail, diagnostics).iloc[0]

        self.assertEqual(summary["detectadas_amplias"], 5)
        self.assertEqual(summary["extras_reales"], 1)
        self.assertEqual(summary["no_encontradas_reales"], 1)
        self.assertEqual(summary["candidatas_revision"], 1)
        self.assertEqual(summary["precision_amplia"], 0.8333)
        self.assertEqual(summary["recall_amplio"], 0.8333)
        self.assertEqual(summary["f1_amplio"], 0.8333)
        pd.testing.assert_frame_equal(detail, before)

    def test_diagnostic_prediction_cannot_match_two_gold(self) -> None:
        detail = compare_model(
            gold(
                [
                    {"etiqueta": "persona", "valor": "Dr. Juan Perez"},
                    {"etiqueta": "persona", "valor": "Sr. Juan Perez"},
                ]
            ),
            preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(diagnostics["tipo_diagnostico"].isin(["detectada_adicional_alta", "detectada_adicional_media"]).sum(), 1)
        self.assertEqual((diagnostics["tipo_diagnostico"] == "no_encontrada_sin_candidato").sum(), 1)

    def test_diagnostic_same_gold_id_can_match_independently_per_model(self) -> None:
        d1 = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Perez"}]),
            preds("m1", [{"etiqueta": "persona", "valor": "Maria Perez"}]),
            CFG,
            "m1",
        )
        d2 = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Perez"}]),
            preds("m2", [{"etiqueta": "persona", "valor": "Maria Perez"}]),
            CFG,
            "m2",
        )
        detail = pd.concat([d1, d2], ignore_index=True)
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        matched = diagnostics[diagnostics["tipo_diagnostico"].eq("detectada_adicional_alta")]
        self.assertEqual(len(matched), 2)
        self.assertEqual(set(matched["modelo"]), {"m1", "m2"})

    def test_diagnostic_gold_cannot_match_two_predictions(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Soledad Perez"}]),
            preds(
                "m",
                [
                    {"etiqueta": "persona", "valor": "Maria Soledad Perez"},
                    {"etiqueta": "persona", "valor": "Soledad Perez"},
                ],
            ),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(diagnostics["tipo_diagnostico"].isin(["detectada_adicional_alta", "detectada_adicional_media"]).sum(), 1)
        self.assertEqual((diagnostics["tipo_diagnostico"] == "extra_real").sum(), 1)

    def test_diagnostic_requires_same_document(self) -> None:
        detail = compare_model(
            gold([{"documento": "doc1", "etiqueta": "persona", "valor": "Dra. Maria Perez"}]),
            preds("m", [{"documento": "doc2", "etiqueta": "persona", "valor": "Maria Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertNotIn("detectada_adicional", set(diagnostics["tipo_diagnostico"]))

    def test_diagnostic_requires_same_label_by_default(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Perez"}]),
            preds("m", [{"etiqueta": "alias", "valor": "Maria Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertNotIn("detectada_adicional", set(diagnostics["tipo_diagnostico"]))

    def test_high_model_score_alone_is_not_enough_for_diagnostic_match(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Carlos Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Dra. Leticia Frappa", "score_modelo": 0.99}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_numeric_diagnostic_uses_shared_normalization_for_high_confidence(self) -> None:
        detail = diagnostic_pair_detail("dni", "17.196.196", "D.N.I Nº 17 196 196")
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        row = diagnostics[diagnostics["tipo_diagnostico"] == "detectada_adicional_alta"].iloc[0]
        self.assertEqual(row["identificador_normalizado_gold"], "17196196")
        self.assertEqual(row["identificador_normalizado_predicho"], "17196196")
        self.assertEqual(row["regla_principal"], "identificador_normalizado_exacto")

    def test_numeric_diagnostic_rejects_different_digits(self) -> None:
        detail = diagnostic_pair_detail("dni", "17.196.196", "D.N.I Nº 17 196 198")
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_amount_diagnostic_uses_shared_normalization_for_high_confidence(self) -> None:
        detail = diagnostic_pair_detail("monto", "$ 438.867", "Pesos 438 867")
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        row = diagnostics[diagnostics["tipo_diagnostico"] == "detectada_adicional_alta"].iloc[0]
        self.assertEqual(row["identificador_normalizado_gold"], "438867")
        self.assertEqual(row["identificador_normalizado_predicho"], "438867")

    def test_title_difference_can_generate_detectada_adicional(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Soledad Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Maria Soledad Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        row = diagnostics[diagnostics["tipo_diagnostico"] == "detectada_adicional_alta"].iloc[0]
        self.assertTrue(row["diferencia_por_titulo"])
        self.assertEqual(row["nivel_confianza"], "alta")

    def test_text_containment_can_generate_detectada_adicional(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Maria Soledad Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Soledad Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        row = diagnostics[diagnostics["tipo_diagnostico"] == "detectada_adicional_media"].iloc[0]
        self.assertTrue(row["hay_contencion"])
        self.assertEqual(row["nivel_confianza"], "media")

    def test_single_surname_fragment_is_candidate_revision_not_media(self) -> None:
        # Avoid Tier 5 overlap match by using a low overlap ratio so it reaches diagnostic_detection
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "WAJSBRUT GONZALO JAVIER ALBERTO ROBERTO", "span_inicio": 10, "span_fin": 60}]),
            preds("m", [{"etiqueta": "persona", "valor": "WAJSBRUT", "span_inicio": 10, "span_fin": 18}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(diagnostics[diagnostics["pred_id"].astype(str).ne("")].iloc[0]["tipo_diagnostico"], "candidata_revision")

    def test_span_overlap_alone_does_not_generate_detectada_adicional(self) -> None:
        # Same label and overlap matches in Tier 5 if ratio >= 0.30; test text mismatch with different labels
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Alfa Beta", "span_inicio": 10, "span_fin": 20}]),
            preds("m", [{"etiqueta": "alias", "valor": "Gamma Delta", "span_inicio": 15, "span_fin": 25}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_short_containment_is_not_enough_for_diagnostic_match(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona_juridica", "valor": "NALDO LOMBARDI S.A."}]),
            preds("m", [{"etiqueta": "persona_juridica", "valor": "Aldo"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_dni_with_prefix_and_same_digits_is_high_confidence(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "dni", "valor": "17.196.196"}]),
            preds("m", [{"etiqueta": "dni", "valor": "DNI 17.196.196"}]),
            CFG,
            "m",
        )
        self.assertEqual(detail.iloc[0]["tipo_resultado"], "exacta_valor")
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertTrue(diagnostics.empty)

    def test_dni_with_one_different_digit_is_not_diagnostic_match(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "dni", "valor": "17.196.196"}]),
            preds("m", [{"etiqueta": "dni", "valor": "17.196.198"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_cbu_with_prefix_and_same_digits_is_high_confidence(self) -> None:
        value = "0000003100012345678901"
        detail = compare_model(
            gold([{"etiqueta": "cbu", "valor": value}]),
            preds("m", [{"etiqueta": "cbu", "valor": f"CBU {value[:8]}-{value[8:]}"}]),
            CFG,
            "m",
        )
        self.assertEqual(detail.iloc[0]["tipo_resultado"], "exacta_valor")
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertTrue(diagnostics.empty)

    def test_completely_different_person_stays_extra_real(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Carlos Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Dra. Leticia Frappa"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertEqual(set(diagnostics["tipo_diagnostico"]), {"no_encontrada_sin_candidato", "extra_real"})

    def test_error_summaries_separate_main_optional_and_total(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Perez"}, {"etiqueta": "alias", "valor": "JP"}]),
            preds("m", []),
            CFG,
            "m",
        )
        principal = error_summary_by_scope(detail, "principal")
        optional = error_summary_by_scope(detail, "opcional")
        total = error_summary_by_scope(detail, "total")
        self.assertEqual(int(principal["cantidad"].sum()), 1)
        self.assertEqual(int(optional["cantidad"].sum()), 1)
        self.assertEqual(int(total["cantidad"].sum()), 2)

    def test_diagnostic_rapidfuzz_scores_come_from_library(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Maria Soledad Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Soledad Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        row = diagnostics[diagnostics["tipo_diagnostico"].isin(["detectada_adicional_alta", "detectada_adicional_media"])].iloc[0]
        self.assertEqual(row["token_sort_ratio"], round(float(fuzz.token_sort_ratio("Maria Soledad Perez", "Soledad Perez")), 2))
        self.assertEqual(row["token_set_ratio"], round(float(fuzz.token_set_ratio("Maria Soledad Perez", "Soledad Perez")), 2))
        self.assertEqual(row["partial_ratio"], round(float(fuzz.partial_ratio("Maria Soledad Perez", "Soledad Perez")), 2))

    def test_diagnostic_uses_only_official_missing_and_extra(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        self.assertTrue(diagnostics.empty)

    def test_diagnostic_summary_counts_wide_detection(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Perez"}, {"etiqueta": "persona", "valor": "Dra. Maria Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}, {"etiqueta": "persona", "valor": "Maria Perez"}]),
            CFG,
            "m",
        )
        official = metrics_by_model(detail)
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        summary = summarize_diagnostic_detection(detail, diagnostics, official).iloc[0]
        self.assertEqual(summary["detectadas_oficiales_principales"], 1)
        self.assertEqual(summary["detectadas_adicionales_alta"], 1)
        self.assertEqual(summary["total_detectadas_amplias_confiables"], 2)
        invariants = diagnostic_invariants(detail, diagnostics)
        self.assertTrue(invariants["invariante_no_encontradas_ok"].all())
        self.assertTrue(invariants["invariante_extras_ok"].all())

    def test_diagnostic_principal_summary_counts_only_principal_gold(self) -> None:
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Dra. Maria Perez"}, {"etiqueta": "alias", "valor": "Dra. Maria Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Maria Perez"}, {"etiqueta": "alias", "valor": "Maria Perez"}]),
            CFG,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, DIAG_CFG)
        principal = summarize_diagnostic_detection(detail, diagnostics, scope="principal").iloc[0]
        optional = summarize_diagnostic_detection(detail, diagnostics, scope="opcional").iloc[0]
        self.assertEqual(principal["total_gold_principal"], 1)
        self.assertEqual(principal["detectadas_adicionales_alta"], 1)
        self.assertEqual(optional["total_gold_opcional"], 1)
        self.assertEqual(optional["detectadas_adicionales_alta"], 1)

    def test_candidate_revision_does_not_increase_reliable_wide_percentage(self) -> None:
        strict_official_cfg = MatchConfig(
            threshold=99,
            length_tolerance=3,
            numeric_labels=CFG.numeric_labels,
            optional_labels=CFG.optional_labels,
            fuzzy_labels=CFG.fuzzy_labels,
        )
        review_cfg = DiagnosticConfig(token_sort_threshold=98, token_set_threshold=98, partial_ratio_threshold=98)
        detail = compare_model(
            gold([{"etiqueta": "persona", "valor": "Juan Carlos Perez"}]),
            preds("m", [{"etiqueta": "persona", "valor": "Juan Carlos Peres"}]),
            strict_official_cfg,
            "m",
        )
        diagnostics = evaluate_diagnostic_detection(detail, review_cfg)
        summary = summarize_diagnostic_detection(detail, diagnostics, scope="principal").iloc[0]
        self.assertEqual((diagnostics["tipo_diagnostico"] == "candidata_revision").sum(), 1)
        self.assertEqual(summary["candidatas_revision"], 1)
        self.assertEqual(summary["total_detectadas_amplias_confiables"], summary["detectadas_oficiales_principales"])

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
            write_dashboard(path, metrics, metrics, metrics, metrics, metrics, detail, [])
            html = path.read_text(encoding="utf-8")
        self.assertIn("m20", html)

    def test_pdf_generation_when_dependencies_are_available(self) -> None:
        try:
            import reportlab  # noqa: F401
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("Dependencias PDF no instaladas")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib no instalado")

        metrics = pd.DataFrame(
            {
                "modelo": ["m1", "m2"],
                "etiqueta": ["persona", "persona"],
                "total_entidades_gold": [2, 2],
                "total_entidades_predichas": [2, 1],
                "exactas": [2, 1],
                "parcial": [0, 0],
                "extra": [0, 0],
                "no_encontrada": [0, 1],
                "duplicada": [0, 0],
                "precision_relajada": [1.0, 1.0],
                "recall_relajado": [1.0, 0.5],
                "f1_relajado": [1.0, 0.6667],
                "cobertura": [1.0, 0.5],
                "f1_estricto": [1.0, 0.6667],
            }
        )
        detail = compare_model(gold([{"etiqueta": "persona", "valor": "A"}]), preds("m1", []), CFG, "m1")
        audit = pd.DataFrame(
            {
                "documento": ["doc1"],
                "etiqueta": ["persona"],
                "valor": ["A"],
                "incluida": [True],
                "categoria": ["obligatoria"],
                "motivo_exclusion": [""],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graph = tmp_path / "grafico.png"
            fig, ax = plt.subplots()
            ax.plot([1, 2], [1, 2])
            fig.savefig(graph)
            plt.close(fig)
            pdf = tmp_path / "dashboard.pdf"
            ok, message = write_dashboard_pdf(
                pdf,
                {
                    "run_id": "test_run",
                    "fecha_hora": "2026-08-03T00:00:00",
                    "tipo_documento": "embargo",
                    "modelos": ["m1", "m2"],
                    "gold": "gold.csv",
                    "resultados": ["m1.csv", "m2.csv"],
                    "rapidfuzz_threshold": 85,
                    "length_tolerance": 3,
                },
                metrics,
                metrics,
                metrics,
                metrics,
                metrics,
                detail,
                [graph],
                audit,
            )
            self.assertTrue(ok, message)
            verified, verify_message = verify_dashboard_pdf(pdf)
            self.assertTrue(verified, verify_message)

    def test_write_csv_adds_numero_archivo_first_column(self) -> None:
        from reports import write_csv
        df = pd.DataFrame({
            "documento": ["doc_001", "doc_002", "doc_001"],
            "etiqueta": ["persona", "dni", "monto"],
            "valor": ["Juan", "123", "100"]
        })
        mapping = {
            "doc_001": "1",
            "doc_002": "2"
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_file = tmp_path / "test_out.csv"
            write_csv(df, csv_file, mapping)
            
            # Read back and verify columns
            from io_utils import read_csv_auto
            df_read = read_csv_auto(csv_file)
            self.assertEqual(list(df_read.columns)[0], "numero_archivo")
            self.assertEqual(list(df_read["numero_archivo"]), ["1", "2", "1"])

    def test_write_csv_reorders_numero_archivo_if_exists(self) -> None:
        from reports import write_csv
        df = pd.DataFrame({
            "documento": ["doc_001", "doc_002"],
            "etiqueta": ["persona", "dni"],
            "numero_archivo": ["x", "y"]
        })
        mapping = {
            "doc_001": "1",
            "doc_002": "2"
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_file = tmp_path / "test_out.csv"
            write_csv(df, csv_file, mapping)
            
            from io_utils import read_csv_auto
            df_read = read_csv_auto(csv_file)
            self.assertEqual(list(df_read.columns)[0], "numero_archivo")
            self.assertEqual(list(df_read["numero_archivo"]), ["x", "y"])

    def test_fragmented_entity_tier5_and_extra_fragmento(self) -> None:
        g = gold([
            {"etiqueta": "persona", "valor": "Arana Van Vlimmeren Gaston Martin", "span_inicio": 190, "span_fin": 223},
            {"etiqueta": "persona", "valor": "ARANA VAN VLIMMEREN", "span_inicio": 1232, "span_fin": 1251},
        ])
        p = preds("model1", [
            {"etiqueta": "persona", "valor": "LUCIANA PAULA C/ ARANA VAN VLIMMEREN", "span_inicio": 173, "span_fin": 209},
            {"etiqueta": "persona", "valor": "GASTON MARTIN", "span_inicio": 210, "span_fin": 223},
        ])
        detail = compare_model(g, p, CFG, "model1")
        
        # gold_1 (Arana Van Vlimmeren Gaston Martin) should be matched as parcial via overlap_span (Tier 5) with model_1
        gold1_match = detail[detail["gold_id"] == "0"].iloc[0]
        self.assertEqual(gold1_match["tipo_resultado"], "parcial")
        self.assertEqual(gold1_match["subtipo_resultado"], "overlap_span")
        self.assertEqual(gold1_match["pred_id"], "model1_0")
        
        # model_2 (GASTON MARTIN) overlaps with gold_1, so it should be extra with subtipo extra_fragmento
        model2_extra = detail[detail["pred_id"] == "model1_1"].iloc[0]
        self.assertEqual(model2_extra["tipo_resultado"], "extra")
        self.assertEqual(model2_extra["subtipo_resultado"], "extra_fragmento")
        self.assertEqual(model2_extra["gold_id"], "")
        self.assertEqual(model2_extra["gold_id_relacionado"], "0")

    def test_tier5_rejects_overlap_without_textual_evidence(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds("m", [{"etiqueta": "persona", "valor": "MARIA GOMEZ", "span_inicio": 105, "span_fin": 118}])
        detail = compare_model(g, p, CFG, "m")
        self.assertEqual(set(detail["tipo_resultado"]), {"no_encontrada", "extra"})
        self.assertNotIn("overlap_span", set(detail["subtipo_resultado"]))

    def test_tier5_accepts_overlap_with_textual_evidence(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds("m", [{"etiqueta": "persona", "valor": "CARLOS PEREZ", "span_inicio": 108, "span_fin": 120}])
        detail = compare_model(g, p, CFG, "m")
        row = detail.iloc[0]
        self.assertEqual(row["tipo_resultado"], "parcial")
        self.assertEqual(row["subtipo_resultado"], "overlap_span")

    def test_extra_fragmento_requires_textual_evidence(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120},
                {"etiqueta": "persona", "valor": "MARIA GOMEZ", "span_inicio": 105, "span_fin": 118},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        extra = detail[detail["pred_id"] == "m_1"].iloc[0]
        self.assertEqual(extra["tipo_resultado"], "extra")
        self.assertEqual(extra["subtipo_resultado"], "")
        self.assertEqual(extra["gold_id_relacionado"], "")

    def test_extra_fragmento_keeps_related_gold_id_when_textual_evidence_exists(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120},
                {"etiqueta": "persona", "valor": "CARLOS PEREZ", "span_inicio": 108, "span_fin": 120},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        extra = detail[detail["pred_id"] == "m_1"].iloc[0]
        self.assertEqual(extra["tipo_resultado"], "extra")
        self.assertEqual(extra["subtipo_resultado"], "extra_fragmento")
        self.assertEqual(extra["gold_id"], "")
        self.assertEqual(extra["gold_id_relacionado"], "0")

    def test_yaml_min_span_overlap_ratio_changes_tier5_behavior(self) -> None:
        yaml_template = """
matching:
  rapidfuzz_threshold: 85
  length_tolerance: 3
  min_span_overlap_ratio: {min_overlap}
  tier5_token_set_threshold: 60
  tier5_partial_ratio_threshold: 70
doc_types:
  embargo:
    optional_labels: []
"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml_template.format(min_overlap="0.60"), encoding="utf-8")
            strict_config = load_config(config_path)
            strict_cfg = build_match_config(strict_config, get_doc_type_config(strict_config, "embargo"))
            config_path.write_text(yaml_template.format(min_overlap="0.50"), encoding="utf-8")
            loose_config = load_config(config_path)
            loose_cfg = build_match_config(loose_config, get_doc_type_config(loose_config, "embargo"))

        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds("m", [{"etiqueta": "persona", "valor": "CARLOS PEREZ", "span_inicio": 110, "span_fin": 120}])
        strict_detail = compare_model(g, p, strict_cfg, "m")
        loose_detail = compare_model(g, p, loose_cfg, "m")
        self.assertEqual(set(strict_detail["tipo_resultado"]), {"no_encontrada", "extra"})
        self.assertEqual(loose_detail.iloc[0]["subtipo_resultado"], "overlap_span")

    def test_yaml_tier5_text_thresholds_change_behavior(self) -> None:
        yaml_template = """
matching:
  rapidfuzz_threshold: 85
  length_tolerance: 3
  min_span_overlap_ratio: 0.30
  tier5_token_set_threshold: {token_set}
  tier5_partial_ratio_threshold: {partial_ratio}
doc_types:
  embargo:
    optional_labels: []
"""
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(yaml_template.format(token_set="101", partial_ratio="101"), encoding="utf-8")
            strict_config = load_config(config_path)
            strict_cfg = build_match_config(strict_config, get_doc_type_config(strict_config, "embargo"))
            config_path.write_text(yaml_template.format(token_set="60", partial_ratio="70"), encoding="utf-8")
            loose_config = load_config(config_path)
            loose_cfg = build_match_config(loose_config, get_doc_type_config(loose_config, "embargo"))

        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds("m", [{"etiqueta": "persona", "valor": "CARLOS PEREZ", "span_inicio": 108, "span_fin": 120}])
        strict_detail = compare_model(g, p, strict_cfg, "m")
        loose_detail = compare_model(g, p, loose_cfg, "m")
        self.assertEqual(set(strict_detail["tipo_resultado"]), {"no_encontrada", "extra"})
        self.assertEqual(loose_detail.iloc[0]["subtipo_resultado"], "overlap_span")

    def test_non_tier5_official_metrics_are_unchanged_by_new_thresholds(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "Juan Perez"}])
        p = preds("m", [{"etiqueta": "persona", "valor": "Juan Perez"}])
        default_detail = compare_model(g, p, CFG, "m")
        strict_detail = compare_model(
            g,
            p,
            cfg_with(min_span_overlap_ratio=0.99, tier5_token_set_threshold=101, tier5_partial_ratio_threshold=101),
            "m",
        )
        self.assertEqual(metrics_by_model(default_detail).to_dict("records"), metrics_by_model(strict_detail).to_dict("records"))

    def test_tier5_and_extra_fragmento_invariants_hold(self) -> None:
        g = gold([{"etiqueta": "persona", "valor": "JUAN CARLOS PEREZ", "span_inicio": 100, "span_fin": 120}])
        p = preds(
            "m",
            [
                {"etiqueta": "persona", "valor": "CARLOS PEREZ", "span_inicio": 108, "span_fin": 120},
                {"etiqueta": "persona", "valor": "JUAN PEREZ", "span_inicio": 100, "span_fin": 110},
            ],
        )
        detail = compare_model(g, p, CFG, "m")
        metrics = metrics_by_model(detail)
        invariants = audit_invariants(detail, g, p, metrics)
        extra_fragmento = detail[detail["subtipo_resultado"] == "extra_fragmento"].iloc[0]
        self.assertTrue(invariants["estado"].eq("ok").all())
        self.assertEqual(metrics.iloc[0]["total_entidades_gold"], 1)
        self.assertEqual(extra_fragmento["gold_id"], "")
        self.assertEqual(extra_fragmento["gold_id_relacionado"], "0")

    def test_distant_containment_is_not_matched_in_diagnostics(self) -> None:
        g = gold([
            {"etiqueta": "persona", "valor": "Arana Van Vlimmeren Gaston Martin", "span_inicio": 190, "span_fin": 223},
            {"etiqueta": "persona", "valor": "ARANA VAN VLIMMEREN", "span_inicio": 1232, "span_fin": 1251},
        ])
        p = preds("model1", [
            {"etiqueta": "persona", "valor": "LUCIANA PAULA C/ ARANA VAN VLIMMEREN", "span_inicio": 173, "span_fin": 209},
            {"etiqueta": "persona", "valor": "GASTON MARTIN", "span_inicio": 210, "span_fin": 223},
        ])
        detail = compare_model(g, p, CFG, "model1")
        diag = evaluate_diagnostic_detection(detail, DIAG_CFG)
        
        # gold_2 (ARANA VAN VLIMMEREN at span 1232-1251) should NOT be associated with model1_0 at span 173-209 because distance > 50
        gold2_diag = diag[diag["gold_id"] == "1"].iloc[0]
        self.assertEqual(gold2_diag["tipo_diagnostico"], "no_encontrada_sin_candidato")


if __name__ == "__main__":
    unittest.main()
