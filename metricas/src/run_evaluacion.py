from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import get_doc_type_config, load_config
from diagnostic_detection import (
    diagnostic_config_from_dict,
    diagnostic_invariants,
    diagnostic_reports,
    evaluate_diagnostic_detection,
    filter_diagnostics_by_scope,
    summarize_diagnostic_detection,
)
from io_utils import build_gold_audit, infer_model_name, prepare_entities, read_csv_auto, validate_spans
from invariants import audit_invariants
from matching import MatchConfig, compare_model
from metrics import metrics_by_label, metrics_by_model, optional_metrics
from pdf_report import verify_dashboard_pdf, write_dashboard_pdf
from plots import create_plots
from regex_compare import compare_models_vs_regex
from reports import cross_model_reports, filtered_reports, write_csv, write_dashboard

MAX_RUN_ID_LENGTH = 60
MAX_SAFE_PATH_LENGTH = 245
LONGEST_OUTPUT_FILENAMES = (
    "entidades_no_detectadas_por_ningun_modelo.csv",
    "entidades_detectadas_solo_por_un_modelo.csv",
    "comparacion_modelos_vs_regex.csv",
    "auditoria_invariantes.csv",
    "dashboard.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua extracciones de entidades contra un gold standard.")
    parser.add_argument("--gold", default=None, help="Ruta del CSV gold standard. Si se omite, usa config.yaml.")
    parser.add_argument("--results", default=None, nargs="+", help="Uno o mas CSV con predicciones. Si se omite, usa config.yaml.")
    parser.add_argument("--outdir", default="metricas/outputs", help="Carpeta de reportes CSV.")
    parser.add_argument("--graph-dir", default="metricas/graficos", help="Carpeta de graficos PNG.")
    parser.add_argument("--config", default="metricas/config.yaml", help="Archivo YAML de configuracion.")
    parser.add_argument("--doc-type", default=None, help="Tipo de documento, por ejemplo embargo u oficio.")
    parser.add_argument("--rapidfuzz-threshold", type=int, default=None, help="Umbral minimo de similitud parcial.")
    parser.add_argument("--length-tolerance", type=int, default=None, help="Tolerancia de longitud para parciales.")
    parser.add_argument("--run-name", default=None, help="Descripcion corta para identificar la corrida.")
    return parser.parse_args()


def safe_slug(value: str, max_length: int = MAX_RUN_ID_LENGTH) -> str:
    slug = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_length].strip("_-") or "corrida"


def build_run_id(
    run_name: str | None,
    result_paths: list[str],
    threshold: int,
    length_tolerance: int,
) -> str:
    del result_paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = safe_slug(run_name or "corrida", max_length=24)
    base = f"{timestamp}_thr{threshold}_len{length_tolerance}_{name}"
    return safe_slug(base)


def unique_run_dir(base_dir: Path, run_id: str) -> Path:
    candidate = base_dir / run_id
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        suffix = f"_{counter}"
        safe_id = f"{run_id[:MAX_RUN_ID_LENGTH - len(suffix)]}{suffix}"
        candidate = base_dir / safe_id
        if not candidate.exists():
            return candidate
        counter += 1


def unique_run_name(output_base: Path, graph_base: Path, run_id: str) -> str:
    counter = 1
    while True:
        suffix = "" if counter == 1 else f"_{counter}"
        safe_id = f"{run_id[:MAX_RUN_ID_LENGTH - len(suffix)]}{suffix}"
        if not (output_base / safe_id).exists() and not (graph_base / safe_id).exists():
            return safe_id
        counter += 1


def validate_run_paths(outdir: Path, graph_dir: Path) -> None:
    candidate_paths = [outdir / filename for filename in LONGEST_OUTPUT_FILENAMES]
    candidate_paths.extend(
        [
            graph_dir / "11_matriz_confusion_etiquetas.png",
            graph_dir / "10_documentos_mayor_cantidad_errores.png",
        ]
    )
    long_paths = [path for path in candidate_paths if len(str(path.resolve())) > MAX_SAFE_PATH_LENGTH]
    if long_paths:
        examples = "\n".join(f"- {path} ({len(str(path.resolve()))} caracteres)" for path in long_paths[:3])
        raise ValueError(
            "La ruta completa de la corrida es demasiado larga para Windows.\n"
            f"Limite seguro configurado: {MAX_SAFE_PATH_LENGTH} caracteres.\n"
            f"Ejemplos de rutas problematicas:\n{examples}\n"
            "Usa un --run-name mas corto o mueve el repositorio a una ruta base mas breve."
        )


def evaluate(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    config = load_config(args.config)
    doc_type = args.doc_type or config["default_doc_type"]
    doc_cfg = get_doc_type_config(config, doc_type)
    input_cfg = config.get("input_paths", {}).get(doc_type, {})
    gold_path = args.gold or input_cfg.get("gold")
    result_paths = args.results
    if result_paths is None:
        result_glob = input_cfg.get("results_glob")
        result_paths = sorted(glob.glob(result_glob)) if result_glob else []
    if not gold_path:
        raise ValueError("Debe indicar --gold o configurar input_paths.<tipo>.gold en config.yaml")
    if not result_paths:
        raise ValueError("Debe indicar --results o configurar input_paths.<tipo>.results_glob en config.yaml")
    threshold = args.rapidfuzz_threshold or config["matching"]["rapidfuzz_threshold"]
    length_tolerance = args.length_tolerance or config["matching"]["length_tolerance"]
    numeric_labels = set(config["matching"]["numeric_labels"])
    fuzzy_labels = set(config["matching"].get("fuzzy_labels", []))
    optional_labels = set(doc_cfg.get("optional_labels", []))

    run_id = build_run_id(args.run_name, result_paths, threshold, length_tolerance)
    output_base = Path(args.outdir) / doc_type
    graph_base = Path(args.graph_dir) / doc_type
    unique_id = unique_run_name(output_base, graph_base, run_id)
    outdir = output_base / unique_id
    graph_dir = graph_base / unique_id
    validate_run_paths(outdir, graph_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    gold_raw = read_csv_auto(gold_path)
    span_errors = validate_spans(gold_raw, config["columns"])
    gold_audit = build_gold_audit(gold_raw, config, optional_labels)
    gold = prepare_entities(gold_raw, config, source="gold", model="gold")

    all_predictions = []
    all_details = []
    match_cfg = MatchConfig(threshold, length_tolerance, numeric_labels, optional_labels, fuzzy_labels)

    for result_path in result_paths:
        model_name = infer_model_name(result_path)
        pred_raw = read_csv_auto(result_path)
        span_errors.extend(validate_spans(pred_raw, config["columns"]))
        pred = prepare_entities(pred_raw, config, source=Path(result_path).name, model=model_name)
        all_predictions.append(pred)
        all_details.append(compare_model(gold, pred, match_cfg, model_name))

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    detail = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    docs_gold = set(gold["documento"])
    docs_pred = set(predictions["documento"]) if not predictions.empty else set()
    validation_rows = [{"tipo": "span_invalido", "detalle": error} for error in span_errors]
    validation_rows.extend({"tipo": "documento_solo_gold", "detalle": doc} for doc in sorted(docs_gold - docs_pred))
    validation_rows.extend({"tipo": "documento_solo_predicciones", "detalle": doc} for doc in sorted(docs_pred - docs_gold))
    validation = pd.DataFrame(validation_rows, columns=["tipo", "detalle"])

    metrics_model = metrics_by_model(detail)
    metrics_label = metrics_by_label(detail, optional_labels=optional_labels)
    metrics_label_all = metrics_by_label(detail, include_optional=True, optional_labels=optional_labels)
    metrics_optional = optional_metrics(detail, optional_labels)
    regex_df = compare_models_vs_regex(gold, predictions, doc_cfg.get("regex_labels", []), numeric_labels)
    invariants_df = audit_invariants(detail, gold, predictions, metrics_model)
    diagnostic_cfg = diagnostic_config_from_dict(config.get("diagnostic_detection", {}))
    diagnostic_detail = evaluate_diagnostic_detection(detail, diagnostic_cfg)
    diagnostic_summary_principal = summarize_diagnostic_detection(detail, diagnostic_detail, metrics_model, scope="principal")
    diagnostic_summary_optional = summarize_diagnostic_detection(detail, diagnostic_detail, metrics_model, scope="opcional")
    diagnostic_summary_total = summarize_diagnostic_detection(detail, diagnostic_detail, metrics_model, scope="total")
    diagnostic_principal = filter_diagnostics_by_scope(diagnostic_detail, "principal")
    diagnostic_optional = filter_diagnostics_by_scope(diagnostic_detail, "opcional")
    diagnostic_invariants_df = diagnostic_invariants(detail, diagnostic_detail)

    write_csv(detail, outdir / "detalle_comparaciones.csv")
    write_csv(metrics_model, outdir / "metricas_por_modelo.csv")
    write_csv(metrics_label, outdir / "metricas_por_etiqueta.csv")
    write_csv(metrics_label_all, outdir / "metricas_por_etiqueta_todas.csv")
    write_csv(metrics_optional, outdir / "metricas_etiquetas_opcionales.csv")
    write_csv(regex_df, outdir / "comparacion_modelos_vs_regex.csv")
    write_csv(validation, outdir / "validaciones.csv")
    write_csv(gold_audit, outdir / "auditoria_gold.csv")
    write_csv(invariants_df, outdir / "auditoria_invariantes.csv")
    write_csv(diagnostic_detail, outdir / "detecciones_diagnosticas.csv")
    write_csv(diagnostic_principal, outdir / "detecciones_diagnosticas_principales.csv")
    write_csv(diagnostic_optional, outdir / "detecciones_diagnosticas_opcionales.csv")
    write_csv(diagnostic_summary_principal, outdir / "resumen_detecciones_diagnosticas_principal.csv")
    write_csv(diagnostic_summary_optional, outdir / "resumen_detecciones_diagnosticas_opcional.csv")
    write_csv(diagnostic_summary_total, outdir / "resumen_detecciones_diagnosticas_total.csv")
    write_csv(diagnostic_invariants_df, outdir / "auditoria_invariantes_diagnosticas.csv")
    for filename, df in diagnostic_reports(diagnostic_detail).items():
        write_csv(df, outdir / filename)

    metadata = {
        "run_id": outdir.name,
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "tipo_documento": doc_type,
        "gold": str(gold_path),
        "resultados": [str(path) for path in result_paths],
        "modelos": [infer_model_name(path) for path in result_paths],
        "rapidfuzz_threshold": threshold,
        "length_tolerance": length_tolerance,
        "diagnostic_detection": config.get("diagnostic_detection", {}),
        "outputs": str(outdir),
        "graficos": str(graph_dir),
    }
    (outdir / "run_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for filename, df in filtered_reports(detail).items():
        write_csv(df, outdir / filename)
    for filename, df in cross_model_reports(detail).items():
        write_csv(df, outdir / filename)

    graphs = create_plots(detail, metrics_model, metrics_label_all, regex_df, graph_dir, diagnostic_summary_principal)
    relative_graphs = [Path(os.path.relpath(graph, start=outdir)) for graph in graphs]
    write_dashboard(
        outdir / "dashboard.html",
        metrics_model,
        metrics_label_all,
        detail,
        relative_graphs,
        gold_audit,
        {
            "principal": diagnostic_summary_principal,
            "opcional": diagnostic_summary_optional,
            "total": diagnostic_summary_total,
        },
        diagnostic_detail,
    )
    pdf_ok, pdf_message = write_dashboard_pdf(
        outdir / "dashboard.pdf",
        metadata,
        metrics_model,
        metrics_label_all,
        detail,
        graphs,
        gold_audit,
        {
            "principal": diagnostic_summary_principal,
            "opcional": diagnostic_summary_optional,
            "total": diagnostic_summary_total,
        },
        diagnostic_detail,
    )
    if pdf_ok:
        verify_ok, verify_message = verify_dashboard_pdf(outdir / "dashboard.pdf")
        pdf_message = verify_message
        if not verify_ok:
            print(f"Advertencia PDF: {verify_message}")
    else:
        print(f"Advertencia PDF: {pdf_message}")
    (outdir / "pdf_status.txt").write_text(pdf_message, encoding="utf-8")
    return outdir, graph_dir, graphs


def main() -> None:
    args = parse_args()
    outdir, graph_dir, graphs = evaluate(args)
    print(f"Reportes generados en: {outdir}")
    print(f"Graficos generados en: {graph_dir}")
    print(f"Cantidad de graficos: {len(graphs)}")


if __name__ == "__main__":
    main()
