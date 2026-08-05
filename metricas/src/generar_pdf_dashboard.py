from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from pdf_report import verify_dashboard_pdf, write_dashboard_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenera dashboard.pdf para una corrida existente.")
    parser.add_argument("--run-dir", required=True, help="Carpeta de corrida dentro de metricas/outputs.")
    parser.add_argument("--graph-dir", default=None, help="Carpeta de graficos. Si se omite, usa run_metadata.yaml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "run_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe {metadata_path}")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    graph_dir = Path(args.graph_dir or metadata.get("graficos", ""))
    graph_files = sorted(graph_dir.glob("*.png"))

    metrics_model = pd.read_csv(run_dir / "metricas_por_modelo.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    metrics_label = pd.read_csv(run_dir / "metricas_por_etiqueta_todas.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    detail = pd.read_csv(run_dir / "detalle_comparaciones.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gold_audit = pd.read_csv(run_dir / "auditoria_gold.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")

    ok, message = write_dashboard_pdf(run_dir / "dashboard.pdf", metadata, metrics_model, metrics_label, detail, graph_files, gold_audit)
    if not ok:
        print(f"Advertencia PDF: {message}")
        (run_dir / "pdf_status.txt").write_text(message, encoding="utf-8")
        return
    verified, verify_message = verify_dashboard_pdf(run_dir / "dashboard.pdf")
    print(verify_message)
    (run_dir / "pdf_status.txt").write_text(verify_message, encoding="utf-8")
    if not verified:
        raise RuntimeError(verify_message)


if __name__ == "__main__":
    main()
