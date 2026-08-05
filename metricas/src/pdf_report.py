from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reports import GRAPH_EXPLANATIONS, METRIC_EXPLANATIONS, balanced_error_samples, gold_audit_by_label, gold_audit_summary


PDF_DEPENDENCY_MESSAGE = (
    "No se pudo generar dashboard.pdf porque faltan dependencias de PDF. "
    "Instalar con: pip install reportlab pypdf"
)


def _require_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(PDF_DEPENDENCY_MESSAGE) from exc
    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "TA_LEFT": TA_LEFT,
        "A4": A4,
        "landscape": landscape,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle,
        "cm": cm,
        "Image": Image,
        "KeepTogether": KeepTogether,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > 90:
        return text[:87] + "..."
    return text


def _paragraph(text: Any, style):
    rl = _require_reportlab()
    return rl["Paragraph"](_text(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


def _table(df: pd.DataFrame, title: str, styles: dict[str, Any], max_rows: int | None = None, columns: list[str] | None = None):
    rl = _require_reportlab()
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors = rl["colors"]
    cm = rl["cm"]

    story = [Paragraph(title, styles["Heading2"]), Spacer(1, 0.18 * cm)]
    if df.empty:
        story.extend([Paragraph("Sin datos.", styles["BodyText"]), Spacer(1, 0.25 * cm)])
        return story
    shown = df.copy()
    if columns:
        shown = shown[[column for column in columns if column in shown.columns]]
    if max_rows is not None:
        shown = shown.head(max_rows)
    data = [[Paragraph(str(column), styles["TableHeader"]) for column in shown.columns]]
    for _, row in shown.iterrows():
        data.append([Paragraph(_text(row[column]), styles["TableCell"]) for column in shown.columns])
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EEF6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BCCCDC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFF")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.35 * cm)])
    return story


def _footer(canvas, doc, run_id: str):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.25, 0.31, 0.39)
    footer = f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')} | run_id: {run_id} | pagina {doc.page}"
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.35 * 28.3465, footer)
    canvas.restoreState()


def _image_flowable(path: Path, max_width: float, max_height: float, styles: dict[str, Any]):
    rl = _require_reportlab()
    Image = rl["Image"]
    KeepTogether = rl["KeepTogether"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    cm = rl["cm"]

    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    title = path.stem.replace("_", " ")
    desc = GRAPH_EXPLANATIONS.get(path.stem, "")
    return KeepTogether(
        [
            Paragraph(title, styles["Heading3"]),
            image,
            Paragraph(desc, styles["Small"]),
            Spacer(1, 0.25 * cm),
        ]
    )


def write_dashboard_pdf(
    pdf_path: str | Path,
    metadata: dict[str, Any],
    metrics_model: pd.DataFrame,
    metrics_label: pd.DataFrame,
    detail: pd.DataFrame,
    graph_files: list[Path],
    gold_audit: pd.DataFrame,
    diagnostic_summary: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    diagnostic_detail: pd.DataFrame | None = None,
) -> tuple[bool, str]:
    try:
        rl = _require_reportlab()
    except RuntimeError as exc:
        return False, str(exc)

    colors = rl["colors"]
    landscape = rl["landscape"]
    A4 = rl["A4"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    ParagraphStyle = rl["ParagraphStyle"]
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Spacer = rl["Spacer"]
    PageBreak = rl["PageBreak"]
    Paragraph = rl["Paragraph"]
    cm = rl["cm"]

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontSize=6.5, leading=8, textColor=colors.HexColor("#102A43")))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=6.2, leading=7.4))
    styles["Title"].fontSize = 20
    styles["Title"].leading = 24
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 5
    styles["Heading3"].fontSize = 11
    styles["Heading3"].spaceBefore = 8
    styles["Heading3"].spaceAfter = 4

    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=page_size,
        leftMargin=0.75 * cm,
        rightMargin=0.75 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )
    max_width = page_size[0] - doc.leftMargin - doc.rightMargin
    max_graph_height = page_size[1] - doc.topMargin - doc.bottomMargin - 2.0 * cm
    if isinstance(diagnostic_summary, dict):
        diagnostic_summary_principal = diagnostic_summary.get("principal", pd.DataFrame())
        diagnostic_summary_optional = diagnostic_summary.get("opcional", pd.DataFrame())
        diagnostic_summary_total = diagnostic_summary.get("total", pd.DataFrame())
    else:
        diagnostic_summary_principal = diagnostic_summary if diagnostic_summary is not None else pd.DataFrame()
        diagnostic_summary_optional = pd.DataFrame()
        diagnostic_summary_total = pd.DataFrame()
    diagnostic_detail = diagnostic_detail if diagnostic_detail is not None else pd.DataFrame()

    story = [
        Paragraph("Informe de metricas de entidades", styles["Title"]),
        Spacer(1, 0.25 * cm),
    ]

    meta_rows = pd.DataFrame(
        [
            {"campo": "fecha_hora_corrida", "valor": metadata.get("fecha_hora", "")},
            {"campo": "run_id", "valor": metadata.get("run_id", "")},
            {"campo": "tipo_documento", "valor": metadata.get("tipo_documento", "")},
            {"campo": "modelos", "valor": ", ".join(metadata.get("modelos", []))},
            {"campo": "gold", "valor": metadata.get("gold", "")},
            {"campo": "predicciones", "valor": " | ".join(metadata.get("resultados", []))},
            {"campo": "rapidfuzz_threshold", "valor": metadata.get("rapidfuzz_threshold", "")},
            {"campo": "length_tolerance", "valor": metadata.get("length_tolerance", "")},
        ]
    )
    story.extend(_table(meta_rows, "Datos de la corrida", styles))
    story.append(Paragraph("Explicacion de metricas", styles["Heading2"]))
    for name, description in METRIC_EXPLANATIONS:
        story.append(Paragraph(f"<b>{name}</b>: {description}", styles["BodyText"]))
    story.append(Spacer(1, 0.25 * cm))

    story.append(PageBreak())
    story.extend(_table(gold_audit_summary(gold_audit), "Auditoria del gold", styles))
    story.extend(_table(gold_audit_by_label(gold_audit), "Gold por etiqueta", styles))
    story.extend(_table(metrics_model, "Ranking y metricas por modelo", styles))
    metrics_columns = [
        "modelo",
        "etiqueta",
        "total_entidades_gold",
        "total_entidades_predichas",
        "exactas",
        "parcial",
        "extra",
        "no_encontrada",
        "duplicada",
        "precision_relajada",
        "recall_relajado",
        "f1_relajado",
        "cobertura",
    ]
    story.extend(_table(metrics_label, "Metricas por etiqueta", styles, columns=metrics_columns))

    story.append(PageBreak())
    story.append(Paragraph("Deteccion diagnostica amplia", styles["Heading2"]))
    story.append(
        Paragraph(
            "Las metricas oficiales permanecen sin cambios. Esta seccion revisa solo casos oficiales no_encontrada y extra con reglas mas flexibles para encontrar variantes asociables.",
            styles["BodyText"],
        )
    )
    story.extend(_table(diagnostic_summary_principal, "Deteccion diagnostica principal", styles))
    story.extend(_table(diagnostic_summary_optional, "Deteccion diagnostica opcional", styles))
    story.extend(_table(diagnostic_summary_total, "Deteccion diagnostica total", styles))
    diagnostic_columns = [
        "documento",
        "modelo",
        "tipo_diagnostico",
        "nivel_confianza",
        "regla_principal",
        "etiqueta_gold",
        "valor_gold",
        "etiqueta_predicha",
        "valor_predicho",
        "token_sort_ratio",
        "token_set_ratio",
        "partial_ratio",
        "motivo_deteccion",
    ]
    story.extend(_table(diagnostic_detail, "Muestra diagnostica", styles, max_rows=40, columns=diagnostic_columns))

    story.append(PageBreak())
    story.append(Paragraph("Graficos", styles["Heading2"]))
    for idx, graph in enumerate(graph_files):
        if idx and idx % 2 == 0:
            story.append(PageBreak())
        story.append(_image_flowable(Path(graph), max_width, max_graph_height / 2.15, styles))

    error_columns = [
        "documento",
        "modelo",
        "tipo_resultado",
        "subtipo_resultado",
        "etiqueta_gold",
        "valor_gold",
        "etiqueta_predicha",
        "valor_predicho",
        "metodo_matching",
    ]
    story.append(PageBreak())
    story.append(Paragraph("Muestra balanceada de errores", styles["Heading2"]))
    story.append(
        Paragraph(
            "Esta tabla no contiene todos los errores: muestra como maximo 3 casos por cada combinacion de modelo y tipo_resultado.",
            styles["BodyText"],
        )
    )
    story.extend(_table(balanced_error_samples(detail, per_group=3), "Casos de ejemplo", styles, max_rows=60, columns=error_columns))

    doc.build(story, onFirstPage=lambda c, d: _footer(c, d, str(metadata.get("run_id", ""))), onLaterPages=lambda c, d: _footer(c, d, str(metadata.get("run_id", ""))))
    return True, str(pdf_path)


def verify_dashboard_pdf(pdf_path: str | Path) -> tuple[bool, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        return False, f"No se pudo verificar dashboard.pdf porque falta pypdf. Instalar con: pip install pypdf ({exc})"

    path = Path(pdf_path)
    if not path.exists():
        return False, "dashboard.pdf no existe"
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    image_count = 0
    for page in reader.pages:
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        for obj in xobjects.values():
            try:
                if obj.get_object().get("/Subtype") == "/Image":
                    image_count += 1
            except Exception:
                continue
    if page_count <= 1:
        return False, f"dashboard.pdf tiene {page_count} pagina(s); se esperaba mas de una"
    if image_count == 0:
        return False, "dashboard.pdf no parece contener imagenes embebidas"
    return True, f"dashboard.pdf verificado: paginas={page_count}, imagenes={image_count}"
