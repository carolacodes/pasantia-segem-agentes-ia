from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


METRIC_EXPLANATIONS = [
    ("total_documentos", "Cantidad de documentos incluidos en la evaluacion de ese modelo o etiqueta."),
    ("total_entidades_gold", "Cantidad de entidades esperadas segun la revision manual. Es la base contra la que se compara."),
    ("total_entidades_predichas", "Cantidad de entidades que el modelo extrajo."),
    ("exacta_span", "Casos donde coinciden documento, etiqueta, valor normalizado y span."),
    ("exacta_valor", "Casos donde coinciden documento, etiqueta y valor normalizado, pero el span difiere o no esta disponible."),
    ("parcial", "Casos donde el modelo encontro la misma etiqueta y un valor muy parecido, pero no identico."),
    ("extra", "Entidades que el modelo extrajo pero que no aparecen en el gold standard."),
    ("no_encontrada", "Entidades que estaban en el gold standard y el modelo no detecto."),
    ("etiqueta_incorrecta", "El valor coincide, pero el modelo uso otra etiqueta."),
    ("duplicada", "El modelo devolvio mas de una vez la misma entidad o una equivalente."),
    ("precision", "De todo lo que el modelo extrajo, que proporcion fue correcta. Sirve para medir ruido o falsos positivos."),
    ("recall", "De todo lo que debia encontrar segun el gold, que proporcion encontro. Sirve para medir omisiones."),
    ("F1-score", "Resume precision y recall en un unico numero. Es util cuando se quiere comparar modelos con una sola metrica."),
    ("cobertura", "Indica cuantas entidades del gold fueron localizadas de algun modo: exactas, parciales o con etiqueta incorrecta."),
]


GRAPH_EXPLANATIONS = {
    "01_precision_recall_f1_por_modelo": "Compara precision, recall y F1 relajado por modelo. Permite ver si un modelo extrae con poco ruido, si cubre muchas entidades o si logra buen equilibrio.",
    "02_f1_estricto_relajado": "Muestra cuanto cambia el F1 cuando solo se aceptan coincidencias exactas frente a cuando tambien se aceptan coincidencias parciales.",
    "03_resultados_por_modelo": "Cuenta exactas, parciales, extras y no encontradas por modelo. Ayuda a entender de donde salen los errores.",
    "04_metricas_por_etiqueta": "Muestra el F1 relajado por etiqueta y modelo. Sirve para ver que etiquetas funcionan bien y cuales necesitan ajuste.",
    "05_cobertura_por_etiqueta_modelo": "Mide que porcentaje de entidades esperadas fue localizado por etiqueta y modelo, aunque alguna tenga diferencia menor o etiqueta incorrecta.",
    "06_comparacion_modelo_vs_regex": "Compara detecciones de los modelos contra detecciones por regex para identificadores y montos.",
    "07_distribucion_scores_confianza": "Muestra como se distribuyen los scores de confianza informados por los modelos. Scores altos no garantizan acierto, pero ayudan a revisar umbrales.",
    "08_distribucion_similitudes_rapidfuzz": "Muestra las similitudes calculadas para coincidencias textuales parciales. Sirve para revisar si el umbral de RapidFuzz es razonable.",
    "09_etiquetas_con_mas_errores": "Ordena las etiquetas con mas errores. Ayuda a priorizar mejoras.",
    "10_documentos_con_mas_errores": "Muestra los documentos donde se concentran mas errores. Sirve para detectar documentos dificiles o problemas de OCR/formato.",
    "11_matriz_confusion_etiquetas": "Cruza etiqueta esperada contra etiqueta predicha. Si hay valores fuera de la diagonal, el modelo suele confundir esas etiquetas.",
}


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")


def filtered_reports(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reports = {
        "entidades_no_encontradas.csv": detail[detail["tipo_resultado"] == "no_encontrada"],
        "entidades_extras.csv": detail[detail["tipo_resultado"] == "extra"],
        "coincidencias_parciales.csv": detail[detail["tipo_resultado"] == "parcial"],
        "entidades_duplicadas.csv": detail[detail["tipo_resultado"] == "duplicada"],
        "etiquetas_incorrectas.csv": detail[detail["tipo_resultado"] == "etiqueta_incorrecta"],
    }
    return reports


def cross_model_reports(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gold_rows = detail[detail["valor_gold"].astype(str).ne("")]
    key_cols = ["documento", "etiqueta_gold", "valor_gold"]
    detected = gold_rows[gold_rows["tipo_resultado"].isin(["exacta_span", "exacta_valor", "parcial"])]
    detected_models = detected.groupby(key_cols)["modelo"].agg(lambda values: sorted(set(values))).reset_index()
    detected_models["cantidad_modelos"] = detected_models["modelo"].map(len)

    all_gold = gold_rows[key_cols + ["modelo"]].drop_duplicates()
    all_models_by_key = all_gold.groupby(key_cols)["modelo"].agg(lambda values: sorted(set(values))).reset_index()
    merged = all_models_by_key.merge(detected_models, on=key_cols, how="left", suffixes=("_evaluados", "_detectaron"))
    merged["modelo_detectaron"] = merged["modelo_detectaron"].apply(lambda x: x if isinstance(x, list) else [])
    merged["cantidad_modelos"] = merged["modelo_detectaron"].map(len)

    none = merged[merged["cantidad_modelos"] == 0].copy()
    single = merged[merged["cantidad_modelos"] == 1].copy()
    none["modelos_detectaron"] = none["modelo_detectaron"].map(lambda values: " | ".join(values))
    single["modelos_detectaron"] = single["modelo_detectaron"].map(lambda values: " | ".join(values))
    return {
        "entidades_no_detectadas_por_ningun_modelo.csv": none.drop(columns=["modelo_detectaron"], errors="ignore"),
        "entidades_detectadas_solo_por_un_modelo.csv": single.drop(columns=["modelo_detectaron"], errors="ignore"),
    }


def dataframe_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    shown = df if max_rows is None else df.head(max_rows)
    return shown.to_html(index=False, escape=True, classes="tabla")


def selected_dataframe_to_html(df: pd.DataFrame, max_rows: int = 20, columns: list[str] | None = None) -> str:
    if df.empty:
        return "<p>Sin datos.</p>"
    shown = df.copy()
    if columns is not None:
        shown = shown[[column for column in columns if column in shown.columns]]
    shown = shown.head(max_rows)
    return shown.to_html(index=False, escape=True, classes="tabla")


def explanations_to_html() -> str:
    rows = "\n".join(
        f"<tr><th>{escape(name)}</th><td>{escape(description)}</td></tr>"
        for name, description in METRIC_EXPLANATIONS
    )
    return f'<table class="tabla explicaciones"><tbody>{rows}</tbody></table>'


def error_report_links_to_html() -> str:
    links = [
        ("entidades_duplicadas.csv", "Duplicadas"),
        ("entidades_extras.csv", "Extras"),
        ("entidades_no_encontradas.csv", "No encontradas"),
        ("etiquetas_incorrectas.csv", "Etiquetas incorrectas"),
    ]
    items = "\n".join(f'<li><a href="{escape(filename)}">{escape(label)}</a></li>' for filename, label in links)
    return f"<ul>{items}</ul>"


def gold_audit_summary(gold_audit: pd.DataFrame) -> pd.DataFrame:
    if gold_audit.empty:
        return pd.DataFrame(columns=["concepto", "cantidad"])
    return pd.DataFrame(
        [
            {"concepto": "filas_gold_originales", "cantidad": len(gold_audit)},
            {"concepto": "entidades_gold_validas", "cantidad": int(gold_audit["categoria"].isin(["obligatoria", "opcional"]).sum())},
            {"concepto": "entidades_obligatorias_metricas_principales", "cantidad": int((gold_audit["categoria"] == "obligatoria").sum())},
            {"concepto": "entidades_opcionales_reporte_separado", "cantidad": int((gold_audit["categoria"] == "opcional").sum())},
            {"concepto": "entidades_excluidas_por_datos_invalidos", "cantidad": int((gold_audit["categoria"] == "excluida").sum())},
        ]
    )


def gold_audit_by_label(gold_audit: pd.DataFrame) -> pd.DataFrame:
    if gold_audit.empty:
        return pd.DataFrame(columns=["etiqueta", "categoria", "cantidad"])
    return gold_audit.groupby(["etiqueta", "categoria"]).size().reset_index(name="cantidad")


def balanced_error_samples(detail: pd.DataFrame, per_group: int = 3) -> pd.DataFrame:
    error_types = ["duplicada", "etiqueta_incorrecta", "extra", "no_encontrada"]
    errors = detail[detail["tipo_resultado"].isin(error_types)].copy()
    if errors.empty:
        return errors
    samples = []
    for (model, result_type), group in errors.groupby(["modelo", "tipo_resultado"], sort=True):
        samples.append(group.head(per_group))
    return pd.concat(samples, ignore_index=True) if samples else errors.head(0)


def write_dashboard(
    path: str | Path,
    metrics_model: pd.DataFrame,
    metrics_label: pd.DataFrame,
    detail: pd.DataFrame,
    graph_files: list[Path],
    gold_audit: pd.DataFrame | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking = metrics_model.sort_values(["f1_relajado", "f1_estricto"], ascending=False) if not metrics_model.empty else metrics_model
    gold_audit = gold_audit if gold_audit is not None else pd.DataFrame()
    gold_summary = gold_audit_summary(gold_audit)
    gold_by_label = gold_audit_by_label(gold_audit)
    error_types = ["duplicada", "etiqueta_incorrecta", "extra", "no_encontrada"]
    errors = detail[detail["tipo_resultado"].isin(error_types)]
    error_summary = (
        errors.groupby(["modelo", "tipo_resultado"]).size().reset_index(name="cantidad")
        if not errors.empty
        else pd.DataFrame(columns=["modelo", "tipo_resultado", "cantidad"])
    )
    error_samples = balanced_error_samples(detail, per_group=3)
    error_sample_columns = [
        "documento",
        "modelo",
        "tipo_resultado",
        "subtipo_resultado",
        "etiqueta_gold",
        "valor_gold",
        "etiqueta_predicha",
        "etiqueta_predicha_original",
        "valor_predicho",
        "score_modelo",
        "score_rapidfuzz",
        "metodo_matching",
        "pred_id_original",
        "etiqueta_original_conservada",
        "valor_original_conservado",
        "span_inicio_original",
        "span_fin_original",
        "pred_id_duplicada",
        "valor_duplicado",
        "span_inicio_duplicado",
        "span_fin_duplicado",
    ]
    graph_blocks = []
    for graph in graph_files:
        title = graph.stem.replace("_", " ")
        description = GRAPH_EXPLANATIONS.get(graph.stem, "Grafico de apoyo para interpretar la evaluacion.")
        graph_blocks.append(
            f'<figure><img src="{escape(graph.as_posix())}" alt="{escape(title)}">'
            f'<figcaption><strong>{escape(title)}</strong><br>{escape(description)}</figcaption></figure>'
        )
    graph_html = "\n".join(graph_blocks)
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard de metricas de entidades</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; line-height: 1.45; }}
    h1, h2 {{ color: #102a43; }}
    .intro {{ max-width: 980px; color: #334e68; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    figure {{ margin: 0; border: 1px solid #d9e2ec; padding: 12px; border-radius: 6px; background: #fff; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #52606d; }}
    .tabla {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
    .tabla th, .tabla td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    .tabla th {{ background: #f0f4f8; }}
  </style>
</head>
<body>
  <h1>Dashboard de metricas de entidades</h1>
  <p class="intro"><a href="dashboard.pdf">Abrir o descargar version PDF del informe</a></p>
  <p class="intro">Este dashboard compara lo que extrajo cada modelo contra una base revisada manualmente. La idea es distinguir tres preguntas: cuanto acierta el modelo, cuanto deja sin detectar y cuanto ruido agrega con entidades de mas o etiquetas equivocadas.</p>
  <h2>Como leer las metricas</h2>
  <p class="intro">Ejemplos rapidos: si el gold contiene 100 entidades obligatorias, todos los modelos deben mostrar total_entidades_gold = 100. Si un modelo extrae 10 entidades y 8 son correctas, la precision es 0,80. Si el gold contiene 10 entidades y el modelo encuentra 6, el recall es 0,60. Si el modelo detecta como persona un nombre que no esta validado como entidad buscada, esa prediccion es extra. Si el valor 20-36003333-4 esta en el gold como cuit_cuil pero el modelo lo etiqueta como dni, es etiqueta_incorrecta.</p>
  {explanations_to_html()}
  <h2>Auditoria del gold</h2>
  <p class="intro">Este resumen sale de la base revisada antes del matching. Explica cuantas entidades entran en la metrica principal, cuantas son opcionales y cuantas se excluyen por problemas de datos.</p>
  {dataframe_to_html(gold_summary)}
  <h2>Gold por etiqueta</h2>
  {dataframe_to_html(gold_by_label)}
  <h2>Ranking de modelos</h2>
  <p class="intro">El ranking ordena los modelos por F1 relajado y luego por F1 estricto. El F1 relajado acepta coincidencias parciales; el estricto solo acepta coincidencias exactas.</p>
  {dataframe_to_html(ranking)}
  <h2>Metricas por etiqueta</h2>
  <p class="intro">Esta tabla permite ver si el rendimiento cambia segun la entidad. Por ejemplo, un modelo puede funcionar bien para personas y mal para identificadores o cuentas.</p>
  {dataframe_to_html(metrics_label)}
  <h2>Graficos</h2>
  <div class="grid">{graph_html}</div>
  <h2>Muestras de errores</h2>
  <p class="intro">Esta seccion no muestra todos los errores. Primero resume la cantidad de errores por modelo y tipo. Luego muestra una muestra balanceada: hasta 3 casos por cada combinacion de modelo y tipo de error. Asi se evita que el dashboard quede dominado por el primer modelo o por un unico tipo de error.</p>
  <h2>Resumen de errores</h2>
  {dataframe_to_html(error_summary, max_rows=100)}
  <h2>Muestra balanceada de errores</h2>
  <p class="intro">Esta tabla no contiene todos los errores: muestra como maximo 3 casos por cada combinacion de modelo y tipo_resultado. Para auditar el universo completo, abrí los CSV completos:</p>
  {error_report_links_to_html()}
  <p class="intro">Se mantienen las columnas gold y predichas aunque algunas esten vacias: esos vacios son parte de la comparacion. En `no_encontrada` quedan vacias las columnas predichas porque el modelo no extrajo la entidad gold. En `extra` quedan vacias las columnas gold porque la prediccion no pudo asociarse a ninguna entidad validada. En `duplicada` quedan vacias las columnas gold porque el duplicado se identifica comparando predicciones entre si antes del matching contra el gold; la fila representa la prediccion repetida descartada. En `etiqueta_incorrecta` se muestran gold y prediccion porque el valor coincide, pero las etiquetas difieren.</p>
  <p class="intro">Para duplicadas, `subtipo_resultado` distingue `duplicado_mismo_valor_mismo_span`, `duplicado_overlap_chunks` y `duplicado_dudoso_sin_span`. Las columnas `*_original*` describen la prediccion conservada; las columnas `*_duplicada*` describen la prediccion repetida descartada.</p>
  {selected_dataframe_to_html(error_samples, max_rows=60, columns=error_sample_columns)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
