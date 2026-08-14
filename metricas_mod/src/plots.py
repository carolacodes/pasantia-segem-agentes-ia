from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt = _load_pyplot()
    plt.close(fig)
    return path


def _bar(df: pd.DataFrame, x: str, y: list[str], title: str, path: Path) -> Path | None:
    if df.empty:
        return None
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index(x)[y].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("valor")
    ax.legend(loc="best")
    return _save(fig, path)


def _bar_single(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> Path | None:
    if df.empty:
        return None
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    return _save(fig, path)


def create_plots(
    detail: pd.DataFrame,
    metrics_model: pd.DataFrame,
    metrics_label: pd.DataFrame,
    regex_compare: pd.DataFrame,
    graph_dir: str | Path,
    diagnostic_summary: pd.DataFrame | None = None,
) -> list[Path]:
    graph_dir = Path(graph_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(graph_dir / ".matplotlib"))
    created: list[Path] = []

    for item in [
        _bar(metrics_model, "modelo", ["precision_relajada", "recall_relajado", "f1_relajado"], "Precision, recall y F1 por modelo", graph_dir / "01_precision_recall_f1_por_modelo.png"),
        _bar(metrics_model, "modelo", ["f1_estricto", "f1_relajado"], "F1 estricto y relajado", graph_dir / "02_f1_estricto_relajado.png"),
        _bar(metrics_model, "modelo", ["exactas", "parcial", "extra", "no_encontrada"], "Resultados por modelo", graph_dir / "03_resultados_por_modelo.png"),
    ]:
        if item:
            created.append(item)

    if not metrics_label.empty:
        label_plot = metrics_label.pivot_table(index="etiqueta", columns="modelo", values="f1_relajado", fill_value=0)
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(12, 6))
        label_plot.plot(kind="bar", ax=ax)
        ax.set_title("F1 relajado por etiqueta")
        created.append(_save(fig, graph_dir / "04_metricas_por_etiqueta.png"))

        coverage_plot = metrics_label.pivot_table(index="etiqueta", columns="modelo", values="cobertura", fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        coverage_plot.plot(kind="bar", ax=ax)
        ax.set_title("Cobertura por etiqueta y modelo")
        created.append(_save(fig, graph_dir / "05_cobertura_por_etiqueta_modelo.png"))

    if not regex_compare.empty:
        regex_summary = regex_compare.groupby("modelo")[["regex_detecta_gold", "modelo_detecta_gold"]].mean().reset_index()
        item = _bar(regex_summary, "modelo", ["regex_detecta_gold", "modelo_detecta_gold"], "Comparacion modelo vs regex", graph_dir / "06_comparacion_modelo_vs_regex.png")
        if item:
            created.append(item)

    scores = pd.to_numeric(detail["score_modelo"], errors="coerce").dropna()
    if not scores.empty:
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(scores, bins=20, color="#2f80ed")
        ax.set_title("Distribucion de scores de confianza")
        created.append(_save(fig, graph_dir / "07_distribucion_scores_confianza.png"))

    similarities = pd.to_numeric(detail["score_rapidfuzz"], errors="coerce").dropna()
    if not similarities.empty:
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(similarities, bins=20, color="#27ae60")
        ax.set_title("Distribucion de similitudes RapidFuzz")
        created.append(_save(fig, graph_dir / "08_distribucion_similitudes_rapidfuzz.png"))

    error_df = detail[detail["tipo_resultado"].isin(["extra", "no_encontrada", "etiqueta_incorrecta", "duplicada"])]
    if not error_df.empty:
        by_label = (
            error_df.assign(etiqueta=lambda df: df["etiqueta_gold"].where(df["etiqueta_gold"].astype(str).ne(""), df["etiqueta_predicha"]))
            .groupby("etiqueta")
            .size()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="errores")
        )
        item = _bar_single(by_label, "etiqueta", "errores", "Etiquetas con mas errores", graph_dir / "09_etiquetas_con_mas_errores.png")
        if item:
            created.append(item)

        by_doc = error_df.groupby("documento").size().sort_values(ascending=False).head(15).reset_index(name="errores")
        item = _bar_single(by_doc, "documento", "errores", "Documentos con mas errores", graph_dir / "10_documentos_con_mas_errores.png")
        if item:
            created.append(item)

    confusion = detail[detail["tipo_resultado"].isin(["exacta_span", "exacta_valor", "parcial", "etiqueta_incorrecta"])]
    if not confusion.empty:
        matrix = pd.crosstab(confusion["etiqueta_gold"], confusion["etiqueta_predicha"])
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_title("Matriz de confusion entre etiquetas")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        created.append(_save(fig, graph_dir / "11_matriz_confusion_etiquetas.png"))

    if diagnostic_summary is not None and not diagnostic_summary.empty:
        item = _bar(
            diagnostic_summary,
            "modelo",
            ["detectadas_oficiales_principales", "detectadas_adicionales_alta", "detectadas_adicionales_media", "candidatas_revision"],
            "Deteccion diagnostica amplia",
            graph_dir / "12_deteccion_diagnostica_amplia.png",
        )
        if item:
            created.append(item)
        item = _bar(
            diagnostic_summary,
            "modelo",
            ["extras_asociables", "extras_reales"],
            "Extras asociables y reales",
            graph_dir / "13_extras_diagnosticos.png",
        )
        if item:
            created.append(item)

    return created
