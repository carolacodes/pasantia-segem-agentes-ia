from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path("data/processed/embargos_limpieza_avanzada.csv")
OUTPUT_PATH = Path("data/processed/embargos_limpieza_avanzada_80_mejores.csv")

TEXT_COLUMNS_PRIORITY = [
    "texto_limpieza_avanzada",
    "texto_limpio",
    "texto_base",
    "texto_ocr",
    "texto_markdown",
]

SEVERE_ALERT_COLUMNS = [
    "alerta_reduccion_alta",
    "alerta_texto_muy_corto",
    "alerta_posible_perdida_datos",
]

DATA_LOSS_ALERT_COLUMNS = [
    "alerta_desaparece_dni",
    "alerta_desaparece_cuil_cuit",
    "alerta_desaparece_cbu_cvu",
    "alerta_desaparece_expediente",
    "alerta_desaparece_monto",
    "alerta_desaparece_fecha",
    "alerta_desaparece_alias",
    "alerta_desaparece_banco",
    "alerta_desaparece_caratula",
    "alerta_desaparece_juzgado_secretaria",
]

KEYWORDS_EMBARGO = [
    "embargo",
    "embargar",
    "autos",
    "juzgado",
    "caratula",
    "carátula",
    "transferencia",
    "banco",
    "cuenta",
    "cbu",
    "cvu",
    "oficio",
]

PATTERNS = {
    "tiene_dni_regex": re.compile(
        r"\b(?:dni|documento|d\.?n\.?i\.?)\s*(?:n(?:ro|°|º)?\.?)?\s*:?\s*\d{1,2}(?:\.?\d{3}){2}\b",
        flags=re.IGNORECASE,
    ),
    "tiene_cuil_cuit_regex": re.compile(
        r"\b(?:cuil|cuit|c\.?u\.?i\.?[lt]\.?)\s*:?\s*\d{2}[-\s]?\d{8}[-\s]?\d\b|\b\d{2}-\d{8}-\d\b",
        flags=re.IGNORECASE,
    ),
    "tiene_cbu_cvu_regex": re.compile(
        r"\b(?:cbu|cvu)\s*:?\s*\d(?:[\s-]?\d){21}\b",
        flags=re.IGNORECASE,
    ),
    "tiene_monto_regex": re.compile(
        r"(?:\$|ars|pesos?)\s*\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?|\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?\s*(?:pesos|ars)\b",
        flags=re.IGNORECASE,
    ),
    "tiene_expediente_regex": re.compile(
        r"\b(?:expte|expediente|autos)\s*(?:n(?:ro|°|º)?\.?)?\s*:?\s*[A-Z0-9./\- ]{4,}\b",
        flags=re.IGNORECASE,
    ),
}

SENSITIVE_PATTERNS = [
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{2}-\d{8}-\d\b"), "[CUIT/CUIL]"),
    (re.compile(r"\b(?:\d[\s-]?){22}\b"), "[CBU/CVU]"),
    (re.compile(r"\b\d{1,2}(?:\.?\d{3}){2}\b"), "[DNI]"),
    (re.compile(r"\$\s*\d[\d.,]*"), "[MONTO]"),
]


def to_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "si", "sí", "yes"}


def find_text_column(df):
    for column in TEXT_COLUMNS_PRIORITY:
        if column in df.columns:
            return column
    object_columns = df.select_dtypes(include="object").columns
    if len(object_columns) == 0:
        raise ValueError("No se encontro una columna de texto para rankear embargos.")
    return max(object_columns, key=lambda col: df[col].fillna("").astype(str).str.len().mean())


def detect_patterns(df, text_column):
    text = df[text_column].fillna("").astype(str)
    for output_column, pattern in PATTERNS.items():
        df[output_column] = text.str.contains(pattern, regex=True, na=False)

    keyword_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(keyword) for keyword in KEYWORDS_EMBARGO) + r")\b",
        flags=re.IGNORECASE,
    )
    df["tiene_palabras_clave_embargo"] = text.str.contains(
        keyword_pattern,
        regex=True,
        na=False,
    )
    return df


def get_text_length(df, text_column):
    if "largo_limpio" in df.columns:
        return pd.to_numeric(df["largo_limpio"], errors="coerce").fillna(0)
    if "len_despues" in df.columns:
        return pd.to_numeric(df["len_despues"], errors="coerce").fillna(0)
    return df[text_column].fillna("").astype(str).str.len()


def normalized_reduction(df):
    if "porcentaje_reduccion" not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    reduction = pd.to_numeric(df["porcentaje_reduccion"], errors="coerce").fillna(0)
    if reduction.max() <= 1:
        reduction = reduction * 100
    return reduction.clip(lower=0)


def compute_ranking(df, text_column):
    score = pd.Series(0.0, index=df.index)

    if "ocr_quality_score" in df.columns:
        score += pd.to_numeric(df["ocr_quality_score"], errors="coerce").fillna(0)

    if "ocr_estado" in df.columns:
        estado = df["ocr_estado"].fillna("").astype(str).str.lower()
        score += estado.eq("legible") * 25
        score -= estado.eq("muy_ruidoso") * 50
        score -= estado.eq("irrecuperable") * 90

    text_length = get_text_length(df, text_column)
    score += (text_length.clip(upper=6000) / 6000) * 30
    score -= text_length.lt(500) * 35
    score -= text_length.lt(200) * 65

    reduction = normalized_reduction(df)
    score -= reduction.clip(upper=100) * 0.25
    score -= reduction.gt(60) * 20
    score += reduction.le(35) * 8

    for column in SEVERE_ALERT_COLUMNS:
        if column in df.columns:
            score -= df[column].map(to_bool) * 30

    for column in DATA_LOSS_ALERT_COLUMNS:
        if column in df.columns:
            score -= df[column].map(to_bool) * 12

    for column in PATTERNS:
        score += df[column].map(to_bool) * 9
    score += df["tiene_palabras_clave_embargo"].map(to_bool) * 12

    if "important_data_count" in df.columns:
        important = pd.to_numeric(df["important_data_count"], errors="coerce").fillna(0)
        score += important.clip(upper=10) * 2

    if "legal_terms_count" in df.columns:
        legal_terms = pd.to_numeric(df["legal_terms_count"], errors="coerce").fillna(0)
        score += legal_terms.clip(upper=10)

    df["ranking_80_score"] = score.round(4)
    df["_texto_len_ranking"] = text_length
    return df


def build_candidate_mask(df):
    mask = pd.Series(True, index=df.index)
    if "ocr_estado" in df.columns:
        bad_state = df["ocr_estado"].fillna("").astype(str).str.lower().isin(
            ["irrecuperable", "muy_ruidoso"]
        )
        mask &= ~bad_state
    if "alerta_texto_muy_corto" in df.columns:
        mask &= ~df["alerta_texto_muy_corto"].map(to_bool)
    if "_texto_len_ranking" in df.columns:
        mask &= df["_texto_len_ranking"].ge(300)
    return mask


def anonymize_preview(text):
    preview = str(text)[:300]
    for pattern, replacement in SENSITIVE_PATTERNS:
        preview = pattern.sub(replacement, preview)
    return preview.replace("\n", " ")


def print_summary(df, selected, text_column):
    print(f"Cantidad total de embargos disponibles: {len(df)}")
    print(f"Cantidad seleccionada: {len(selected)}")

    if "ocr_estado" in df.columns:
        print("\nDistribucion de ocr_estado:")
        print(df["ocr_estado"].fillna("sin_dato").value_counts().to_string())

    print("\nRanking de seleccionados:")
    print(f"score minimo: {selected['ranking_80_score'].min():.4f}")
    print(f"score maximo: {selected['ranking_80_score'].max():.4f}")
    print(f"score promedio: {selected['ranking_80_score'].mean():.4f}")

    print("\nPatrones detectados en seleccionados:")
    for column in [
        "tiene_dni_regex",
        "tiene_cuil_cuit_regex",
        "tiene_cbu_cvu_regex",
        "tiene_monto_regex",
        "tiene_expediente_regex",
    ]:
        print(f"{column}: {int(selected[column].sum())}")

    preview_columns = ["id", "ranking_80_score"]
    if "ocr_quality_score" in selected.columns:
        preview_columns.append("ocr_quality_score")
    if "ocr_estado" in selected.columns:
        preview_columns.append("ocr_estado")

    preview = selected.head(10).copy()
    preview["longitud_texto"] = preview["_texto_len_ranking"].astype(int)
    preview["preview_300_anonimizado"] = preview[text_column].map(anonymize_preview)
    preview_columns += ["longitud_texto", "preview_300_anonimizado"]

    print("\nVista previa anonimizada de los primeros 10 seleccionados:")
    print(preview[preview_columns].to_string(index=False))


def main():
    df = pd.read_csv(INPUT_PATH)
    text_column = find_text_column(df)

    df = detect_patterns(df, text_column)
    df = compute_ranking(df, text_column)

    candidate_mask = build_candidate_mask(df)
    candidates = df[candidate_mask].copy()
    if len(candidates) < 80:
        candidates = df.copy()

    sort_columns = ["ranking_80_score"]
    ascending = [False]
    if "ocr_quality_score" in candidates.columns:
        sort_columns.append("ocr_quality_score")
        ascending.append(False)
    sort_columns.append("_texto_len_ranking")
    ascending.append(False)

    selected = candidates.sort_values(sort_columns, ascending=ascending).head(80).copy()
    selected = selected.drop(columns=["_texto_len_ranking"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print_summary(df, df.loc[selected.index].assign(ranking_80_score=selected["ranking_80_score"]), text_column)
    print(f"\nArchivo guardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
