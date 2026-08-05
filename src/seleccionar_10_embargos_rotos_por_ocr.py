from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TASK_DIR = Path("data") / "10_embargos_rotos_por_ocr"
INPUT_DIR = TASK_DIR / "input"
OUTPUT_DIR = TASK_DIR / "output_10_embargos_rotos_por_ocr"
OUTPUT_CSV_NAME = "10_embargos_rotos_por_ocr.csv"
OUTPUT_JSON_NAME = "ids_10_embargos_rotos_por_ocr.json"

ID_COLUMN_PRIORITY = [
    "id",
    "numero_archivo",
    "nro_archivo",
    "archivo",
    "id_original",
    "id_embargo",
    "embargo_id",
]

TEXT_COLUMN_PRIORITY = [
    "texto_ocr",
    "ocr",
    "texto_original",
    "texto",
    "texto_limpio",
    "texto_markdown",
]

COMMON_LEGAL_PUNCTUATION = set(".,;:()[]{}-/\\_$%#@+*=<>\"'¿?¡!°º&\n\r\t ")
VOWELS = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)
ALNUM_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]", flags=re.UNICODE)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reasons: dict[str, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def find_input_csv(input_dir: Path) -> Path:
    if not input_dir.exists():
        fail(f"No existe la carpeta de entrada: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not csv_files:
        fail(f"No se encontro ningun CSV en: {input_dir}")
    if len(csv_files) == 1:
        return csv_files[0]

    embargo_matches = [p for p in csv_files if "embargo" in p.name.lower()]
    if len(embargo_matches) == 1:
        return embargo_matches[0]

    found = ", ".join(p.name for p in csv_files)
    fail(
        "Hay mas de un CSV en input y no se puede determinar con seguridad "
        f"cual es la base de embargos. Archivos encontrados: {found}"
    )


def read_csv_preserving_columns(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def detect_id_column(df: pd.DataFrame) -> str:
    columns_by_lower = {str(col).strip().lower(): col for col in df.columns}
    for candidate in ID_COLUMN_PRIORITY:
        if candidate in columns_by_lower:
            col = columns_by_lower[candidate]
            if df[col].notna().any():
                return col

    likely = [
        col
        for col in df.columns
        if "id" in str(col).lower()
        or "archivo" in str(col).lower()
        or "numero" in str(col).lower()
        or "nro" in str(col).lower()
    ]
    if len(likely) == 1 and df[likely[0]].notna().any():
        return likely[0]

    fail(
        "No se pudo determinar con seguridad la columna de identificacion. "
        f"Columnas encontradas: {list(df.columns)}"
    )


def detect_text_column(df: pd.DataFrame) -> str:
    columns_by_lower = {str(col).strip().lower(): col for col in df.columns}
    viable: list[tuple[int, float, int, str]] = []

    for priority, candidate in enumerate(TEXT_COLUMN_PRIORITY):
        if candidate not in columns_by_lower:
            continue
        col = columns_by_lower[candidate]
        text = df[col].dropna().astype(str)
        avg_len = float(text.str.len().mean()) if not text.empty else 0.0
        max_len = int(text.str.len().max()) if not text.empty else 0
        if avg_len >= 80 or max_len >= 500:
            viable.append((priority, -avg_len, -max_len, col))

    if viable:
        viable.sort()
        return viable[0][3]

    long_text_columns = []
    for col in df.columns:
        text = df[col].dropna().astype(str)
        avg_len = float(text.str.len().mean()) if not text.empty else 0.0
        max_len = int(text.str.len().max()) if not text.empty else 0
        if avg_len >= 500 or max_len >= 1500:
            long_text_columns.append((avg_len, max_len, col))

    if len(long_text_columns) == 1:
        return long_text_columns[0][2]

    fail(
        "No se pudo determinar con seguridad la columna de texto OCR. "
        f"Columnas encontradas: {list(df.columns)}"
    )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def ratio(count: float, total: float) -> float:
    return count / total if total else 0.0


def count_weird_chars(text: str) -> int:
    total = 0
    for char in text:
        if char.isalnum() or char in COMMON_LEGAL_PUNCTUATION or char.isspace():
            continue
        total += 1
    return total


def count_unusual_mixed_tokens(tokens: list[str]) -> int:
    normal_legal = re.compile(
        r"^(?:dni|cuil|cuit|cbu|cvu|expte|expediente|art|ley|nro|no|num|"
        r"\$)?[.:/-]?\d[\d. ,:/-]*[a-zA-Z]?$",
        flags=re.IGNORECASE,
    )
    count = 0
    for token in tokens:
        clean = token.strip(".,;:()[]{}")
        if not clean:
            continue
        has_letter = bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", clean))
        has_digit = any(ch.isdigit() for ch in clean)
        if has_letter and has_digit and not normal_legal.match(clean):
            count += 1
    return count


def score_ocr_deterioration(text: str) -> ScoreResult:
    raw = safe_text(text)
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    length = len(normalized)
    tokens = TOKEN_RE.findall(normalized)
    words = WORD_RE.findall(normalized)
    nonspace_chars = [ch for ch in normalized if not ch.isspace()]
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]

    replacement_count = normalized.count("�")
    weird_count = count_weird_chars(normalized)
    non_alnum_count = sum(1 for ch in nonspace_chars if not ALNUM_RE.match(ch))
    symbol_sequences = len(re.findall(r"[^\w\s]{3,}", normalized, flags=re.UNICODE))
    repeated_chars = len(re.findall(r"(.)\1{3,}", normalized, flags=re.UNICODE))
    fragmented_words = len(
        re.findall(
            r"\b(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]\s){3,}[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]\b",
            normalized,
        )
    )
    short_alpha_tokens = [
        word
        for word in words
        if len(word) <= 2 and word.lower() not in {"de", "la", "el", "en", "y", "a"}
    ]
    long_words = [word for word in words if len(word) >= 4]
    no_vowel_words = [word for word in long_words if not any(ch in VOWELS for ch in word)]
    mixed_tokens = count_unusual_mixed_tokens(tokens)
    short_lines = [line for line in lines if len(line) <= 25]
    ocr_confusions = len(
        re.findall(r"l\|I|\|l|0O|O0|1l|l1|\brn\b|vv", normalized, flags=re.IGNORECASE)
    )

    reasons = {
        "texto_vacio_o_muy_corto": 1.0 if length < 80 else 0.0,
        "caracteres_reemplazo": replacement_count,
        "caracteres_extranos_ratio": ratio(weird_count, max(length, 1)),
        "simbolos_no_alfanumericos_ratio": ratio(non_alnum_count, max(len(nonspace_chars), 1)),
        "secuencias_simbolos": symbol_sequences,
        "tokens_cortos_ratio": ratio(len(short_alpha_tokens), max(len(words), 1)),
        "palabras_fragmentadas": fragmented_words,
        "tokens_letras_numeros_raros": mixed_tokens,
        "repeticiones_anormales": repeated_chars,
        "palabras_sin_vocal_ratio": ratio(len(no_vowel_words), max(len(long_words), 1)),
        "lineas_cortadas_ratio": ratio(len(short_lines), max(len(lines), 1)),
        "saltos_linea_por_1000": ratio(normalized.count("\n") * 1000, max(length, 1)),
        "confusiones_ocr": ocr_confusions,
    }

    score = (
        reasons["texto_vacio_o_muy_corto"] * 8.0
        + reasons["caracteres_reemplazo"] * 2.0
        + reasons["caracteres_extranos_ratio"] * 90.0
        + reasons["simbolos_no_alfanumericos_ratio"] * 8.0
        + reasons["secuencias_simbolos"] * 0.9
        + reasons["tokens_cortos_ratio"] * 14.0
        + reasons["palabras_fragmentadas"] * 1.2
        + reasons["tokens_letras_numeros_raros"] * 0.5
        + reasons["repeticiones_anormales"] * 0.7
        + reasons["palabras_sin_vocal_ratio"] * 10.0
        + reasons["lineas_cortadas_ratio"] * 5.0
        + reasons["saltos_linea_por_1000"] * 0.08
        + reasons["confusiones_ocr"] * 0.35
    )

    return ScoreResult(score=round(score, 6), reasons=reasons)


def id_to_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def main() -> None:
    root = repo_root()
    input_dir = root / INPUT_DIR
    output_dir = root / OUTPUT_DIR
    input_csv = find_input_csv(input_dir)

    original_mtime_ns = input_csv.stat().st_mtime_ns
    original_size = input_csv.stat().st_size

    df = read_csv_preserving_columns(input_csv)
    if df.empty:
        fail(f"El CSV de entrada esta vacio: {input_csv}")

    original_columns = list(df.columns)
    id_col = detect_id_column(df)
    text_col = detect_text_column(df)

    working = df.reset_index(names="_orden_original")
    grouped = working.groupby(id_col, sort=False, dropna=False)
    unique_scores = []

    for embargo_id, group in grouped:
        texts = group[text_col].dropna().astype(str)
        text_for_score = texts.iloc[0] if not texts.empty else ""
        score_result = score_ocr_deterioration(text_for_score)
        unique_scores.append(
            {
                "id": embargo_id,
                "json_id": id_to_json_value(embargo_id),
                "score": score_result.score,
                "reasons": score_result.reasons,
                "orden_original": int(group["_orden_original"].min()),
            }
        )

    if len(unique_scores) < 10:
        fail(
            "La base contiene menos de 10 identificadores unicos. "
            f"IDs unicos encontrados: {len(unique_scores)}"
        )

    selected = sorted(
        unique_scores,
        key=lambda item: (-item["score"], item["orden_original"], str(item["id"])),
    )[:10]
    selected_ids = [item["id"] for item in selected]
    selected_json_ids = [item["json_id"] for item in selected]

    if len(set(map(str, selected_json_ids))) != 10:
        fail("La seleccion produjo IDs duplicados.")

    output_rows = df[df[id_col].isin(selected_ids)].copy()
    output_rows = output_rows[original_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / OUTPUT_CSV_NAME
    output_json = output_dir / OUTPUT_JSON_NAME

    output_rows.to_csv(output_csv, index=False, encoding="utf-8")
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(selected_json_ids, file, ensure_ascii=False, indent=2)
        file.write("\n")

    validate_outputs(
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        df=df,
        output_rows=output_rows,
        id_col=id_col,
        selected_json_ids=selected_json_ids,
        original_columns=original_columns,
        original_mtime_ns=original_mtime_ns,
        original_size=original_size,
    )

    print_summary(
        root=root,
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        id_col=id_col,
        text_col=text_col,
        total_rows=len(df),
        unique_ids=df[id_col].nunique(dropna=True),
        selected=selected,
    )


def validate_outputs(
    *,
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    df: pd.DataFrame,
    output_rows: pd.DataFrame,
    id_col: str,
    selected_json_ids: list[Any],
    original_columns: list[str],
    original_mtime_ns: int,
    original_size: int,
) -> None:
    if not output_csv.exists():
        fail(f"No se creo el CSV de salida: {output_csv}")
    if not output_json.exists():
        fail(f"No se creo el JSON de salida: {output_json}")

    csv_check = pd.read_csv(output_csv)
    with output_json.open("r", encoding="utf-8") as file:
        json_check = json.load(file)

    if len(json_check) != 10:
        fail(f"El JSON debe contener exactamente 10 IDs. Contiene: {len(json_check)}")
    if len(set(map(str, json_check))) != 10:
        fail("El JSON contiene IDs duplicados.")

    csv_ids = [id_to_json_value(value) for value in pd.unique(csv_check[id_col])]
    if set(map(str, csv_ids)) != set(map(str, json_check)):
        fail("Los IDs del JSON no coinciden con los IDs del CSV generado.")

    original_ids = set(map(str, pd.unique(df[id_col].dropna())))
    if not set(map(str, json_check)).issubset(original_ids):
        fail("Hay IDs seleccionados que no existen en el CSV original.")

    if list(csv_check.columns) != original_columns:
        fail("No se conservaron todas las columnas originales en el mismo orden.")
    if list(output_rows.columns) != original_columns:
        fail("La seleccion interna altero el orden de columnas original.")

    if df[id_col].is_unique and len(csv_check) != 10:
        fail(f"El CSV de salida debe tener exactamente 10 filas. Tiene: {len(csv_check)}")
    if not df[id_col].is_unique and csv_check[id_col].nunique(dropna=True) != 10:
        fail("El CSV de salida debe contener exactamente 10 IDs unicos.")

    if input_csv.stat().st_mtime_ns != original_mtime_ns or input_csv.stat().st_size != original_size:
        fail("El CSV de entrada fue modificado durante la ejecucion.")

    if selected_json_ids != json_check:
        fail("El JSON generado no respeta el orden deterministico de seleccion.")


def print_summary(
    *,
    root: Path,
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    id_col: str,
    text_col: str,
    total_rows: int,
    unique_ids: int,
    selected: list[dict[str, Any]],
) -> None:
    print("Proceso completado correctamente.")
    print()
    print(f"CSV de entrada: {relative(input_csv, root)}")
    print(f"Columna de identificacion: {id_col}")
    print(f"Columna utilizada para evaluar el OCR: {text_col}")
    print(f"Filas analizadas: {total_rows}")
    print(f"IDs unicos analizados: {unique_ids}")
    print("Embargos seleccionados: 10")
    print()
    print("CSV generado:")
    print(relative(output_csv, root))
    print()
    print("JSON generado:")
    print(relative(output_json, root))
    print()
    print("IDs seleccionados:")
    print(json.dumps([item["json_id"] for item in selected], ensure_ascii=False))
    print()
    print("Puntajes internos de deterioro:")
    for item in selected:
        print(f"ID {item['json_id']}: {item['score']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
