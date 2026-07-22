"""Normaliza entidades agrupadas con variantes sin modificar los originales.

Esta etapa lee el JSON de entidades agrupadas por variantes, agrega
``valor_referencia`` a cada entidad y genera copias normalizadas en
``data/limpieza/``. El campo original ``valor`` de cada variante se conserva
sin cambios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "80 json csv excel - variantes - todas las entidades"
OUTPUT_DIR = PROJECT_ROOT / "data" / "limpieza"

OUTPUT_JSON = OUTPUT_DIR / "embargos_entidades_variantes_normalizado.json"
OUTPUT_CSV = OUTPUT_DIR / "embargos_entidades_variantes_normalizado.csv"
OUTPUT_XLSX = OUTPUT_DIR / "embargos_entidades_variantes_normalizado.xlsx"

CSV_COLUMNS = [
    "numero_archivo",
    "id",
    "nombre_archivo",
    "clasificacion",
    "texto_limpio",
    "id_etiqueta",
    "etiqueta",
    "valor_referencia",
    "id_variante",
    "valor",
    "metodo",
    "span_inicio",
    "span_fin",
]

DIGIT_ONLY_LABELS = {"DNI", "CUIT_CUIL", "CBU", "CVU"}


def sha256_file(path: Path) -> str:
    """Calcula hash para verificar que un archivo no cambie."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_single_file(folder: Path, suffix: str) -> Path:
    """Busca automáticamente un único archivo por extensión."""
    matches = sorted(folder.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No se encontró ningún archivo {suffix} en {folder}")
    if len(matches) > 1:
        preferred = [p for p in matches if "variantes" in p.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
        names = "\n".join(f"- {p.name}" for p in matches)
        raise RuntimeError(f"Hay más de un archivo {suffix}; no se puede elegir automáticamente:\n{names}")
    return matches[0]


def clean_spaces(value: str) -> str:
    """Quita espacios sobrantes sin cambiar el contenido semántico."""
    return re.sub(r"\s+", " ", value).strip()


def normalize_value(label: Any, value: Any) -> str | None:
    """Normaliza un valor solamente para construir valor_referencia."""
    if value is None:
        return None

    text = str(value)
    if not text.strip():
        return None

    label_text = str(label or "").strip()
    label_upper = label_text.upper()

    if label_upper == "MONTO":
        normalized = text.strip()
        normalized = normalized.replace("$", "")
        normalized = normalized.replace("ARS", "")
        normalized = normalized.replace("ars", "")
        normalized = normalized.replace("pesos", "")
        normalized = normalized.replace("PESOS", "")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace(".", "")
        normalized = re.sub(r"[^0-9,]", "", normalized)
        return normalized or None

    if label_upper in DIGIT_ONLY_LABELS:
        normalized = re.sub(r"\D+", "", text)
        return normalized or None

    if label_text == "persona":
        normalized = clean_spaces(text).lower()
        return normalized or None

    if label_upper == "ALIAS":
        normalized = clean_spaces(text).lower()
        # En alias, los puntos pueden ser parte del valor; solo retiramos
        # espacios accidentales alrededor de esos puntos.
        normalized = re.sub(r"\s*\.\s*", ".", normalized)
        return normalized or None

    normalized = clean_spaces(text)
    return normalized or None


def choose_reference_value(label: Any, variants: list[dict[str, Any]]) -> str | None:
    """Elige el valor normalizado más largo entre las variantes."""
    normalized_values = [
        normalized
        for variant in variants
        for normalized in [normalize_value(label, variant.get("valor"))]
        if normalized
    ]
    if not normalized_values:
        return None
    return max(normalized_values, key=len)


def add_reference_values(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Devuelve una copia de los documentos con valor_referencia agregado."""
    normalized_documents = deepcopy(documents)
    for document in normalized_documents:
        reordered_entities = []
        for entity in document.get("entidades") or []:
            variants = entity.get("variantes") or []
            reordered_entity = {
                "id_etiqueta": entity.get("id_etiqueta"),
                "etiqueta": entity.get("etiqueta"),
                "valor_referencia": choose_reference_value(entity.get("etiqueta"), variants),
                "variantes": variants,
            }
            for key, value in entity.items():
                if key not in reordered_entity:
                    reordered_entity[key] = value
            reordered_entities.append(reordered_entity)
        document["entidades"] = reordered_entities
    return normalized_documents


def flatten_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convierte el JSON agrupado en filas CSV, una fila por variante."""
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_fields = {
            "numero_archivo": document.get("numero_archivo"),
            "id": document.get("id"),
            "nombre_archivo": document.get("nombre_archivo"),
            "clasificacion": document.get("clasificacion"),
            "texto_limpio": document.get("texto_limpio"),
        }
        for entity in document.get("entidades") or []:
            entity_fields = {
                "id_etiqueta": entity.get("id_etiqueta"),
                "etiqueta": entity.get("etiqueta"),
                "valor_referencia": entity.get("valor_referencia"),
            }
            variants = entity.get("variantes") or []
            if not variants:
                rows.append({**document_fields, **entity_fields})
                continue
            for variant in variants:
                row = {
                    **document_fields,
                    **entity_fields,
                    "id_variante": variant.get("id_variante"),
                    "valor": variant.get("valor"),
                    "metodo": variant.get("metodo"),
                    "span_inicio": variant.get("span_inicio"),
                    "span_fin": variant.get("span_fin"),
                }
                rows.append(row)
    return rows


def validate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula métricas de control para inspección final."""
    entities_by_label: Counter[str] = Counter()
    variants_by_label: Counter[str] = Counter()
    top_entities: list[dict[str, Any]] = []
    total_entities = 0
    total_variants = 0
    null_reference_values = 0

    for document in documents:
        for entity in document.get("entidades") or []:
            label = str(entity.get("etiqueta") or "")
            variants = entity.get("variantes") or []
            variant_count = len(variants)

            total_entities += 1
            total_variants += variant_count
            entities_by_label[label] += 1
            variants_by_label[label] += variant_count
            if entity.get("valor_referencia") is None:
                null_reference_values += 1

            top_entities.append(
                {
                    "numero_archivo": document.get("numero_archivo"),
                    "nombre_archivo": document.get("nombre_archivo"),
                    "id_etiqueta": entity.get("id_etiqueta"),
                    "etiqueta": entity.get("etiqueta"),
                    "valor_referencia": entity.get("valor_referencia"),
                    "cantidad_variantes": variant_count,
                }
            )

    return {
        "documentos": len(documents),
        "total_entidades": total_entities,
        "total_variantes": total_variants,
        "entidades_por_etiqueta": dict(sorted(entities_by_label.items())),
        "variantes_por_etiqueta": dict(sorted(variants_by_label.items())),
        "valores_referencia_nulos": null_reference_values,
        "top_20_entidades_con_mas_variantes": sorted(
            top_entities,
            key=lambda item: item["cantidad_variantes"],
            reverse=True,
        )[:20],
    }


def write_json(path: Path, documents: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx_if_available(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        return False

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "entidades_normalizadas"
    sheet.append(CSV_COLUMNS)
    for row in rows:
        sheet.append([row.get(column) for column in CSV_COLUMNS])
    workbook.save(path)
    return True


def ensure_outputs_do_not_exist(paths: list[Path], overwrite_output: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite_output:
        names = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "No se sobrescriben archivos existentes. Borrar manualmente o usar "
            f"--overwrite-output si corresponde.\n{names}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agrega valor_referencia a entidades agrupadas por variantes."
    )
    parser.add_argument(
        "--with-xlsx",
        action="store_true",
        help="También intenta generar un XLSX si openpyxl está instalado.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Permite reemplazar salidas normalizadas existentes en data/limpieza/.",
    )
    args = parser.parse_args()

    input_json = find_single_file(INPUT_DIR, ".json")
    input_csv = find_single_file(INPUT_DIR, ".csv")
    input_hashes_before = {
        input_json: sha256_file(input_json),
        input_csv: sha256_file(input_csv),
    }

    output_paths = [OUTPUT_JSON, OUTPUT_CSV]
    if args.with_xlsx:
        output_paths.append(OUTPUT_XLSX)
    ensure_outputs_do_not_exist(output_paths, args.overwrite_output)

    with input_json.open("r", encoding="utf-8-sig") as fh:
        documents = json.load(fh)
    if not isinstance(documents, list):
        raise TypeError("El JSON de entrada debe ser una lista de documentos.")

    normalized_documents = add_reference_values(documents)
    flattened_rows = flatten_documents(normalized_documents)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_JSON, normalized_documents)
    write_csv(OUTPUT_CSV, flattened_rows)

    generated_paths = [OUTPUT_JSON, OUTPUT_CSV]
    if args.with_xlsx:
        if write_xlsx_if_available(OUTPUT_XLSX, flattened_rows):
            generated_paths.append(OUTPUT_XLSX)
        else:
            print("XLSX opcional no generado: openpyxl no está instalado.")

    input_hashes_after = {
        input_json: sha256_file(input_json),
        input_csv: sha256_file(input_csv),
    }
    originals_unchanged = input_hashes_before == input_hashes_after

    metrics = validate(normalized_documents)

    print("Validaciones finales")
    print("====================")
    print(f"Cantidad de documentos procesados: {metrics['documentos']}")
    print(f"Total de entidades agrupadas: {metrics['total_entidades']}")
    print(f"Total de variantes: {metrics['total_variantes']}")
    print(f"Cantidad de entidades por etiqueta: {metrics['entidades_por_etiqueta']}")
    print(f"Cantidad de variantes por etiqueta: {metrics['variantes_por_etiqueta']}")
    print(f"Cantidad de valores_referencia nulos: {metrics['valores_referencia_nulos']}")
    print("Top 20 entidades con más variantes:")
    for item in metrics["top_20_entidades_con_mas_variantes"]:
        print(
            "- "
            f"archivo={item['numero_archivo']} | "
            f"etiqueta={item['etiqueta']} | "
            f"id_etiqueta={item['id_etiqueta']} | "
            f"variantes={item['cantidad_variantes']}"
        )
    print(f"Archivos originales sin modificaciones: {originals_unchanged}")
    print("Rutas de salida generadas:")
    for path in generated_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
