from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "default_doc_type": "embargo",
    "input_paths": {
        "embargo": {
            "gold": "metricas/inputs/embargos/input_revision_manual/embargos_input.csv",
            "results_glob": "metricas/inputs/embargos/pruebas_modelo/*.csv",
        }
    },
    "columns": {
        "document_id_candidates": ["id", "numero_archivo", "nombre_archivo"],
        "label": "etiqueta",
        "value": "valor",
        "text": "texto_limpio",
        "score": "score",
        "span_start": "span_inicio",
        "span_end": "span_fin",
    },
    "matching": {
        "rapidfuzz_threshold": 85,
        "length_tolerance": 3,
        "numeric_labels": ["dni", "cuit_cuil", "cbu", "cvu", "monto"],
        "fuzzy_labels": ["persona", "persona_juridica", "alias"],
        "label_aliases": {
            "person": "persona",
            "national_id_number": "dni",
            "government_id": "dni",
            "tax_id": "cuit_cuil",
            "bank_account": "cbu",
            "account_number": "cbu",
            "money": "monto",
            "amount": "monto",
        },
    },
    "doc_types": {
        "embargo": {
            "required_labels": [],
            "optional_labels": [
                "alias",
                "persona_juridica",
                "cuit_cuil",
                "dni",
                "cvu",
                "cbu",
                "monto",
            ],
            "regex_labels": ["cuit_cuil", "dni", "cvu", "cbu", "monto"],
        }
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)


def get_doc_type_config(config: dict[str, Any], doc_type: str) -> dict[str, Any]:
    doc_types = config.get("doc_types", {})
    if doc_type not in doc_types:
        available = ", ".join(sorted(doc_types)) or "(sin tipos configurados)"
        raise ValueError(f"Tipo de documento desconocido: {doc_type}. Disponibles: {available}")
    return doc_types[doc_type]
