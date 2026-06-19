# AGENTS.md

## Project Overview

This project belongs to a paid internship between FaCENA - UNNE and SEGEM S.A. / CECONEA.

The goal is to support the development of AI agents for legal process optimization, focused on reading, cleaning, analyzing, extracting and reviewing legal documents such as oficios and embargos.

The initial work is not to build the full AI agent immediately. The first stage is to prepare clean and reliable data from CSV and JSON files so future LLM/entity extraction workflows can operate correctly.

## Main Pipeline

The project follows this pipeline:

```text
CSV / JSON
↓
Load with Python
↓
Explore columns and examples
↓
Detect legal text field
↓
Clean HTML with Beautiful Soup
↓
Normalize spaces, tabs and line breaks
↓
Fix accents/encoding if needed
↓
Convert to Markdown
↓
Save clean files
↓
Test entity extraction with GLiNER/Qwen
```

## Current Technical Scope

The current scope is:

1. Load CSV and JSON files.
2. Explore their structure.
3. Detect the column or field containing the legal document text.
4. Clean HTML and noisy text.
5. Normalize whitespace, tabs and line breaks.
6. Fix encoding issues when needed.
7. Convert legal document text to Markdown.
8. Save processed outputs.
9. Prepare the data for later entity extraction using GLiNER and/or Qwen.

## Expected Project Structure

```text
pasantia-segem-agentes-ia/
│
├── data/
│   ├── raw/
│   │   ├── Entradas origen drive.csv
│   │   └── Entradas origen drive.json
│   │
│   ├── processed/
│   │   ├── oficios_embargos_limpios.csv
│   │   ├── oficios_embargos_limpios.json
│   │   ├── embargos_limpios.csv
│   │   ├── entidades_embargos_muestra_gliner.csv
│   │   ├── plantilla_analisis_manual_oficios_embargos.csv
│   │   ├── muestras_lectura/
│   │   └── muestras_markdown/
│   │
│   └── samples/
│
├── notebooks/
│   ├── 01_limpieza_csv_json.ipynb
│   └── 02_limpieza_avanzada_embargos.ipynb
│
├── src/
│   ├── cleaning.py
│   └── markdown_utils.py
│
├── docs/
│   └── flujo_oficios_embargos.md
│
├── AGENTS.md
├── CLAUDE.md
├── requirements.txt
└── README.md
```

## Data Rules

Never modify files inside `data/raw/`.

All processed files must be saved inside `data/processed/`.

Do not expose real sensitive legal data in documentation, public commits, prompts, examples or logs.

If examples are needed, anonymize:

- person names;
- DNI;
- CUIT;
- CUIL;
- emails;
- phone numbers;
- addresses;
- CBU;
- CVU;
- aliases;
- bank accounts;
- case numbers;
- judicial file numbers;
- amounts, if required by confidentiality.

Always preserve the original text column if possible.

Do not overwrite original fields. Create new fields such as:

```text
texto_limpio
texto_markdown
tipo_detectado
tiene_html
largo_original
largo_limpio
```

## Important Legal Document Types

The project mainly works with:

- oficios;
- embargos;
- generated responses;
- Salesforce-related records;
- reviewer assignment data.

## Entity Extraction Targets

For oficios, relevant entities may include:

- investigated person;
- DNI/CUIT/CUIL;
- account number;
- CBU/CVU;
- alias;
- card number;
- judicial file number;
- court/judicial body;
- date;
- requested information;
- mentioned platform;
- requested account status;
- requested transactions/movements.

For embargos, relevant entities may include:

- embargoed person;
- DNI/CUIT/CUIL;
- amount;
- currency;
- origin account, if present;
- destination account;
- CBU/CVU;
- alias;
- bank;
- judicial file number;
- court/judicial body;
- date;
- embargo order;
- deposit instructions.

## Cleaning Rules

Use Python and keep the code simple.

Recommended libraries:

```text
pandas
beautifulsoup4
lxml
html5lib
ftfy
markdownify
tqdm
```

Cleaning should handle:

- HTML tags;
- HTML entities such as `&nbsp;`, `&amp;`, etc.;
- scripts and styles;
- encoding issues;
- accents when broken;
- tabs;
- repeated spaces;
- excessive line breaks;
- empty lines;
- noisy copied text;
- legal text formatting.

Do not automatically lowercase the main clean text column.

If lowercase text is needed, create a separate column such as:

```text
texto_normalizado
```

Do not remove accents by default. If accent removal is useful for search or matching, create a separate column.

## Coding Guidelines

Prefer readable Python over complex abstractions.

Use clear function names.

Keep reusable code inside `src/`.

Keep experimental work inside `notebooks/`.

Add comments when a transformation may affect legal meaning.

Handle null values safely.

Avoid assuming that all rows have the same structure.

When processing both CSV and JSON, keep comparable output schemas.

## Suggested Output Columns

Whenever possible, processed datasets should include:

```text
id_original
tipo_detectado
texto_original
texto_limpio
texto_markdown
tiene_html
largo_original
largo_limpio
errores_limpieza
fuente_archivo
```

## Google Colab Notes

The first notebook should be:

```text
notebooks/01_limpieza_csv_json_colab.ipynb
```

It should include:

1. dependency installation;
2. file upload;
3. CSV loading;
4. JSON loading;
5. column exploration;
6. legal text field detection;
7. cleaning functions;
8. Markdown conversion;
9. output export;
10. sample inspection.

## Testing / Validation

Before saving the final processed file, always inspect samples manually.

Check:

- original text vs clean text;
- whether names and legal data remain readable;
- whether line breaks are reasonable;
- whether HTML was removed correctly;
- whether Markdown is usable;
- whether the document type detection makes sense.

## Future Work

Later stages may include:

- GLiNER entity extraction;
- Qwen-based extraction;
- LLM comparison between oficio and generated response;
- automatic review status;
- missing-data detection;
- inconsistency detection;
- reviewer dashboard;
- replacement of Excel-based assignment flow.

## Agent Behavior

When acting as a coding agent:

1. Read this file first.
2. Do not rewrite the whole project unless asked.
3. Make small, safe, reviewable changes.
4. Preserve raw data.
5. Keep sensitive data private.
6. Explain assumptions briefly.
7. Prefer creating reusable functions.
8. Update documentation when changing pipeline behavior.
9. Do not invent column names if the dataset structure is unknown.
10. Ask for the actual column names or inspect the files before implementing final logic.
