import html
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from ftfy import fix_text
from unidecode import unidecode


def parece_html(texto):
    """Detecta si un texto parece contener HTML."""
    if pd.isna(texto):
        return False

    texto = str(texto).lower()
    patrones_html = [
        "<p",
        "<div",
        "<br",
        "<span",
        "<table",
        "<html",
        "&nbsp;",
        "</",
    ]
    return any(patron in texto for patron in patrones_html)


def limpiar_texto_legal(texto):
    """Limpia texto legal OCR sin alterar su contenido sustantivo."""
    if pd.isna(texto):
        return ""

    texto = str(texto)
    texto = fix_text(texto)
    texto = html.unescape(texto)

    # Beautiful Soup se usa de forma preventiva. Solo elimina marcado y contenido
    # no visible; no corrige palabras OCR ni descarta fragmentos del documento.
    soup = BeautifulSoup(texto, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    texto = soup.get_text(separator="\n")

    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = texto.replace("\t", " ").replace("\xa0", " ")

    # La normalizacion es deliberadamente conservadora: se trabaja linea por
    # linea para preservar parrafos y saltos que pueden tener significado legal.
    lineas = []
    for linea in texto.split("\n"):
        linea = re.sub(r" {2,}", " ", linea).strip()
        lineas.append(linea)

    texto = "\n".join(lineas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def normalizar_para_busqueda(texto):
    """Crea una version auxiliar sin acentos y en minusculas para busquedas."""
    if pd.isna(texto):
        return ""

    texto = fix_text(str(texto)).lower()
    texto = unidecode(texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def convertir_a_markdown(row):
    """Construye una representacion Markdown simple del documento limpio."""
    return (
        "# Documento legal\n\n"
        "## Metadatos\n\n"
        f"* ID: {row['id']}\n"
        f"* Nombre: {row['nombre']}\n"
        f"* Clasificación: {row['clasificacion']}\n\n"
        "## Texto limpio\n\n"
        f"{row['texto_limpio']}"
    )


def guardar_muestras_markdown(
    df,
    output_dir,
    cantidad_por_tipo=5,
    random_state=42,
):
    """Guarda muestras reproducibles de oficios y embargos en Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for clasificacion in ("Oficio", "Embargo"):
        registros = df[df["clasificacion"] == clasificacion]
        cantidad = min(cantidad_por_tipo, len(registros))

        if cantidad == 0:
            continue

        muestra = registros.sample(n=cantidad, random_state=random_state)
        prefijo = clasificacion.lower()

        for _, row in muestra.iterrows():
            ruta_salida = output_dir / f"{prefijo}_{row['id']}.md"
            ruta_salida.write_text(str(row["texto_markdown"]), encoding="utf-8")


def detectar_tipo_documento(texto):
    """
    Clasificacion simple inicial: oficio, embargo o desconocido.

    Esta regla es provisoria y se puede mejorar en etapas posteriores.
    """
    if pd.isna(texto):
        return "desconocido"

    texto = str(texto).lower()

    if "embargo" in texto or "trabar embargo" in texto or "embargar" in texto:
        return "embargo"

    if "oficio" in texto or "solicito informe" in texto or "informe si" in texto:
        return "oficio"

    return "desconocido"


def analizar_columnas_texto(df):
    """Detecta columnas que podrian contener el cuerpo legal."""
    resultados = []

    for col in df.columns:
        if df[col].dtype == "object":
            largos = df[col].dropna().astype(str).str.len()

            if len(largos) > 0:
                resultados.append(
                    {
                        "columna": col,
                        "promedio_caracteres": round(largos.mean(), 2),
                        "max_caracteres": int(largos.max()),
                        "valores_no_nulos": int(largos.count()),
                    }
                )

    if not resultados:
        return pd.DataFrame(
            columns=[
                "columna",
                "promedio_caracteres",
                "max_caracteres",
                "valores_no_nulos",
            ]
        )

    return pd.DataFrame(resultados).sort_values(
        by="promedio_caracteres",
        ascending=False,
    )
