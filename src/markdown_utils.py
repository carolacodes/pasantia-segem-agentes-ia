import re
import html
import pandas as pd
from ftfy import fix_text
from markdownify import markdownify as md


def html_a_markdown(texto):
    """
    Convierte HTML/texto a Markdown.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto)
    texto = fix_text(texto)
    texto = html.unescape(texto)

    markdown = md(texto, heading_style="ATX")

    # Limpieza básica
    markdown = re.sub(r"[ ]{2,}", " ", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip()