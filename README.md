# Pasantía SEGEM - Agentes IA para Procesos Legales

Proyecto de trabajo para limpieza, análisis y procesamiento de documentos legales como oficios y embargos.

## Pipeline

CSV / JSON → Python → Limpieza HTML → Normalización → Markdown → Extracción de entidades → Revisión con LLMs.

## Tecnologías iniciales

- Python
- Pandas
- Beautiful Soup
- Markdownify
- ftfy
- Google Colab

## Estado actual

Etapa inicial: exploración y limpieza de base de datos en CSV y JSON.

## Data

- **(80 embargos procesados nuevos formato):** Contiene una nueva estructura json normalizada para validar con mejor claridad las entidades. Este json no tiene revision manual, esta preparado para luego hacer una revision manual.

**Estructura nueva:**

```
[
  {
    “numero_archivo”: null,
    "id": null,
    "nombre_archivo": null,
    "clasificacion": null,
    "texto_limpio": null,

    "cantidad_entidades_encontradas": null,

    "entidades": [
      {
        "etiqueta": null,
        "valor": null,
        "metodo": null,
        "span_inicio": null,
        "span_fin": null,
      }
    ],
  }
]

*Método: como se obtuvo la entidad (gliner,regex,manual)

```

- **(80 json csv revision):** Contiene mismo archivos que la carpeta **(80 embargos procesados nuevo formato)** pero con modificaciones -> eliminacion de entidades extraidas por el modelo que tenian errores OCR, y corrección de errores del modelo marcados como metodo `manual`

- **(80 json csv excel - corregido):** Contiene mismos archivos que la carpeta **(80 json csv excel revision)** pero se le agrego un nuevo metodo de extraccion `corregido`, actualizacion de errores en los metodos `manual` de la version (80 json csv excel revision). Se cambio las etiquetas que tenian nombre de metodo `manual` por `corregido`

- **(80 json csv excel - variantes - todas las entidades):** Contiene mismos archivos que la carpeta **(80 json csv excel - corregido)** pero con una modificacion en la estructura del json para poder marcar etiquetas que contienen muchas variaciones de extraccion.

**Estructura nueva:**

```
[
  {
    “numero_archivo”: null,
    "id": null,
    "nombre_archivo": null,
    "clasificacion": null,
    "texto_limpio": null,

    "entidades": [
      {
        “id_etiqueta”:null,
        "etiqueta": null,
        “variantes”:[
        {
	        “id_variante”: null,
            "valor": null,
            "metodo": null,
            "span_inicio": null,
            "span_fin": null
        }
        ]

      }
    ],
  }
]
```

## data/embargos_input

Contiene 40 archivos de embargos con manualmente, preparados para usarlo en el calculo de las metricas.

# data/prueba

- **prueba:** Contiene dos carpetas de pruebas piloto de variantes
  - **(80 json csv excel - variante - prueba 1):** Primer prueba fallida de variantes de la entidad persona.
  - **(80 json csv excel - variantes - prueba 2):** Segunda prueba exitosa solo con la entidad persona.
