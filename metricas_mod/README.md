# Metricas de extraccion de entidades

Esta carpeta evalua entidades extraidas por modelos contra una base validada manualmente o `gold standard`.

El objetivo no es solo producir numeros: tambien ayuda a entender que errores comete cada modelo, que etiquetas detecta mejor, donde omite informacion y si una regla por regex podria resolver mejor algunos campos.

## Estructura

- `inputs/`: bases de entrada colocadas manualmente.
- `src/`: codigo Python de carga, matching, metricas, reportes, graficos y dashboard.
- `outputs/`: reportes CSV y dashboard generados.
- `graficos/`: visualizaciones PNG generadas.
- `tests/`: datos sinteticos para validar la herramienta sin usar datos reales.
- `config.yaml`: etiquetas, columnas, rutas y reglas configurables por tipo documental.

## Dependencias

El proyecto ya usa `pandas` y `PyYAML`. Para esta herramienta tambien se necesitan:

```bash
pip install -r metricas/requirements.txt
```

La version PDF del dashboard usa:

- `reportlab` para generar `dashboard.pdf`;
- `pypdf` para verificar paginas e imagenes embebidas.

## Como colocar las bases

La estructura recomendada para embargos es:

```
metricas/inputs/
|-- embargos/
|   |-- input_revision_manual/
|   |   `-- embargos_input.csv
|   `-- pruebas_modelo/
|       |-- 80_384_64_gliner2.csv
|       |-- 80_384_64_pii_v1.csv
|       `-- 80_384_64_large_v25.csv
`-- oficios/
    |-- input_revision_manual/
    `-- pruebas_modelo/
```

No se modifican los archivos de entrada. La herramienta solo lee los CSV y escribe salidas en `outputs/` y `graficos/`.

## Ejecucion

Desde la raiz del repositorio:

```bash
python metricas/src/run_evaluacion.py --doc-type embargo
```

Ese comando usa las rutas configuradas en `config.yaml`:

```text
metricas/inputs/embargos/input_revision_manual/embargos_input.csv
metricas/inputs/embargos/pruebas_modelo/*.csv
```

Tambien se puede ejecutar indicando rutas de manera explicita:

```bash
python metricas/src/run_evaluacion.py --gold metricas/inputs/embargos/input_revision_manual/embargos_input.csv --results metricas/inputs/embargos/pruebas_modelo/80_384_64_gliner2.csv metricas/inputs/embargos/pruebas_modelo/80_384_64_pii_v1.csv metricas/inputs/embargos/pruebas_modelo/80_384_64_large_v25.csv --outdir metricas/outputs --graph-dir metricas/graficos --doc-type embargo --rapidfuzz-threshold 85 --length-tolerance 3
```

Tambien se puede evaluar un solo modelo pasando un unico archivo en `--results`.

Para ejecutar la base conjunta configurada en `config.yaml`:

```bash
python metricas/src/run_evaluacion.py --doc-type conjunta --run-name conjunta
```

Cada ejecucion crea una carpeta nueva para no pisar resultados anteriores. El nombre de la carpeta se arma con:

- fecha y hora;
- umbral de RapidFuzz;
- tolerancia de longitud;
- una descripcion corta, si se usa `--run-name`; si no se indica, usa `corrida`.

No se agregan nombres de archivos CSV ni nombres concatenados de modelos al nombre de carpeta. Eso evita rutas demasiado largas en Windows. Los modelos, archivos completos usados, umbral y tolerancia quedan registrados dentro de `run_metadata.yaml`.

Ejemplo:

```text
metricas/outputs/embargo/20260805_130000_thr85_len3_conjunta/
metricas/graficos/embargo/20260805_130000_thr85_len3_conjunta/
```

Si queres agregar una descripcion a la corrida:

```bash
python metricas/src/run_evaluacion.py --doc-type embargo --run-name conjunta
```

Eso genera un nombre parecido a:

```text
20260805_130000_thr85_len3_conjunta
```

`--run-name` no reemplaza la fecha ni los parametros; solo agrega una descripcion breve al final. Si ya existe una carpeta con el mismo nombre, la herramienta crea otra con sufijo numerico, por ejemplo `_2` o `_3`. Antes de escribir los archivos tambien valida que la ruta completa de los reportes principales no sea excesiva.

## Configuracion

`config.yaml` permite cambiar:

- rutas por tipo de documento;
- columnas candidatas para identificar documentos;
- nombres de columnas de etiqueta, valor, score y spans;
- etiquetas opcionales;
- etiquetas evaluadas tambien por regex;
- etiquetas numericas o identificadores que no usan similitud difusa;
- normalizacion por etiqueta de identificadores (`dni`, `cuit_cuil`, `cbu`, `cvu`) y montos;
- alias de etiquetas de modelos a etiquetas canonicas del gold;
- umbral de RapidFuzz, tolerancia de longitud y reglas del Tier 5 por overlap de spans;
- reglas de deteccion diagnostica amplia para revisar variantes que quedaron como `no_encontrada` y `extra`.

El mapeo de etiquetas evita comparar, por ejemplo, `person` contra `persona` como si fueran etiquetas distintas. La configuracion inicial incluye:

```yaml
label_aliases:
  person: persona
  national_id_number: dni
  government_id: dni
  tax_id: cuit_cuil
  bank_account: cbu
  account_number: cbu
```

Para usar otro tipo documental, por ejemplo `oficio`, agregar o ajustar su entrada en `doc_types` y ejecutar con:

```bash
python metricas/src/run_evaluacion.py --doc-type oficio
```

## Matching

La comparacion se realiza por documento y etiqueta. Cada prediccion se usa como maximo una vez.

- `exacta_span`: coincide documento, etiqueta, valor normalizado y span.
- `exacta_valor`: coincide documento, etiqueta y valor normalizado, pero el span es diferente o no esta disponible.
- `parcial`: misma etiqueta y valor textual suficientemente parecido segun RapidFuzz.
- `extra`: el modelo extrajo algo que no esta en el gold.
- `no_encontrada`: existe en el gold, pero el modelo no la extrajo.
- `etiqueta_incorrecta`: coincide el valor, pero no la etiqueta.
- `duplicada`: el modelo repitio la misma entidad normalizada o una equivalente.

Para `dni`, `cuit_cuil`, `cbu`, `cvu` y `monto` no se usa fuzzy matching. Los identificadores se normalizan segun su etiqueta antes de comparar: se aceptan prefijos como `DNI`, `CUIT`, `CUIL`, `CBU` o `CVU`, y separadores como espacios, puntos o guiones, siempre que quede un unico candidato compatible con la etiqueta. Si un valor mezcla varios identificadores o resulta ambiguo, no se fuerza una equivalencia. Los montos se comparan como valores numericos normalizados, aceptando variantes como `$`, `ARS`, espacios y separadores argentinos sin modificar el valor original mostrado en los reportes.

La normalizacion numerica tambien reconoce variantes frecuentes de OCR y formato, como `D.N.I.`, `C.U.I.T.`, `C.V.U.`, `Nº`, `N°`, dos puntos, puntos, guiones y espacios internos. La misma funcion de normalizacion se reutiliza en el diagnostico amplio para etiquetas numericas (`dni`, `cuit_cuil`, `cbu`, `cvu`, `monto`): si un par oficial `no_encontrada` + `extra` del mismo modelo, documento y etiqueta tiene valores normalizados validos e identicos, se marca como `detectada_adicional_alta`. No se usa RapidFuzz para aceptar identificadores o montos numericos diferentes.

El matching se resuelve en tiers. El Tier 5 (`parcial / overlap_span`) recupera entidades fragmentadas cuando hay misma etiqueta y solapamiento fisico de spans, pero el overlap por si solo no alcanza: tambien debe existir evidencia textual minima. Esa evidencia se configura en `config.yaml` con `min_span_overlap_ratio`, `tier5_token_set_threshold` y `tier5_partial_ratio_threshold`. La coincidencia por Tier 5 se acepta solo si supera el overlap minimo y al menos uno de los scores textuales configurados.

`extra_fragmento` sigue contando como `extra`, no como acierto ni como nueva entidad gold. Representa una prediccion sobrante y redundante que se relaciona con una entidad gold ya detectada, por ejemplo un fragmento textual de un nombre completo. Para evitar falsos fragmentos, tambien exige misma etiqueta, overlap de span y evidencia textual suficiente; si los textos no tienen relacion, queda como `extra` normal.

## Deteccion diagnostica amplia

Ademas del matching oficial, la herramienta genera una evaluacion diagnostica separada. Esta segunda lectura no cambia precision, recall ni F1. Solo revisa entidades que oficialmente quedaron como `no_encontrada` y predicciones que oficialmente quedaron como `extra`.

La deteccion diagnostica intenta encontrar variantes asociables dentro del mismo documento y, por defecto, con la misma etiqueta. Usa reglas mas flexibles: contencion textual validada, diferencias por titulos como `Dr.` o `Dra.`, identificadores normalizados exactamente y scores reales de RapidFuzz (`token_sort_ratio`, `token_set_ratio`, `partial_ratio`). El overlap de spans es solo una senal complementaria: no alcanza por si solo.

Los niveles son `detectada_adicional_alta`, `detectada_adicional_media` y `candidata_revision`. Solo alta y media se suman al porcentaje amplio confiable; las candidatas quedan para revision manual. En los reportes tambien se separan `no_encontrada_sin_candidato`, `extra_asociable` y `extra_real`.

Ejemplo: si el gold contiene `Dra. Maria Soledad Perez` y la prediccion dice `Maria Soledad Perez`, el resultado oficial puede seguir siendo `no_encontrada` + `extra`, pero el diagnostico puede marcarlo como `detectada_adicional_alta`.

## Que significa cada metrica

La evaluacion principal excluye etiquetas opcionales configuradas. Las opcionales no penalizan si faltan, pero si aparecen correctamente se informan en `metricas_etiquetas_opcionales.csv`.

- Evaluacion estricta: cuenta `exacta_span` y `exacta_valor`.
- Evaluacion relajada: cuenta como acierto cuando hay coincidencia exacta o parcial. Es util para nombres escritos con diferencias pequenas.
- Precision: responde "de todo lo que el modelo extrajo, cuanto era correcto". Si es baja, el modelo esta agregando ruido o falsos positivos.
- Recall: responde "de todo lo que tenia que encontrar, cuanto encontro". Si es bajo, el modelo esta dejando pasar entidades.
- F1-score: combina precision y recall en un solo numero. Sirve para comparar modelos cuando se quiere equilibrio entre no inventar y no omitir.
- Cobertura: mide si la entidad esperada fue localizada de alguna forma, incluso si hubo diferencia parcial o etiqueta incorrecta. Sirve para saber si el modelo vio el dato, aunque lo haya clasificado mal.

Columnas principales:

- `total_documentos`: documentos evaluados.
- `total_entidades_gold`: entidades esperadas segun la revision manual.
- `total_entidades_predichas`: entidades extraidas por el modelo.
- `exactas`: suma de `exacta_span` y `exacta_valor`.
- `exacta_span`: etiqueta correcta, valor correcto y span correcto.
- `exacta_valor`: etiqueta correcta y valor correcto, con span distinto o no disponible.
- `parcial`: etiqueta correcta y valor parecido, pero no identico.
- `extra`: el modelo extrajo algo que no estaba en el gold.
- `no_encontrada`: el gold tenia una entidad y el modelo no la encontro.
- `etiqueta_incorrecta`: el valor aparece, pero con otra etiqueta.
- `duplicada`: el modelo repitio una entidad.

Ejemplo simple: si el gold tiene 100 entidades y el modelo extrae 80, de las cuales 60 son correctas, la precision mira 60 sobre 80 y el recall mira 60 sobre 100.

## Como interpretar los graficos

- `01_precision_recall_f1_por_modelo.png`: compara rendimiento general. Un modelo con alta precision suele inventar poco; uno con alto recall omite poco.
- `02_f1_estricto_relajado.png`: muestra si los errores son leves. Si el F1 relajado sube mucho frente al estricto, hay muchas coincidencias parecidas pero no exactas.
- `03_resultados_por_modelo.png`: muestra la composicion de aciertos y errores. Ayuda a ver si el problema principal son extras, omisiones o parciales.
- `04_metricas_por_etiqueta.png`: muestra rendimiento por etiqueta. Sirve para detectar, por ejemplo, si `persona` funciona bien pero `cbu` no.
- `05_cobertura_por_etiqueta_modelo.png`: muestra que porcentaje de cada etiqueta fue localizada por cada modelo.
- `06_comparacion_modelo_vs_regex.png`: compara modelos contra regex para identificadores y montos. Sirve para decidir si una regla simple supera o complementa al modelo.
- `07_distribucion_scores_confianza.png`: muestra los scores que entrega el modelo. Si muchos errores tienen score alto, el score no alcanza como filtro confiable.
- `08_distribucion_similitudes_rapidfuzz.png`: muestra que tan parecidos son los valores parciales. Ayuda a ajustar el umbral de RapidFuzz.
- `09_etiquetas_con_mas_errores.png`: prioriza etiquetas a revisar.
- `10_documentos_con_mas_errores.png`: ayuda a encontrar documentos dificiles, con OCR malo o formatos raros.
- `11_matriz_confusion_etiquetas.png`: muestra confusiones entre etiquetas. Lo ideal es que los valores se concentren en la diagonal.
- `12_deteccion_diagnostica_amplia.png`: compara detectadas oficiales, detectadas adicionales y no encontradas sin candidato.
- `13_extras_diagnosticos.png`: separa extras asociables de extras reales.

## Salidas

En `metricas/outputs/<tipo_documento>/<corrida>/` se generan:

- `detalle_comparaciones.csv`
- `metricas_por_modelo.csv`
- `metricas_por_modelo_opcionales.csv`
- `metricas_por_modelo_total.csv`
- `metricas_por_etiqueta.csv`
- `metricas_por_etiqueta_todas.csv`
- `metricas_etiquetas_opcionales.csv`
- `comparacion_modelos_vs_regex.csv`
- `entidades_no_encontradas.csv`
- `entidades_extras.csv`
- `coincidencias_parciales.csv`
- `entidades_duplicadas.csv`
- `etiquetas_incorrectas.csv`
- `entidades_no_detectadas_por_ningun_modelo.csv`
- `entidades_detectadas_solo_por_un_modelo.csv`
- `validaciones.csv`
- `dashboard.html`
- `dashboard.pdf`
- `run_metadata.yaml`
- `auditoria_gold.csv`
- `auditoria_invariantes.csv`
- `detecciones_diagnosticas.csv`
- `detecciones_diagnosticas_principales.csv`
- `detecciones_diagnosticas_opcionales.csv`
- `resumen_detecciones_diagnosticas_principal.csv`
- `resumen_detecciones_diagnosticas_opcional.csv`
- `resumen_detecciones_diagnosticas_total.csv`
- `resumen_amplio_por_modelo.csv`
- `no_encontradas_con_candidato.csv`
- `no_encontradas_sin_candidato.csv`
- `candidatas_revision.csv`
- `extras_asociables.csv`
- `extras_reales.csv`

`metricas_por_modelo.csv` mantiene el ranking principal historico: usa solamente las entidades principales, sin etiquetas opcionales. `metricas_por_modelo_opcionales.csv` calcula las mismas columnas agrupando solo las etiquetas opcionales configuradas para el tipo documental, como `dni`, `cuit_cuil`, `cbu`, `cvu`, `monto`, `alias` o `persona_juridica` cuando correspondan. `metricas_por_modelo_total.csv` incluye principales y opcionales para una vista global del rendimiento por modelo. Estas dos vistas nuevas son complementarias y no reemplazan el ranking principal.

En las vistas por scope, los resultados asociados a una entidad gold (`exacta_span`, `exacta_valor`, `parcial`, `no_encontrada` y `etiqueta_incorrecta`) se asignan al scope del gold. Los resultados sin entidad gold (`extra` y `duplicada`) se asignan al scope de la prediccion. Asi la cobertura de cada tabla puede reconstruirse con sus columnas visibles: `(exactas + parcial + etiqueta_incorrecta) / total_entidades_gold`.

`metricas_por_etiqueta.csv` respeta la evaluacion principal y excluye etiquetas opcionales. `metricas_por_etiqueta_todas.csv` incluye todas las etiquetas canonicas y es la tabla recomendada para revisar cobertura por etiqueta.

En `detalle_comparaciones.csv`, `gold_id` utiliza un formato numérico simple (`0`, `1`, `2`, ...) para facilitar su filtrado en planillas de cálculo, quedando vacío cuando no hay una entidad gold asociada (como en las predicciones `extra`). Para predicciones fragmentadas sobrantes (`subtipo_resultado = extra_fragmento`), `gold_id` se mantiene vacío para no alterar `total_entidades_gold`, conservando la referencia en `gold_id_relacionado`.

En `detalle_comparaciones.csv`, `score_rapidfuzz` solo se completa cuando el metodo de matching fue RapidFuzz, es decir, para coincidencias `parcial`. En coincidencias exactas o por valor normalizado queda vacio. La columna `metodo_matching` indica que regla produjo la asociacion.

`run_metadata.yaml` registra la fecha, el gold usado, los archivos de modelos, el umbral, la tolerancia y las rutas de salida.

`auditoria_gold.csv` muestra cada fila del gold revisado, si entra en la metrica principal, si es obligatoria u opcional y el motivo de exclusion cuando corresponde.

`auditoria_invariantes.csv` verifica reglas basicas de consistencia: mismo total gold obligatorio entre modelos, cada entidad gold clasificada exactamente una vez y cada prediccion clasificada exactamente una vez.

`detecciones_diagnosticas.csv` contiene la evaluacion amplia. Conserva el resultado oficial por separado y agrega datos como nivel de confianza, regla principal, cantidad de senales, identificadores normalizados, ratios de RapidFuzz, diferencia de longitud, overlap de spans, contencion textual y motivo de deteccion. Los resumenes principal, opcional y total (`resumen_detecciones_diagnosticas_*`) incluyen una columna `extra_fragmento` para contabilizar estas predicciones fragmentadas de forma transparente sin alterar la metrica oficial ni inflar el inventario gold.

`resumen_amplio_por_modelo.csv` es una vista ejecutiva complementaria para el universo total. Calcula `detectadas_amplias` como entidades gold detectadas oficialmente (`exacta_span`, `exacta_valor`, `parcial`) o recuperadas por diagnostico confiable (`detectada_adicional_alta`, `detectada_adicional_media`). `candidatas_revision` queda visible pero no cuenta como acierto, `no_encontradas_reales` usa `no_encontrada_sin_candidato`, y `extras_reales` excluye `extras_asociables` y `extra_fragmento`. Sus metricas `precision_amplia`, `recall_amplio` y `f1_amplio` no reemplazan precision, recall ni F1 oficiales.

El dashboard se abre localmente desde `metricas/outputs/<tipo_documento>/<corrida>/dashboard.html` y contiene explicaciones breves para las metricas y graficos.

`dashboard.pdf` se genera automaticamente en la misma carpeta de la corrida. Si falta `reportlab` o `pypdf`, la herramienta muestra una advertencia clara, deja un `pdf_status.txt` con el motivo y no interrumpe la generacion de CSV, graficos y HTML.

Para regenerar manualmente solo el PDF de una corrida existente, sin recalcular metricas:

```bash
python metricas/src/generar_pdf_dashboard.py --run-dir metricas/outputs/embargo/<corrida>
```

Si la carpeta de graficos no esta registrada correctamente en `run_metadata.yaml`, se puede indicar:

```bash
python metricas/src/generar_pdf_dashboard.py --run-dir metricas/outputs/embargo/<corrida> --graph-dir metricas/graficos/embargo/<corrida>
```

Las carpetas `metricas/outputs/embargo/`, `metricas/outputs/oficio/`, `metricas/graficos/embargo/` y `metricas/graficos/oficio/` funcionan como contenedores de corridas. No deberian tener CSV, HTML o PNG sueltos directamente adentro; esos archivos deben quedar dentro de una carpeta de corrida.

La seccion `Muestras de errores` del dashboard no reemplaza a los CSV completos. Muestra:

- un resumen con cantidad de errores por modelo y tipo;
- una muestra balanceada de hasta 3 casos por modelo y tipo de error.

Las columnas gold y predichas se conservan aunque esten vacias, porque esos vacios explican el error:

- En `no_encontrada` quedan vacias las columnas predichas porque el modelo no extrajo la entidad gold.
- En `extra` quedan vacias las columnas gold porque la prediccion no pudo asociarse a ninguna entidad validada.
- En `duplicada` quedan vacias las columnas gold porque el duplicado se identifica comparando predicciones entre si antes del matching contra el gold. La fila representa la prediccion repetida descartada.
- En `etiqueta_incorrecta` se muestran gold y prediccion porque el valor coincide, pero las etiquetas difieren.

Para duplicadas, `subtipo_resultado` distingue `duplicado_mismo_valor_mismo_span`, `duplicado_overlap_chunks` y `duplicado_dudoso_sin_span`. El reporte `entidades_duplicadas.csv` incluye la prediccion conservada (`pred_id_original`, `valor_original_conservado`, spans originales) y la prediccion descartada (`pred_id_duplicada`, `valor_duplicado`, spans duplicados).

Esto evita que el dashboard muestre solo los primeros errores del archivo, que podrian pertenecer a un unico modelo. Para auditar todos los casos hay que usar los reportes completos como `entidades_no_encontradas.csv`, `entidades_extras.csv`, `entidades_duplicadas.csv` y `etiquetas_incorrectas.csv`.

## Validaciones

La herramienta avisa en `validaciones.csv` sobre:

- archivos inexistentes;
- columnas requeridas faltantes;
- spans no numericos o invalidos;
- documentos presentes solo en gold;
- documentos presentes solo en predicciones.

Los valores vacios de documento, etiqueta o valor se descartan antes de comparar para evitar falsos matches.

## Prueba sintetica

Para validar sin datos reales:

```bash
python metricas/src/run_evaluacion.py --gold metricas/tests/gold_sintetico.csv --results metricas/tests/pred_gliner2_sintetico.csv metricas/tests/pred_pii_v1_sintetico.csv --outdir metricas/outputs --graph-dir metricas/graficos --doc-type embargo
```
