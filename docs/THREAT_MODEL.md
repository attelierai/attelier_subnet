# Modelo de amenazas de la evaluación automática

El validador ejecuta localmente el render, el juez Luna y este gate. Sus JSON,
vídeos, recibos y el plan de ronda son evidencia de ese operador local. Un minero
no debe poder aportar un checkpoint privado, alterar el panel prescrito, sustituir
el vídeo candidato entre comparaciones, reutilizar una respuesta del juez, ni
convertir una admisión posterior al cierre en una candidatura de la ronda.

El gate exige snapshots públicos en producción, hashes de vídeo idénticos para el
candidato frente a golden e incumbent, dos órdenes A/B opuestas por slot, controles
recalculados, un panel de producción resuelto desde su sealed pack y `response_id`
único globalmente entre calidad y controles. El plan de confirmación incluye el
candidato bloqueado, su hotkey y bloque de admisión, las admisiones hasta
`close_block`, y el manifiesto/estado certificado del incumbent. La decisión sólo
acepta el candidato si esos campos coinciden con la evidencia de calidad y timing.

La admisión pública implementada no confía en la lista local `plan.admitted`:
el planificador, el publicador de locks y el ancla reconstruyen los commitments de
modelo del snapshot finalizado C, vuelven a resolver sus revisiones HF inmutables y
aplican la misma orden, deduplicado y cupo. Un candidato omitido, posterior al
cierre o un manifiesto sustituido no puede pasar esa comprobación. El lock de
finalista también está implementado y cubierto por replay: se verifica el primer
lock válido de comité tras C y un lock desconocido en esa ventana falla cerrado.
Esto evita reinterpretar la ronda cuando un snapshot deja de resolverse, pero
impide solapar locks de rondas distintas en la misma ventana.

Estas garantías siguen siendo estructurales y de evidencia, no consenso. La
operación testnet observada sólo ha llegado a duelo; los locks de confirmación y
sus replays de código no prueban una publicación finalizada en vivo. Además, la
admisión al cierre no prueba que el artefacto siga disponible: un propietario puede
borrar o gatear la revisión después. No existe una ruta operativa de pesos reales
hoy; cualquier activación futura debe exigir evidencia fresca sin credenciales de
resolución, licencia y fetch verificado de todos los archivos del ganador justo
antes de enviar pesos. Si falla, su participación queda sin asignar y no revive una
corona anterior. Una firma o
recibo controlado por el operador tampoco prueba que render, medios o percepción
de Luna sean honestos. Faltan un estado certificado del incumbent y un quórum de
validadores independientes que reproduzcan o acuerden la evaluación antes de una
WTA, crown o pesos. Un comité malicioso continúa fuera del alcance de este gate.

En producción el panel procede de un `forge/sealed-pack/v1`: se compromete su hash
antes del cierre, se revela después de la entropía y se selecciona de forma
determinista. Los packs sellados reducen el sobreajuste a una gramática pública
finita, pero no prueban secreto, novedad semántica ni ausencia de filtración. Las
gramáticas procedurales v1--v3 sólo reproducen recibos históricos. La admisión del
snapshot debe preceder a la revelación. La no reutilización de paneles de
confirmación sigue siendo un requisito operativo pendiente: sólo se comprueba
unicidad dentro de cada pack. El código no consulta un historial de textos o
`source_group` ya revelados y permite resellarlos con otro contexto y salt.
Antes de activar premios, hace falta un historial auditable y una regla de
no reutilización que los validadores independientes comprueben.

Las políticas históricas v1 y v2 permanecen sólo para replay diagnóstico: v1
promediaba las órdenes y hacía dura la discrepancia superior a 0.25; v2 tomaba el
mínimo por eje. V3 usa la media simétrica de los dos deltas por eje y agrupa las
semillas dentro del prompt antes de su intervalo: los dos órdenes nunca cuentan como
dos prompts independientes. La magnitud del desacuerdo y los cambios de signo se
persisten por eje y panel; no se esconden ni se convierten en una penalización mínima.
Un esquema incompleto, incertidumbre o evidencia de imágenes A/B que no se invierte sigue
siendo `INCOMPLETE`. Sus controles machine-known, recalculados y frescos aún deben
pasar antes de una promoción; no hay calibración v3 aprobada.

El mínimo histórico tenía un coste estadístico: `min(X,Y) = (X+Y)/2 - |X-Y|/2`.
Penalizaba todo desacuerdo entre órdenes, incluido el ruido del juez. Los controles
idénticos no estiman ese sesgo entre vídeos distintos. En V3 una transformación
non-self con píxeles exactamente iguales al origen se marca `N/A`: no es detección ni
fallo y reduce el tamaño informativo; una familia con menos de 32 fuentes informativas
queda inconclusa. La sensibilidad a corrupciones prescritas no sustituye una validación
de equivalencia ni justifica ajustar la política después de ver el panel.

El control preregistrado del 2026-10-04 comparó ocho pares del mismo base8 sin
modificar, con semillas independientes y asignación de candidato prefijada al
azar. El promedio entre órdenes fue -0.0390625; el mínimo, -0.125. La diferencia
0.0859375 cuantifica aquí la penalización adicional por desacuerdo entre órdenes.
No es una tasa calibrada de falsos rechazos: dos semillas pueden producir vídeos
de calidad individual distinta y la muestra es pequeña. La política sigue
congelada, sin ajustar márgenes ni umbrales para convertir el resultado en PASS.

Antes del cupo de admisión, el planificador descarta únicamente `full_transformer`
con el mismo multiconjunto de blobs `safetensors` (SHA-256 y tamaño), la misma base
y la misma receta normalizada. Conserva la admisión con bloque más temprano; empates de bloque
se resuelven por hash de manifiesto y hotkey, no por un orden de llegada no probado.
Expone duplicados tempranos, overflow y entradas pendientes de preflight. Es una
defensa contra forks públicos byte-idénticos que intenten consumir las 16 plazas;
no equivale a una deduplicación completa. Los LoRA, sus escalas/configuración y
las equivalencias algebraicas o de repacking siguen siendo distintos hasta el
preflight de tensores. Un atacante con muchas hotkeys también puede agotar la cola
antes del preflight mediante pesos distintos; resolverlo requiere una cola cerrada,
preflight y backfill verificables antes de fijar el cupo.

## Cobertura espacial e integridad raster congeladas

El evaluador v3 entrega a Luna seis imágenes por lado: tres hojas cronológicas
que cubren los 81 frames, una hoja de nueve frames nativos secretos posteriores al
render, tres crops 4x `NEAREST`, y nueve transiciones secretas estratificadas. Cada
transición muestra `t`, `t+1` y el delta absoluto; es evidencia para localizar una
discontinuidad, nunca una puntuación automática de movimiento. Las coordenadas e
índices se registran y se aplican idénticamente a ambos lados. El gate verifica
cobertura por estrato, hashes de entrada y la inversión A/B entre órdenes. El paquete
ocupa 29,886 patches por par, bajo el techo de 30,000. Su nueva identidad no reescribe
los recibos v1/v2 congelados.

Esto redujo un punto ciego observado: las revisiones históricas de tres frames
fijos y de nueve nativos no detectaron un caso de un píxel. En probes congelados
codificados y decodificados, `checker1`, `checker2` e `image-instructions` quedaron
`COMPLETE`; `checker1` activó `RASTER_INTEGRITY_ANOMALY` y su fuente fue `CLEAR`,
mientras que los otros dos quedaron `CLEAR` para la regla raster. Los recibos son
`.codex/receipts/frozen-probe-{checker1,checker2,image-instructions}.json`.

La regla raster revisa todos los frames con energía Nyquist repetida por tiles. Es
un guard conservador de integridad, no un juicio perceptual: una anomalía deja la
evidencia `INCOMPLETE` antes de gastar en Luna y nunca aplica una penalización al
minero. Puede producir falsos positivos y no cubre ataques semánticos, de baja
frecuencia ni adaptativos. Un probe por familia no estima sensibilidad.

La campaña congelada del 2026-10-04 completó 32 prompts distintos de los ocho
prompts del duelo, con cinco controles por fuente y dos órdenes por comparación
(320 llamadas). El resultado es **FAIL**: detectó 19/32 controles `shuffle`, cuyo
límite superior binomial unilateral ajustado por cinco familias es 0.7881, por
debajo de la sensibilidad exigida de 0.8. Los otros resultados fueron 0/32 falsas
alarmas `self`, 31/32 `freeze`, 32/32 `blur` y 32/32 `wrong_video`, sin respuestas
inválidas. El recibo privado es `.codex/receipts/overnight-controls-32.json`.
Es un resultado histórico v2, no una medición del nuevo juez. La promoción sigue
bloqueada porque v3 aún no tiene calibración Wan aprobada. El probe v3 completado
(`temporal-v3-probe-result.json`) usa ocho escenas procedurales de una familia,
24 pares construidos y 48 llamadas. Detectó 4/4 congelaciones incorrectas, 4/4
desórdenes temporales y 4/4 direcciones invertidas, con 0/12 falsas alarmas entre
copias idénticas y escenas correctamente estáticas; ningún par inválido. Cumple
su criterio preregistrado de diagnóstico, no demuestra sensibilidad en vídeo
natural ni autoriza promoción. No se ajustó su política después de ver resultados.

La repetición posterior con v3 de los 32 vídeos Wan históricos marcó 10/32 casos,
sin pares inválidos. Aplicar la misma media de órdenes a las respuestas antiguas
da 11/32; no se observa mejora en este material. Es una prueba de desarrollo con
fuentes conocidas, no una calibración independiente ni evidencia de que los 22
casos no marcados sufran todos una degradación visible. No se retocó el juez de
nuevo ni se sustituyeron los recibos originales para presentar un resultado positivo.

Una prueba temporal posterior, explícitamente exploratoria, volvió a consultar
los dos primeros falsos negativos con instrucciones centradas en continuidad:
ambos siguieron sin detectarse. Un control CPU de interpolación lineal aumentó su
error en los 32 vídeos desordenados, incluidos los 13 fallos de Luna. Farnebäck
aportó poco sobre esa señal lineal; un vídeo congelado obtuvo error casi cero.
Por tanto esa métrica tampoco representa calidad por sí sola. No se ha elegido
un umbral después de ver los resultados ni se ha incorporado al gate. Cualquier
nueva política exige controles frescos, movimiento legítimo difícil y pruebas de
equivalencia entre realizaciones distintas antes de activarse.
