# MPPI opcional para el bridge del Earth Rover

Dos selectores independientes en `configs/frodobot_rover.yaml`:

```yaml
navigation:
  trajectory_algorithm: mppi  # polynomial por defecto
safety:
  recovery_algorithm: mppi    # legacy por defecto
```

Se pueden combinar libremente: polynomial/legacy mantiene el comportamiento
anterior; mppi/legacy cambia solo el planificador; polynomial/mppi cambia solo
la recuperacion; mppi/mppi cambia ambos. Sin selectores, el bridge conserva
el comportamiento anterior y no importa ni instancia los modulos MPPI.
Los selectores pertenecen al bridge online; el CLI offline `run_path_planner.py`
sigue usando polinomicas. El bloque `planner` anterior permanece intacto.
También se puede elegir `trajectory_algorithm: nomad` y combinarlo con recovery
MPPI; ver [README_NOMAD.md](README_NOMAD.md). En esa combinación la planificación
normal usa las propuestas NoMaD y el MPPI solamente interviene en recuperación.

## Ejecucion

Desde `genie/`, editar los selectores del YAML y ejecutar primero:

```bash
python3 -m genie_rover.bridge --config configs/frodobot_rover.yaml --max-seconds 30 --debug-dir debug/mppi
```

Es dry-run: necesita SDK/camara y SAM-TP, pero no manda movimiento. Las pruebas
unitarias de abajo no necesitan ninguno de ellos. Los archivos `*_mppi.npz`
guardan `path_xy_m` (derecha, adelante), `controls` (m/s, rad/s) y costo.

Para `--go`, medir `mppi_actuation.linear_mps_per_unit` y
`angular_rps_per_unit` y luego poner `calibrated: true`. Los valores 1.0 son
placeholders, no una calibracion del Mini. Las escalas representan velocidad
fisica dividida por magnitud de comando SDK en el rango utilizado. La relacion
puede ser no lineal: este adaptador presupone aproximacion lineal local.
`navigation.angular_sign` conserva su convencion anterior. El arranque rechaza
limites metricos que excedan las saturaciones normalizadas del SDK/navigation.

## Modulos y flujo

- `genie_rover/mppi.py`: optimizador NumPy sin I/O, rollout diferencial,
  muestreo perturbado alrededor del plan anterior y semillas de giros/avances.
  Pondera por `exp(-(costo-minimo)/temperatura)`, comprueba la combinacion y
  conserva el mejor candidato si la combinacion colisiona o cuesta mas.
- `genie_rover/mppi_adapter.py`: lectura del BEV fresco y mapa persistente,
  conversion metrica al SDK, selectores y estado de recuperacion.
- `genie_rover/bridge.py`: orquestacion, adquisicion de observaciones,
  seleccion del backend y ejecucion de un unico pulso por observacion.

MPPI produce tanto una trayectoria como controles fisicos. El bridge utiliza
su primer control directamente; no lo vuelve a pasar por `PathFollower` ni por
la histeresis lateral, porque eso modificaria la trayectoria validada. El flujo
polinomico conserva ambos modulos.

El optimizador simula [adelante, izquierda, theta antihorario], en metros y
radianes. La salida de visualizacion conserva [derecha, adelante]. Consulta
`PersistentMap` con la pose actual, incluyendo laterales y parte trasera; las
celdas frescas observadas tienen prioridad. No se usa el resize cuadrado del
planner polinomico para calcular distancias MPPI.

## Colisiones y ejecucion

Se comprueba un disco que debe circunscribir toda la huella del rover, tanto
en los puntos de control como en subpasos para reducir saltos sobre obstaculos.
Todas las celdas muestreadas fuera del disco inicialmente ocupado deben ser
conocidas y superar `min_traversability`. Solo lo desconocido debajo del disco
actual se exceptua; un obstaculo conocido ahi invalida incluso un giro. Es una
comprobacion discreta sobre un mapa estimado, no una garantia fisica de seguridad.
Elegir `collision_step_m` segun la resolucion/obstaculos; la cobertura de camara
puede impedir arrancar si la zona alrededor de la huella es desconocida.

Al seleccionar cualquiera de los backends MPPI, el bridge detiene el rover
antes de adquirir la observacion. Cada accion MPPI dura como maximo el pulso
`dt` previsto y termina con un envio de freno en `finally`; despues se vuelve
a observar. El intervalo real incluye percepcion, optimizacion, red y pulso:
`1/dt` NO es la frecuencia efectiva. Esta primera version es deliberadamente
stop-observe-plan-act y su modelo cinemático no incluye aceleraciones ni inercia.
Incluso con planner polinomico y recovery MPPI se frena al inicio de cada
observacion para mantener coherencia del mapa del recovery. Con ambos selectores
originales este cambio no existe.

Se rechazan observaciones que envejecieron durante adquisicion/optimizacion
mas de `max_observation_age_s`. Se conserva el detector de video congelado.
El freno es software: una caida de proceso/red o la inercia real requieren un
watchdog y parada del robot; el SDK conserva su ultimo comando.

## Recovery

Se activa ante planes vacios repetidos, giros sin avance o bloqueo frontal
persistente. Un episodio elige una submeta entre adelante, izquierda, derecha
y diagonales traseras, usando transitabilidad/conocimiento del mapa y penalizando
direcciones ya intentadas. Mantiene la submeta en coordenadas del mundo durante
`attempt_steps` si hay odometria; cada pulso se recalcula con nueva observacion.
La meta GPS cede temporalmente prioridad a salir del bloqueo.

Con el frente bloqueado se prohibe velocidad positiva: solo se consideran giro
y retroceso verificado contra el mapa. Sin memoria trasera conocida se descarta
la reversa. Si no existe movimiento valido, frena y prueba otra submeta en la
siguiente observacion; nunca cae al avance forzado original. Se vuelve a normal
cuando el frente esta libre y se detecto `progress_m` de traslacion o
`progress_rad` de rotacion desde el inicio. Sin odometria, solo puede usar el
frente libre como criterio de salida. Tras `max_steps`, se detiene el bridge
para evitar intentos indefinidos. Esta estrategia acotada no garantiza resolver
callejones, errores de mapa ni bloqueos fisicos.

## Pruebas offline

Desde la raiz del repositorio:

```bash
PYTHONPATH=genie python3 -m unittest discover -s genie/tests -v
```

Cubren combinaciones de selectores, comportamiento legacy, integracion del
bridge con SDK/percepcion simulados, limites/signos, avance, obstaculos,
reversa desconocida, colision entre pasos, observaciones vencidas, freno y
presupuesto de recuperacion. No sustituyen calibracion ni pruebas fisicas.
# Recovery original (`legacy`)

El recovery original alterna barridos derecha/izquierda entre llamadas, usando
`navigation.angular_sign` para convertir el sentido al comando SDK. Cada barrido
dura como maximo `safety.recovery_turn_s` en la espera local y termina intentando
enviar un freno, incluso si falla el envio o se solicita parar. En dry-run no espera.
No evalua cual lado es mas transitable ni verifica la huella durante el giro;
puede oscilar entre orientaciones sin encontrar salida. No reemplaza al recovery
MPPI basado en mapa ni garantiza seguridad frente a fallos de transporte.
La seleccion actual del YAML sigue siendo MPPI; para probar el original usar
`safety.recovery_algorithm: legacy`.
