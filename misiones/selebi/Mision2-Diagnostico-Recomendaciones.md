# Rover atascado contra la reja — plan de acción priorizado

Diagnóstico y plan sobre el log de la corrida (frames 0006–0247, cp#1 → cp#3).
Repo: `IROS26-LaRovernetta`, rama `fixVelocidad-Tom`.

---

## Resumen del problema

El rover se traba contra una reja y sigue insistiendo por el mismo lugar. No es
un solo bug: son cuatro causas encadenadas, y la de más abajo invalida a todas
las de arriba.

| # | Causa | Evidencia en el log |
|---|-------|---------------------|
| 1 | La odometría no avanza | 250 frames con `linear≈1.0`, pose va de (0,0) a (+2.76,−0.68) mientras el GPS avanza ~39 m |
| 2 | El mapa persistente no acumula nada | `mapa: N celdas` converge a 2994 = `celdas BEV=2994` |
| 3 | La recuperación elige "adelante" (la reja) | `elijo rumbo +0 grados por mapa (libre=5%)` con −90° en 17–100% |
| 4 | El giro de escape se aborta a mitad | `frente libre por mapa tras girar +45 grados, corto` |

Ordenado por *causa raíz primero*: cada bloque asume que el anterior está
hecho, porque tunear la recuperación con el mapa roto es perder el tiempo.

---

## P0 — Odometría (bloquea todo lo demás)

Sin pose que avance, el mapa persistente no puede existir, y sin mapa la
recuperación decide a ciegas.

1. **Medir el error de escala.** Andar 10 m rectos en manual y comparar
   `odometry.pose` contra el desplazamiento GPS. En el log: 250 frames, el GPS
   avanzó ~39 m al cp#2, la pose reporta 2.7 m. Factor ~14x — eso no es deriva,
   es una constante mal puesta.
2. **Revisar en `genie/configs/frodobot_rover.yaml` → `odometry`:**
   `wheel_radius_m: 0.0527`, `track_width_m: 0.15`, y sobre todo
   `left_rpm_indices` / `right_rpm_indices`. Un factor de ~14 huele a unidades:
   RPM vs rad/s (×2π/60 = 0.105 → ~9.5x), o a `dt` mal calculado por frame.
3. **Verificar `gps_correction: true`.** Está activo con
   `min_gps_displacement_m: 1.0` y aun así la pose no se corrigió en 250 frames.
   O nunca dispara, o corrige y algo la pisa. Poner un print del delta que aplica.
4. **Criterio de aceptación:** correr 20 m rectos y que `pose` cierre dentro del
   10% del GPS. No pasar a P1 sin esto.

---

## P1 — Que el mapa persistente sirva de algo

Archivos: `genie/genie_rover/persistent_map.py`, `genie/genie_rover/bridge.py`

5. **Rastro libre.** En el loop principal, después de `integrate()`, marcar la
   huella del robot en la pose actual como `value=1.0, conf=1.0` (un disco del
   ancho del chasis). Es la única forma de tener cobertura *detrás*: hoy
   `integrate()` calcula `x_robot = (h - 1 - filas) * forward_range`, que es
   siempre ≥ 0, así que ninguna observación escribe jamás una celda detrás del
   robot. Esto habilita el retroceso, que en todo el log no corrió ni una vez
   (`mapa detras del robot: libre=0% cobertura=0%`, siempre).
6. **Decay por distancia, no por tiempo.** En `_apply_decay`, reemplazar `dt`
   por metros recorridos desde la última llamada (o congelar el decay cuando el
   comando lineal es 0). Con `decay_per_s: 0.08` y `min_confidence: 0.15`, una
   celda que nadie vuelve a observar cae bajo el umbral en
   `ln(1/0.15)/0.08 ≈ 24 s`. Como el decaimiento corre por reloj, se borra la
   memoria justo mientras el robot está frenado contra el obstáculo — o sea,
   exactamente cuando más la necesita.
7. **Parche rápido mientras tanto:** `min_confidence: 0.05`, `decay_per_s: 0.02`.
8. **Criterio de aceptación:** el contador `mapa: N celdas` debe quedar
   claramente por encima de `celdas BEV=2994` mientras el robot está quieto. Si
   converge a 2994, la memoria sigue sin aportar nada.

---

## P2 — Selección del rumbo de escape

Archivos: `genie/genie_rover/bridge.py::_recover_informado`, `frodobot_rover.yaml`

9. **Sacar el 0° de `recovery_headings_deg`** → `[90.0, -90.0, 180.0]`. O al
   menos excluirlo cuando la llamada viene de `_retroceso_y_recover`, donde ya
   se sabe que el frente está bloqueado. Hoy elegir 0° **no manda ningún
   comando**: por eso hay 20 frames seguidos de `OBSTACULO al frente` sin que
   pase nada.
10. **Agregar `recovery_min_libre_pct: 55`.** Hoy es un argmax sin piso: eligió
    `+0` con `libre=5%`. Si ningún candidato pasa el piso, caer al VLM o al
    barrido ciego en vez de reconfirmar la pared.
11. **Tratar lo no observado como neutral, no como descalificado.** Con
    `recovery_min_cobertura_pct: 25` los laterales (16–25%) quedan afuera y el
    frente (70–80%) gana siempre — el sesgo es puramente "hacia dónde miró la
    cámara". Un lateral desconocido es mejor apuesta que un frente
    conocido-bloqueado. Bajarlo a ~10, o ponderar `libre × f(cobertura)` en vez
    de filtrar duro.

---

## P3 — El giro y el anti-bucle

12. **Que `_girar_hacia` no corte antes de completar el giro comprometido.** Hoy
    hace `frente libre por mapa tras girar +45 grados, corto`: consulta el
    **mismo mapa** que acaba de equivocarse, corta a mitad, y el follower lo
    reapunta al goal (que está detrás de la reja). Exigir giro mínimo ≥90° y
    verificar con un BEV **fresco de cámara**, no con el mapa.
13. **Lista de rumbos ya intentados**, con timeout de unos segundos, para no
    reelegir el que acaba de fallar.
14. **Contador de recuperaciones sin desplazamiento neto:** si hubo ≥3 recoveries
    y la pose se movió <0.5 m, escalar a P4.

---

## P4 — Modo pared (el problema de fondo con una reja)

15. Una reja es un obstáculo **largo**. Girar 45° y volver a apuntar al
    checkpoint siempre devuelve a la reja — es exactamente el ciclo de los
    frames 0006–0247. Al escalar (paso 14): fijar un lado, **ignorar el rumbo al
    goal** y avanzar paralelo al obstáculo unos 3–5 m antes de volver a
    considerar el checkpoint.
16. Latch complementario: no re-apuntar al goal hasta acumular X metros de
    progreso lateral.

---

## Verificación

17. `test_near_regime_offline.py` ya ejercita `_map_free_and_coverage`,
    `_retroceder`, `_girar_hacia` y `_recover_informado` sin robot ni GPU.
    **Agregar un caso con una pared larga** (no el obstáculo de 0.6×0.6 m
    actual) y asertar que el rumbo elegido no es 0° y que el retroceso
    efectivamente corre.
18. `python -m genie_rover.persistent_map` para los cambios de P1 — agregar un
    assert de que el rastro libre queda detrás del robot.

---

## Notas sueltas del log

- `[percep] frame 1024x576 != calibracion 1920x1080: reescalo K` — el reescalado
  de K es correcto para fx/fy/cx/cy, pero la distorsión de lente no se reescala.
  A revisar si el BEV cercano sale deformado.
- `error en la iteracion (1/5): 500 Server Error: /control` — un 500 aislado del
  SDK en el frame 0067. No es la causa del atasco, pero conviene loguear el
  cuerpo de la respuesta por si se vuelve frecuente.
