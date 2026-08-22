# Sesión de mapeo: RTAB-Map (odometría/loop-closure) + PersistentMap (grilla)

Arma un mapa reusable a partir de una corrida del rover, combinando dos
piezas que resuelven problemas distintos:

| Pieza | Rol | Corre en |
|---|---|---|
| RTAB-Map | corrige la deriva de la odometría de rueda con cierre de bucles visual (`TF map -> odom`) | ROS2 (esta carpeta) |
| `MapSessionBridge` (genie_rover) | arma la grilla de obstáculos real, con la percepción SAM-TP en vivo, y la exporta a `.yaml`+`.pgm` | Python directo al SDK (`genie/genie_rover/Indoor/map_session.py`) |

**Por qué separadas:** la cámara es RGB monocular (sin profundidad ni
lidar). RTAB-Map con mono solo puede hacer *odometría visual y cierre de
bucles* de forma confiable — no tiene con qué reconstruir una grilla de
ocupación 2D navegable. Esa grilla la arma, en cambio, `PersistentMap` con
la propia percepción del rover (que sí sabe estimar transitabilidad desde
una sola imagen). RTAB-Map le presta a esa grilla una pose con menos deriva
que la odometría cruda.

## 1. Aplicar el parche a `persistent_map.py`

Una sola vez: seguí las instrucciones en
`genie/genie_rover/persistent_map_export_addon.py` para agregar el método
`export_ros_map()` a la clase `PersistentMap`.

## 2. (Opcional) Calibrar/confirmar odometría y cámara

- `wheel_radius_m`, `track_width_m`, `rotation_sign`: los mismos valores que
  ya tengas calibrados en tu config de genie_rover (`genie/configs/*.yaml`,
  sección `odometry:`). Si nunca los calibraste, ver
  `genie/tools/calibrate_wheel_scale.py` e `identify_wheels.py`.
- `fx, fy, cx, cy, width, height`: la misma matriz que uses en
  `camera.intrinsics` de tu config de genie_rover. Si no tenés una
  calibración real, `camera_info_publisher.py` **falla a propósito** en vez
  de inventar una — pasale algo razonable (`fx=fy≈ancho_px`,
  `cx=ancho/2, cy=alto/2`) sabiendo que el registro va a ser menos preciso.

## 3. Levantar la sesión ROS2 (RTAB-Map + bridge)

```bash
# SDK corriendo aparte (hypercorn main:app)
ros2 launch rtabmap_mapping.launch.py \
    sdk_url:=http://localhost:8000 \
    wheel_radius_m:=0.0527 track_width_m:=0.15 \
    camera_width:=1280 camera_height:=720 \
    camera_fx:=900.0 camera_fy:=900.0 camera_cx:=640.0 camera_cy:=360.0 \
    database_path:=$HOME/maps/sesion1.db
```

Dejalo corriendo. Vas a ver en el log de `rtabmap` cuántos *loop closures*
va encontrando a medida que el rover recorre el lugar dos veces por el mismo
pasillo.

## 4. Correr el programa de mapeo (genie_rover)

En otra terminal, en el entorno donde ya corrés `bridge.py`/`indoor_bridge.py`:

```bash
cd genie
# simulacro primero, para ver que arranca bien sin mover el robot
python -m genie_rover.Indoor.map_session --config configs/indoor_mapping.yaml

# de verdad, con correccion de RTAB-Map (mapping.rtabmap_correction.enabled:
# true en el config, o editalo antes de correr)
python -m genie_rover.Indoor.map_session --config configs/indoor_mapping.yaml \
    --go --max-seconds 300 --map-out maps/sesion1
```

El rover va a explorar solo (frontera: hacia lo desconocido más cercano)
evitando obstáculos, igual que hace `indoor_bridge.py` normalmente, pero sin
buscar ningún cono. Cada `export_every_s` segundos, y siempre al final
(tiempo agotado, Ctrl+C, o error), guarda `maps/sesion1_NNN.yaml`+`.pgm`.

Recorré el lugar volviendo a pasar por zonas ya visitadas al menos una vez
— sin loop closures, RTAB-Map no tiene nada que corregir.

## 5. Reusar el mapa con tu programa

Usá el `.yaml` final (`maps/sesion1_final.yaml`) como `external_map` en
cualquier corrida de `indoor_bridge.py` (por ejemplo la misión real de
cono):

```yaml
memory:
  external_map:
    enabled: true
    yaml_path: "maps/sesion1_final.yaml"
    start_pose_map:
      x_m: 0.0      # donde estás parado AHORA, medido en el mapa de arriba
      y_m: 0.0
      yaw_deg: 0.0
```

Recordá la limitación ya documentada en `external_map.py`: sin
relocalización, `start_pose_map` tiene que reflejar de verdad dónde arranca
el robot en ESTE mapa, medido a mano una vez.

## Límites conocidos

- La grilla exportada es **binaria** (libre / ocupado / desconocido) — la
  incertidumbre continua de `PersistentMap.value` se redondea al exportar,
  es un límite del formato `.yaml+.pgm`, no del mapeo en sí.
- Si `rtabmap_correction.enabled: false` (o ROS2 no está disponible), todo
  funciona igual pero usando sólo dead-reckoning rueda+giro — sirve para
  sesiones cortas en un espacio chico donde la deriva no llega a importar.
- RTAB-Map con cámara mono nunca corrige **escala** por su cuenta (por eso
  se lo alimenta con la odometría de rueda, que sí tiene escala métrica
  correcta, en vez de dejar que haga VO monocular puro).
