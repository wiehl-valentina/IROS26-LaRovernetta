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

**Ya está hecho**: `PersistentMap.export_ros_map()` está en
`genie/genie_rover/persistent_map.py`. `persistent_map_export_addon.py` quedó
sólo como referencia histórica — no hay que volver a pegar nada.

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

**Forma recomendada** — un solo comando que saca TODOS los números del mismo
yaml de genie que va a usar `map_session.py`, así los dos lados no divergen:

```bash
# SDK corriendo aparte:  ./rover_launch.sh sdk
./rover_launch.sh mapping-ros2 --db ~/maps/sesion1.db \
    --config genie/configs/indoor_mapping.yaml
```

Deriva y pasa: intrínsecos (`fx/fy/cx/cy`, tamaño), **coeficientes de
distorsión**, radio de rueda, ancho de trocha, signo de rotación, índices de
rpm, uso de giróscopo, corrección GPS y los extrínsecos de cámara
(altura/pitch/offset, calculados de la matriz `camera.pose` 4x4 si está).
Para ver qué va a mandar, sin lanzar nada:

```bash
python3 Indoor_Instalacion_SDK_SLAM/ros2/config_to_ros_params.py \
    --style pretty genie/configs/indoor_mapping.yaml
```

**Forma manual** (equivalente, si preferís `ros2 launch` a mano):

```bash
cd Indoor_Instalacion_SDK_SLAM/ros2/mapping
ros2 launch rtabmap_mapping.launch.py \
    database_path:=~/maps/sesion1.db \
    $(python3 ../config_to_ros_params.py ../../../genie/configs/indoor_mapping.yaml)
```

Argumentos útiles del launch:

| Argumento | Default | Para qué |
|---|---|---|
| `database_path` | `~/maps/rtabmap_mapping.db` | se expande `~` y se crea la carpeta padre antes de arrancar |
| `delete_db_on_start` | `true` | ponelo en `false` para **continuar** una base existente en vez de borrarla |
| `camera_d` | distorsión del Mini | `[k1,k2,p1,p2,k3]`; `[]` = lente perfecta |
| `feed_fps` | `15` | FPS que le pide al `/feed` del SDK |

El launch imprime al arrancar a qué archivo va a escribir, así que si el `.db`
no aparece donde esperabas te enterás en el momento y no al final.

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
