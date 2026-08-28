# Mapeo SLAM (RTAB-Map) — grabar, exportar y usar en modo indoor

Guía práctica del flujo completo: levantar el stack, grabar una sesión de
mapeo mientras manejás el rover a mano desde el dashboard del SDK, exportar
el mapa 2D, y usarlo después con `indoor_bridge.py` — incluyendo el caso en
que no controlás dónde arranca el rover (relocalización).

---

## 0. Prerequisitos (una sola vez)

### WSL2 / ROS2 (si corrés Windows)

Si `./rover_launch.sh mapping-ros2 ...` tira:

```
/opt/ros/humble/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable
```

es porque el script tiene modo estricto de bash (`set -u`) y el `source` de
ROS2 pisa una variable no definida. Arreglo permanente:

```bash
echo 'export AMENT_TRACE_SETUP_FILES=0' >> ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### Ventanas gráficas (rviz2, rtabmap-databaseViewer, etc.) no aparecen

WSLg se cuelga a veces. Antes de nada, desde **PowerShell** (no desde Ubuntu):

```powershell
wsl --update
wsl --shutdown
```

Volvé a abrir Ubuntu y probá:

```bash
sudo apt install -y x11-apps
xeyes
```

Si `xeyes` abre, listo. Si vuelve a fallar después de haber andado una vez,
repetí `wsl --shutdown` desde PowerShell (mata todos los procesos de WSL,
así que vas a tener que relevantar cada terminal).

### Dependencias Python del bridge

```bash
pip install requests websocket-client opencv-python
```

(`rclpy` y `cv_bridge` vienen del lado de ROS2, no de pip — necesitás tener
ROS2 sourceado en la terminal donde corrés el bridge).

---

## 1. Grabar una sesión de mapeo (manejando desde el dashboard)

Esta versión de `earth_rover_bridge.py` tiene `/cmd_vel` **deshabilitado a
propósito** (capa de seguridad) — el manejo es exclusivamente desde el
dashboard web del SDK, no por teleop ROS. El bridge solo publica cámara +
odometría (rueda+giro) para que RTAB-Map arme el mapa.

**Terminal 1 — SDK:**

```bash
cd ~/IROS26-LaRovernetta/earth-rovers-sdk
hypercorn main:app
curl -X POST http://localhost:8000/start-mission   # si tu proyecto usa MISSION_SLUG
```

**Terminal 2 — bridge ROS2 (cámara + odometría):**

```bash
cd ~/IROS26-LaRovernetta/earth-rovers-sdk/examples/ros2
python3 earth_rover_bridge.py --ros-args -p sdk_url:=http://localhost:8000
```

Confirmá en el log que dice `Bridging Earth Rovers SDK at http://localhost:8000`
sin `Connection refused` (si aparece, el SDK de la Terminal 1 no está arriba
todavía).

**Terminal 3 — sesión de mapeo RTAB-Map:**

```bash
./rover_launch.sh mapping-ros2 --db ~/maps/sesion1.db
```

Mientras esto corre, manejá el rover **desde el dashboard del SDK** (no
desde una terminal). RTAB-Map va sumando nodos al mapa a medida que te movés
— lo ves en el log como `WM=NN` creciendo.

**Cuando termines de recorrer, Ctrl+C en la Terminal 3.** RTAB-Map guarda la
base de datos automáticamente:

```
rtabmap: Saving database/long-term memory...done! (located at /home/<user>/.ros/rtabmap_mapping.db, NN MB)
```

✅ **Corregido**: antes, si la carpeta de `--db` no existía o el path traía
`~` sin expandir, RTAB-Map caía en silencio a `~/.ros/rtabmap_mapping.db` y la
sesión "se perdía". Ahora `rtabmap_mapping.launch.py` expande el path y crea
la carpeta padre antes de arrancar, e imprime a dónde va a escribir:

```
[rtabmap_mapping.launch.py] database_path -> /home/vos/maps/sesion1.db (se borra al arrancar)
```

Si igual querés confirmarlo

```bash
ls -la ~/.ros/*.db ~/maps/*.db 2>/dev/null
```

y si hace falta, copialo a donde lo querías:

```bash
cp ~/.ros/rtabmap_mapping.db ~/maps/sesion1.db
```

### Señales de que algo se cortó durante la grabación

- `Lookup would require extrapolation into the future` (warnings de TF
  `odom`→`base_link`): normal, ruido de milisegundos entre timestamps. No
  corta la sesión por sí solo.
- `Did not receive data since 5 seconds!` repetido: la cámara dejó de
  publicar — revisá la Terminal 2 (bridge) por errores de `/feed`, o si se
  cortó la conexión al SDK/WiFi del rover.

---

## 2. Ver / inspeccionar la sesión grabada

```bash
rtabmap-databaseViewer ~/maps/sesion1.db
```

(si el binario no existe con ese nombre exacto, buscá con
`dpkg -L ros-humble-rtabmap | grep -i viewer`).

Si al abrir te pregunta **"The database is using 1 different parameter(s)...
Do you want to use database's parameters?"** → contestá **Sí**, así
interpreta la sesión con la config real con la que se grabó.

### Ver frame a frame

- Click en un nodo de la lista de la izquierda para saltar a ese frame.
- Flechas ← → del teclado para avanzar/retroceder de a uno.
- Barra de reproducción abajo para play/pause automático.
- "Graph view" / "Constraints view" te muestra la trayectoria completa y los
  cierres de bucle (loop closures) — click en una conexión para ver el par
  de imágenes que comparó.

### Sobre el mapa 3D

Esta sesión usa cámara **monocular** (`/earth_rover/front/image_raw`, sin
tópico de profundidad ni lidar). Eso significa que **"Export 3D map"**
("From RGB-D images") no va a tener geometría real detrás — no hay
profundidad medida. Lo que sí sirve, y es lo que usa el resto del stack, es
la grilla 2D de ocupación (siguiente sección).

---

## 3. Exportar el mapa 2D (formato `map_server`)

Dentro de `rtabmap-databaseViewer`:

1. **File → Regenerate optimized 2D map...** (el resto de las opciones de
   2D map están grises hasta que hacés esto una vez). Elegí resolución
   `0.05` m/px (o `0.03` para más detalle).
2. Cuando pregunte **"Which type?"** → elegí **"Default occupancy grid"**
   (no "From OctoMap projection" — esa es para datos 3D densos que acá no
   tenemos).
3. Esperá el mensaje "Map generated" — esto solo lo calculó en memoria,
   todavía no guardó nada en disco.
4. Ahora sí: **File → Export optimized 2D map...** → elegí carpeta y
   nombre, por ejemplo `~/maps/sesion1`.

Esto escribe dos archivos:

```
~/maps/sesion1.pgm    # imagen: blanco=libre, negro=ocupado, gris=desconocido
~/maps/sesion1.yaml   # metadata: resolution, origin, thresholds
```

Es el mismo formato estándar de ROS `map_server` que ya sabe leer
`external_map.py` de este proyecto.

Confirmá que quedaron ahí:

```bash
ls ~/maps/
```

---

## 4. Usarlo en modo indoor — caso A: sabés dónde arranca el rover

Si controlás/medís con certeza dónde va a estar parado el robot al arrancar
`indoor_bridge.py`, editá tu config (ej. `configs/indoor_cone_search.yaml`):

```yaml
memory:
  external_map:
    enabled: true
    yaml_path: "maps/sesion1.yaml"
    start_pose_map:
      x_m: 0.0      # posición del robot AHORA, en el marco de ESTE mapa
      y_m: 0.0
      yaw_deg: 0.0
```

Marcá con cinta en el piso el punto de arranque y medí su pose en el mapa
una sola vez con cuidado. Si el robot arranca en otro lugar o mirando para
otro lado, el mapa importado queda rotado/corrido respecto de la realidad —
peor que arrancar con el mapa vacío.

```bash
./rover_launch.sh indoor-bridge --go --search-mode frontier --max-seconds 300
```

---

## 5. Usarlo en modo indoor — caso B: NO controlás dónde arranca (relocalización)

Este es el caso real de una competencia tipo ERC: te dan acceso y solo
podés disparar el inicio, sin saber la posición exacta de arranque. Acá no
alcanza con `external_map` solo — hace falta que **RTAB-Map relocalice
solo**, reconociendo por apariencia (imagen) dónde está dentro del mapa ya
grabado.

**Paso 0 — confirmar el nombre del flag de localización** (una sola vez,
antes de la misión real):

```bash
ros2 launch earth-rovers-sdk/examples/ros2/mapping/rtabmap_mapping.launch.py \
    sdk_url:=http://localhost:8000 \
    database_path:=/home/user/maps/sesion1 \
    localization:=true
```

Buscá un argumento tipo `localization` en la salida.

**Terminal 1 — SDK** (igual que siempre).

**Terminal 2 — bridge** (igual que siempre).

**Terminal 3 — RTAB-Map en modo LOCALIZACIÓN**, cargando el `.db` ya
grabado, sin agregar nodos nuevos:

```bash
ros2 launch <ruta>/examples/ros2/mapping/rtabmap_mapping.launch.py \
    db_path:=~/maps/rtabmap_mapping localization:=true
```

**Config de `genie_rover`** — activá la corrección de pose vía RTAB-Map:

```yaml
mapping:
  rtabmap_correction:
    enabled: true
    map_frame: "map"
    base_frame: "base_link"
    lookup_timeout_s: 0.2
```

y **dejá `memory.external_map` deshabilitado** (no hace falta `start_pose_map`
manual acá) — la pose la toma directo de la corrección TF `map -> base_link`
que publica RTAB-Map en cuanto reconoce dónde está.

**Terminal 4 — la misión real:**

```bash
./rover_launch.sh indoor-bridge --go --search-mode frontier --max-seconds 300
```

Vas a ver en consola `"correccion de RTAB-Map ACTIVADA"`. Al principio la
posición puede estar mal (arranca en un placeholder 0,0,0) hasta que RTAB-Map
reconoce algo que coincide con el mapa grabado — ahí se autocorrige sola,
frame a frame.

**Limitación real, no resoluble por software**: si el rover arranca en una
zona que la sesión de mapeo original **nunca vio**, no hay forma de que se
ubique hasta que llegue a territorio conocido. Esto funciona bien si el
punto de arranque varía dentro del área ya mapeada; no funciona si puede
arrancar en una zona completamente nueva cada vez.

**Respaldo recomendado**: dejá `mapping.export_every_s` bajo (15-20s) en la
sesión de grabación original, para tener exports parciales si la sesión se
corta antes de terminar.

---

## Resumen — comandos de una sola línea

```bash
# ---- grabar sesión (Terminal 1/2/3, en paralelo) --------------------------
cd ~/IROS26-LaRovernetta/earth-rovers-sdk && hypercorn main:app
python3 earth_rover_bridge.py --ros-args -p sdk_url:=http://localhost:8000
./rover_launch.sh mapping-ros2 --db ~/maps/sesion1.db

# ---- ver e inspeccionar ----------------------------------------------------
rtabmap-databaseViewer ~/maps/sesion1.db

# ---- usar en modo indoor, arranque CONOCIDO --------------------------------
./rover_launch.sh indoor-bridge --go --search-mode frontier --max-seconds 300

# ---- usar en modo indoor, arranque DESCONOCIDO (relocalización) -----------
ros2 launch <ruta>/examples/ros2/mapping/rtabmap_mapping.launch.py \
    db_path:=~/maps/sesion1.db localization:=true
./rover_launch.sh indoor-bridge --go --search-mode frontier --max-seconds 300
```
