# Guía de Ejecución: La Rovernetta (ROS 2 EKF + GeNIE)

El sistema de La Rovernetta está compuesto por tres capas modulares que se comunican en red local:
1. **Servidor SDK (Earth Rovers SDK)**: Conexión WebRTC/Agora con el rover, expone la API HTTP en el puerto `8000` y telemetría de sensores.
2. **Capa Sensorial (ROS 2 EKF)**: Lee los sensores crudos del SDK (IMU, Odometría, GPS), ejecuta el Filtro de Kalman Extendido dual (`robot_localization`) y exporta el rumbo (*heading*) estimado sin deriva magnética vía UDP al puerto `9876`.
3. **Cerebro Autónomo (GeNIE / Python)**: Analiza el video con IA (SAM-TP), proyecta los obstáculos en un mapa 2D, recibe el rumbo por UDP y despacha los comandos de velocidad angular y lineal.

---

## 1. Instalación de Dependencias de ROS 2

Todas las dependencias de ROS 2 están autocontenidas dentro de la carpeta `ros2_ws_src/`.

### Opción A: Instalación Automática en el Host (Ubuntu 22.04 / 24.04 / 26.04)
Ejecuta el script instalador para configurar los paquetes del sistema y de Python automáticamente:
```bash
cd ros2_ws_src
./install_dependencies.sh
```
*(Este script detecta tu distribución de ROS 2 —Jazzy, Humble o Lyrical— e instala los paquetes necesarios como `robot-localization`, `tf2-ros`, `geographic-msgs` y `pygeomag`).*

### Opción B: Modo Docker (Aislado, sin instalar ROS 2 en el host)
Si prefieres no instalar paquetes en tu máquina host o estás en otro sistema operativo, puedes utilizar el contenedor Docker dedicado incluido en `ros2_ws_src/` (ver Sección 3).

---

## 2. Compilar el Workspace de ROS 2

Para compilar los paquetes de ROS 2 (`earth_rovers_sdk`, `mini_plus_localization`, `er_bringup`):
```bash
cd ros2_ws_src
./build.sh
```
*(O directamente desde la raíz si ya tienes el entorno configurado).*

---

## 3. Flujo de Ejecución (Paso a Paso)

Abre tres terminales para correr el sistema completo:

### TERMINAL 1: Servidor SDK
Inicia el servidor local del SDK (puerto 8000):
```bash
./start_sdk_local.sh
```
*(Dashboard disponible en http://localhost:8000).*

### TERMINAL 2: Filtro de Kalman (ROS 2 EKF)

#### En máquina anfitriona (Nativo):
Desde la raíz del repositorio:
```bash
./run_ekf.sh
```
*(Por defecto se conecta a `http://localhost:8000` y exporta el heading a `127.0.0.1:9876`).*

*Si el SDK o el rover estuvieran en otra IP remota:*
```bash
./run_ekf.sh http://172.31.187.39:8000 172.31.187.39
```

#### O vía Docker (Aislado):
```bash
./run_ekf_docker.sh
```

### TERMINAL 3: Iniciar el Cerebro Autónomo (GeNIE)
```bash
# Modo Real (con movimiento motriz y activación de checkpoints):
./run_mission.sh --go --start-mission

# Modo Seguro / Dry-Run (simulacro sin torque en los motores):
./run_mission.sh --start-mission
```

---

## Detención de Emergencia
Para frenar el rover de emergencia:
1. Presiona `Ctrl + C` en la **Terminal 3** (GeNIE). El sistema atrapará la señal y enviará un comando de frenado a cero (0) a los motores antes de cerrarse.
2. Como respaldo secundario, haz click en el botón **STOP** en el dashboard web en [http://localhost:8000](http://localhost:8000).
