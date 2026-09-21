# Workspace ROS 2: La Rovernetta (Capa Sensorial & EKF)

Este directorio (`ros2_ws_src`) contiene el subsistema de ROS 2 para **La Rovernetta**. Se encarga de conectarse con el Earth Rover mediante el SDK, ingerir la telemetría cruda (odometría de ruedas, IMU MPU6050, GNSS), procesarla con un Filtro de Kalman Extendido dual (`robot_localization`), y exportar el rumbo (*heading*) estimado sin deriva magnética hacia el cerebro autónomo (GeNIE) a través de un socket UDP en el puerto `9876`.

---

## Estructura de Paquetes

* **`earth_rovers_sdk`**: Nodo puente (`bridge_node.py`) que se suscribe a los datos del rover vía HTTP/WebSocket, calcula la declinación magnética con `pygeomag`, filtra saltos espaciales del GNSS y publica tópicos estándar de ROS 2 (`/imu/data`, `/wheel_odom`, `/gps/fix`, `/earth_rover/compass_heading`).
* **`mini_plus_localization`**: Implementación del stack de localización:
  * `launch/ekf.launch.py`: Ejecuta EKF local (`odometry/local`), EKF global (`odometry/global`), `navsat_transform` y bridges de orientación.
  * `scripts/udp_heading_exporter.py`: Toma el rumbo fusionado y lo emite vía UDP en formato binario de baja latencia al puerto `9876`.
* **`er_bringup`**: Launch files maestros del sistema:
  * `localization_bypass.launch.py`: Inicia el bridge, el stack de EKF y el exportador UDP en un solo comando unificado.

---

## Requisitos y Dependencias

Las dependencias del sistema y de Python están detalladas en:
* [`apt-packages.txt`](apt-packages.txt): Paquetes APT requeridos (`robot-localization`, `tf2-ros`, `geographic-msgs`, etc.).
* [`requirements-ros.txt`](requirements-ros.txt): Dependencias Python (`requests`, `websocket-client`, `pygeomag`, etc.).

---

## 1. Instalación Rápida (Máquina Anfitriona / Nativo)

Para instalar automáticamente todas las dependencias necesarias en Ubuntu:
```bash
./install_dependencies.sh
```

---

## 2. Compilación del Workspace

Para compilar todos los paquetes con `colcon`:
```bash
./build.sh
```
*(O manualmente: `source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install`)*

---

## 3. Ejecución del Filtro de Kalman

### Opción A: Modo Nativo
```bash
./run_ekf.sh
```
O desde la raíz del repositorio:
```bash
./run_ekf.sh
```

**Parámetros configurables:**
* Por defecto se conecta a `http://localhost:8000` y exporta a `127.0.0.1:9876`.
* Si el SDK o el rover están en otra IP:
  ```bash
  ./run_ekf.sh http://172.31.187.39:8000 172.31.187.39
  ```
  O usando variables de entorno:
  ```bash
  SDK_URL=http://172.31.187.39:8000 TARGET_IP=172.31.187.39 ./run_ekf.sh
  ```

### Opción B: Modo Docker (Aislado, sin instalar ROS 2 en el host)
Si tu máquina no cuenta con ROS 2 instalado o prefieres no tocar los paquetes del host:
```bash
./run_ekf_docker.sh
```
O desde la raíz del repositorio:
```bash
./run_ekf_docker.sh
```
Este contenedor utiliza `network_mode: host` para comunicarse sin problemas con el servidor local del SDK (`:8000`) y con GeNIE (`UDP :9876`).
