# Guía de Ejecución: La Rovernetta (ROS 2 EKF + GeNIE)

El sistema de La Rovernetta está compuesto por dos capas modulares que deben ejecutarse en paralelo:
1. **Capa Sensorial (ROS 2 EKF)**: Se encarga de conectarse a la API del rover, leer los sensores crudos (IMU, GPS, Odometría), fusionarlos matemáticamente y calcular un rumbo (heading) preciso sin deriva magnética.
2. **Cerebro Autónomo (GeNIE / Python)**: Analiza el video con Inteligencia Artificial (SAM-TP), proyecta los obstáculos en un mapa 2D usando el rumbo calculado por el EKF, planifica la ruta, y envía los comandos de los motores.

A continuación, se detalla cómo instalar y ejecutar ambas capas desde cero en una computadora anfitriona (sin usar Docker).

---

## 1. Instalación de ROS 2 (Jazzy Jalisco)

> **Nota:** ROS 2 Jazzy requiere **Ubuntu 24.04**. Si estás en Ubuntu 22.04, deberás instalar ROS 2 Humble y reemplazar la palabra `jazzy` por `humble` en los comandos a continuación.

Abre una terminal y ejecuta los comandos oficiales para instalar ROS 2 Base y las herramientas de compilación:

```bash
# 1. Asegúrate de tener un sistema actualizado
sudo apt update && sudo apt install software-properties-common curl -y

# 2. Agrega la clave y el repositorio de ROS 2
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. Instala ROS 2 y las dependencias de La Rovernetta
sudo apt update
sudo apt install ros-jazzy-ros-base python3-colcon-common-extensions ros-jazzy-robot-localization ros-jazzy-tf2-ros -y
```

---

## 2. Compilar el Workspace de ROS 2

Una vez instalado ROS 2, debes compilar el workspace que se encuentra en la carpeta `ros2_ws_src/` de este repositorio.

1. Abre una terminal en la raíz de este repositorio (`IROS26-LaRovernetta`).
2. Ingresa a la carpeta del workspace:
   ```bash
   cd ros2_ws_src
   ```
3. Configura el entorno de ROS y compila los paquetes:
   ```bash
   source /opt/ros/jazzy/setup.bash
   colcon build
   ```

*(Solo debes compilar la primera vez o cuando modifiques el código fuente dentro de `ros2_ws_src`).*

---

## 3. Flujo de Ejecución (Paso a Paso)

Cada vez que vayas a correr una misión, necesitas abrir **dos terminales**. Reemplaza `IP_DEL_ROVER` por la IP real del Earth Rover en tu red (por ejemplo, `172.31.187.39`).

### TERMINAL 1: Iniciar el EKF de ROS 2
Esta terminal mantendrá vivo el filtro de Kalman.

```bash
# Entrar al workspace
cd IROS26-LaRovernetta/ros2_ws_src

# Configurar el entorno (siempre necesario en una terminal nueva)
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Lanzar el nodo puente y el EKF
ros2 launch er_bringup localization_bypass.launch.py sdk_url:=http://IP_DEL_ROVER:8000 target_ip:=IP_DEL_ROVER
```
*Deja esta terminal corriendo.*

### TERMINAL 2: Iniciar el Cerebro GeNIE
Esta terminal ejecutará la conducción autónoma.

```bash
# Entrar a la carpeta de la IA
cd IROS26-LaRovernetta/genie

# Activar el entorno virtual de conda o Python (según cómo hayas instalado SAM2/Torch)
# source .venv_genie/bin/activate
# o 
# conda activate sam_tp

# Lanzar el rover de forma autónoma
./run_mission.sh --go
```

**Parámetros útiles del script `run_mission.sh`:**
* `./run_mission.sh --go`: Inicia el movimiento real de los motores.
* `./run_mission.sh` (sin `--go`): Ejecuta en modo "Dry Run" o Simulacro. Procesará la visión y mostrará qué decidiría hacer, pero sin mandarle torque a las ruedas. Útil para debugear de forma segura en un escritorio.

### Detención de Emergencia
Para frenar el rover de emergencia, basta con hacer click en la **Terminal 2** (GeNIE) y presionar `Ctrl + C`. El sistema atrapará la señal y enviará un comando de frenado a cero (0) a los motores antes de cerrarse.
