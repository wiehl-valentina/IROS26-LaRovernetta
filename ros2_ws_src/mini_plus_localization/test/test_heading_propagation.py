#!/usr/bin/env python3
"""
test_heading_propagation.py — Test offline de propagación continua de rumbo con giróscopo (Brief 22 / V.3.4)

Verifica que ekf_heading_bridge:
1. Propaga el rumbo suavemente integrando gyro_z entre actualizaciones discretas del compás (~2s).
2. Evita saltos bruscos: la trayectoria angular es continua.
3. La incertidumbre crece durante la integración y se resetea al recibir un nuevo compás.
4. Al llegar el nuevo compás, la discontinuidad de realineación es mínima.
5. Si no hay flujo de IMU, mantiene compatibilidad publicando el compás directo.
"""

import importlib.util
import math
import os
import pytest
import rclpy
from builtin_interfaces.msg import Time as MsgTime
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

# Carga dinámica del script ejecutable ekf_heading_bridge.py
SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "ekf_heading_bridge.py"
)
spec = importlib.util.spec_from_file_location("ekf_heading_bridge", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EkfHeadingBridge = mod.EkfHeadingBridge


@pytest.fixture
def ros_context():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def make_odom_msg(heading_deg: float, stamp_sec: float) -> Odometry:
    """Crea un mensaje Odometry con el yaw ENU equivalente al rumbo de compás."""
    # heading_deg (0=North, clockwise) -> yaw_enu = 90 - heading_deg
    yaw_enu_rad = math.radians((90.0 - heading_deg) % 360.0)
    
    msg = Odometry()
    sec = int(stamp_sec)
    nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp = MsgTime(sec=sec, nanosec=nanosec)
    msg.header.frame_id = "map"
    
    # Cuaternión 2D (yaw alrededor de Z)
    msg.pose.pose.orientation.z = math.sin(yaw_enu_rad / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw_enu_rad / 2.0)
    return msg


def make_imu_msg(omega_z_rad_s: float, stamp_sec: float) -> Imu:
    """Crea un mensaje Imu con velocidad angular en Z."""
    msg = Imu()
    sec = int(stamp_sec)
    nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp = MsgTime(sec=sec, nanosec=nanosec)
    msg.header.frame_id = "base_link"
    msg.angular_velocity.z = float(omega_z_rad_s)
    return msg


def test_continuous_gyro_propagation_smoothness_and_realign(ros_context):
    """V.3.4: Compás cada 2s con saltos discretos, giróscopo constante entre medio.
    
    Verifica trayectoria suave e incertidumbre creciente con realineación de bajo salto.
    """
    node = EkfHeadingBridge()

    published_headings = []
    published_uncertainties = []

    node.heading_pub.publish = lambda m: published_headings.append(m.data)
    node.uncertainty_pub.publish = lambda m: published_uncertainties.append(m.data)

    base_t = 100.0
    odom0 = make_odom_msg(heading_deg=0.0, stamp_sec=base_t)
    node._on_odom(odom0)

    assert math.isclose(node.current_heading_deg, 0.0, abs_tol=1e-5)
    assert node.heading_uncertainty_deg == 3.0

    # 2. El rover empieza a girar a la derecha a 15.0 deg/s (omega_z = -15 deg/s)
    # Entre t=0.0s y t=2.0s, el giróscopo publica a 10 Hz (20 muestras de 0.1s)
    omega_deg_s = -15.0  # Giro horario (CW), heading debe crecer
    omega_rad_s = math.radians(omega_deg_s)

    for i in range(0, 21):
        t_sec = base_t + i * 0.1
        imu_msg = make_imu_msg(omega_z_rad_s=omega_rad_s, stamp_sec=t_sec)
        node._on_imu(imu_msg)

    # Tras 20 pasos (2.0s de propagación continua), el rumbo estimado debe ser ~30.0°
    assert len(published_headings) >= 19
    estimated_heading_before_compass = node.current_heading_deg
    assert math.isclose(estimated_heading_before_compass, 30.0, abs_tol=0.2)

    # La incertidumbre debió crecer durante los 2 segundos de integración continua
    # sigma = sqrt(3.0^2 + 0.5^2 * 2.0) = sqrt(9 + 0.5) = 3.082°
    expected_sigma = math.sqrt(3.0**2 + (0.5**2) * 2.0)
    assert math.isclose(node.heading_uncertainty_deg, expected_sigma, abs_tol=0.05)
    assert node.heading_uncertainty_deg > 3.0

    # 3. t = 2.0s: Llega nuevo compás con lectura real física 30.4° (con leve ruido de 0.4°)
    odom1 = make_odom_msg(heading_deg=30.4, stamp_sec=base_t + 2.0)
    heading_before = node.current_heading_deg
    node._on_odom(odom1)
    heading_after = node.current_heading_deg

    # Verificación V.3.4: El salto de realineación debe ser PEQUEÑO (< 1.0°)
    realign_jump = abs((heading_after - heading_before + 540.0) % 360.0 - 180.0)
    assert realign_jump < 1.0, f"Salto de realineación demasiado grande: {realign_jump:.2f}°"

    # La incertidumbre debe resetearse a la incertidumbre base del compás (3.0°)
    assert node.heading_uncertainty_deg == 3.0

    node.destroy_node()


def test_gyro_bias_compensation(ros_context):
    """Verifica que el bias del giróscopo se sustrae correctamente."""
    node = EkfHeadingBridge()
    node.gyro_bias_z = 0.05  # Bias constante de 0.05 rad/s

    base_t = 100.0
    # Inicializar compás a 90°
    odom = make_odom_msg(heading_deg=90.0, stamp_sec=base_t)
    node._on_odom(odom)

    # Enviar lectura de giróscopo idéntica al bias (rover estático con sensor descalibrado)
    imu0 = make_imu_msg(omega_z_rad_s=0.05, stamp_sec=base_t)
    node._on_imu(imu0)

    imu1 = make_imu_msg(omega_z_rad_s=0.05, stamp_sec=base_t + 1.0)
    node._on_imu(imu1)

    # Al sustraer el bias, omega_efectivo = 0.05 - 0.05 = 0.0 -> heading debe permanecer en 90°
    assert math.isclose(node.current_heading_deg, 90.0, abs_tol=1e-3)

    node.destroy_node()


def test_backwards_compatibility_without_imu(ros_context):
    """Verifica que si no hay mensajes IMU, el nodo sigue publicando el compás directo como antes."""
    node = EkfHeadingBridge()

    published = []
    node.heading_pub.publish = lambda m: published.append(m.data)

    base_t = 100.0
    odom0 = make_odom_msg(heading_deg=45.0, stamp_sec=base_t)
    node._on_odom(odom0)

    odom1 = make_odom_msg(heading_deg=48.0, stamp_sec=base_t + 2.0)
    node._on_odom(odom1)

    assert len(published) == 2
    assert math.isclose(published[0], 45.0, abs_tol=0.1)
    assert math.isclose(published[1], 48.0, abs_tol=0.1)

    node.destroy_node()
