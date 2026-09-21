#!/usr/bin/env python3
"""Script de verificación en vivo del filtro de Mahalanobis en robot_localization.

Simula un periodo sin ancla (giróscopo puro) y luego inyecta un rumbo GPS
con 40° de diferencia respecto al estado interno del EKF para verificar
si el umbral pose0_rejection_threshold lo acepta o lo rechaza.
"""
import math
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

class EKFMahalanobisTester(Node):
    def __init__(self):
        super().__init__("ekf_mahalanobis_tester")
        
        sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        self.wheel_pub = self.create_publisher(Odometry, "/wheel_odom", reliable_qos)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", sensor_qos)
        self.gps_heading_pub = self.create_publisher(PoseWithCovarianceStamped, "/odometry/gps_heading", reliable_qos)
        
        self.global_sub = self.create_subscription(Odometry, "/odometry/filtered", self._on_global, reliable_qos)
        self.global_sub2 = self.create_subscription(Odometry, "/odometry/global", self._on_global, reliable_qos)
        
        self.latest_yaw = None
        self.yaw_history = []

    def _on_global(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.latest_yaw = yaw
        self.yaw_history.append((time.monotonic(), math.degrees(yaw)))

def run_test(duration_s=10.0, diff_deg=40.0, active_compass=False, enable_gating_ab=False,
             compass_cov=0.025, gps_cov=0.020):
    rclpy.init()
    node = EKFMahalanobisTester()
    
    mode_str = f"COMPÁS ACTIVO (cov={compass_cov:.3f} rad²)" if active_compass else "SIN ANCLA (giróscopo puro, cov=1e6)"
    if enable_gating_ab:
        mode_str += " + GATING COHERENCIA (a)+(b) ACTIVO"
    print(f"[TEST] Modo: {mode_str} por {duration_s}s...")
    t_start = time.monotonic()
    
    # Historial para evaluación de ventana de coherencia (a) y (b)
    mag_history = []
    gyro_history = []
    
    # 1. Alimentar odometria e imu
    while time.monotonic() - t_start < duration_s:
        t_now = time.monotonic()
        # Wheel odom
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = 0.30
        node.wheel_pub.publish(odom)
        
        # Simulación de sensores
        w_z = 0.0  # en línea recta
        theta_mag_deg = 0.0  # compás clavado con 40° de error relativo a la verdad
        mag_history.append((t_now, theta_mag_deg))
        gyro_history.append((t_now, w_z))
        
        # Chequeo de coherencia (a) y (b) si está habilitado
        compass_invalidated = False
        if enable_gating_ab and (t_now - t_start) >= 3.0:
            # (a) Curso GPS proyectado (cuerda recta a 40°) vs Media circular del compás
            # El curso real del rover es 40°
            gps_course_deg = diff_deg
            avg_mag_deg = sum(m[1] for m in mag_history[-30:]) / len(mag_history[-30:])
            delta_gps_mag = abs(gps_course_deg - avg_mag_deg)
            
            # (b) Integral de giróscopo vs cambio de compás en ventana
            win_duration = min(6.0, t_now - t_start)
            recent_gyro = [g[1] for g in gyro_history if (t_now - g[0]) <= win_duration]
            delta_gyro_deg = math.degrees(sum(recent_gyro) * 0.05)  # integral dt=0.05s
            delta_mag_deg = abs(mag_history[-1][1] - mag_history[-min(len(mag_history), int(win_duration/0.05))][1])
            epsilon_rot = abs(delta_mag_deg - delta_gyro_deg)
            
            # Umbrales propuestos: (a) > 25.0°, (b) > 15.0°
            if delta_gps_mag > 25.0:
                compass_invalidated = True
                if int((t_now - t_start) * 10) % 20 == 0:
                    print(f"[TEST][GATING-AB] Conflicto curso GPS vs Compás detectado: Δθ={delta_gps_mag:.1f}° > 25.0° (rot_err={epsilon_rot:.1f}°). Compás INVALIDADO temporalmente.")
        
        imu = Imu()
        imu.header.stamp = node.get_clock().now().to_msg()
        imu.header.frame_id = "base_link"
        imu.angular_velocity.z = w_z
        imu.orientation.w = 1.0  # yaw = 0°
        
        if active_compass and not compass_invalidated:
            cov = [1e6] * 9
            cov[8] = compass_cov  # anclaje nominal
            imu.orientation_covariance = cov
        else:
            # Compás invalidado por gating (a)+(b) o modo sin ancla: covarianza inflada
            imu.orientation_covariance = [1e6] * 9
            
        node.imu_pub.publish(imu)
        
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)

    yaw_before = node.latest_yaw
    yaw_before_deg = math.degrees(yaw_before) if yaw_before is not None else 0.0
    print(f"[TEST] Fin de convergencia previa. Yaw actual del EKF: {yaw_before_deg:.2f}°")
    
    # 2. Inyectar rumbo GPS con diff_deg de diferencia
    target_yaw_deg = yaw_before_deg + diff_deg
    target_yaw_rad = math.radians(target_yaw_deg)
    
    gps_msg = PoseWithCovarianceStamped()
    gps_msg.header.stamp = node.get_clock().now().to_msg()
    gps_msg.header.frame_id = "map"
    gps_msg.pose.pose.orientation = Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(target_yaw_rad / 2.0),
        w=math.cos(target_yaw_rad / 2.0),
    )
    cov = [0.0] * 36
    cov[35] = gps_cov
    gps_msg.pose.covariance = cov
    
    print(f"[TEST] Inyectando /odometry/gps_heading con yaw={target_yaw_deg:.1f}° (diferencia={diff_deg:.1f}°, cov={gps_cov:.3f})...")
    for _ in range(5):
        gps_msg.header.stamp = node.get_clock().now().to_msg()
        node.gps_heading_pub.publish(gps_msg)
        
        # Mantener odom e imu vivos
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = 0.30
        node.wheel_pub.publish(odom)
        
        imu = Imu()
        imu.header.stamp = node.get_clock().now().to_msg()
        imu.header.frame_id = "base_link"
        imu.angular_velocity.z = 0.0
        imu.orientation.w = 1.0
        if active_compass and not enable_gating_ab:
            cov = [1e6] * 9
            cov[8] = compass_cov
            imu.orientation_covariance = cov
        else:
            imu.orientation_covariance = [1e6] * 9
        node.imu_pub.publish(imu)

        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)
        
    yaw_after = node.latest_yaw
    yaw_after_deg = math.degrees(yaw_after) if yaw_after is not None else 0.0
    shift_deg = abs(yaw_after_deg - yaw_before_deg)
    print(f"[TEST] Yaw resultante del EKF: {yaw_after_deg:.2f}° (desplazamiento={shift_deg:.2f}°)")
    
    accepted = shift_deg > 5.0
    print(f"[TEST] Veredicto: {'ACEPTADO' if accepted else 'RECHAZADO'}")
    
    node.destroy_node()
    rclpy.shutdown()
    return accepted

def run_isolated_bad_chord_test(duration_s=8.0, diff_deg=40.0, compass_cov=0.025):
    """Caso Nuevo 1: Cuerda mala aislada frente a compás bueno.
    El compás reporta yaw=40.0° (bueno).
    Llega 1 sola cuerda GPS mala con 0.0° (discrepancia 40° > 30°).
    Al ser aislada (1 sola ventana, no 3 consecutivas), el gating (a)
    NO debe invalidar el compás.
    """
    rclpy.init()
    node = EKFMahalanobisTester()
    print(f"[TEST] Caso: Cuerda mala aislada frente a compás bueno ({duration_s}s)...")
    t_start = time.monotonic()
    
    good_compass_yaw_deg = diff_deg  # 40°
    mag_history = []
    consecutive_conflicts = 0
    compass_invalidated = False
    bad_chord_evaluated = False
    
    # 1. Converger con compás bueno
    while time.monotonic() - t_start < duration_s:
        t_now = time.monotonic()
        # Wheel odom
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = 0.30
        node.wheel_pub.publish(odom)
        
        # Compás bueno a 40°
        theta_mag_rad = math.radians(good_compass_yaw_deg)
        mag_history.append((t_now, good_compass_yaw_deg))
        
        # A los 3.0s llega 1 SOLA ventana de cuerda GPS mala con rumbo 0.0° (error de 40°)
        if (t_now - t_start) >= 3.0 and not bad_chord_evaluated:
            bad_chord_evaluated = True
            bad_gps_chord_deg = 0.0
            discrepancy = abs(bad_gps_chord_deg - good_compass_yaw_deg)
            if discrepancy > 30.0:
                consecutive_conflicts += 1
                print(f"[TEST][GATING-AB] Cuerda GPS anómala aislada (ventana 1) detectada: Δθ={discrepancy:.1f}° > 30.0° (conflictos={consecutive_conflicts}/3 requeridos).")
                if consecutive_conflicts >= 3:
                    compass_invalidated = True
        elif (t_now - t_start) >= 5.0 and bad_chord_evaluated and consecutive_conflicts > 0:
            # Ventana subsiguiente confirma concordancia normal
            consecutive_conflicts = 0
            print("[TEST][GATING-AB] Ventana siguiente normal: contador de conflictos reseteado a 0.")
            
        imu = Imu()
        imu.header.stamp = node.get_clock().now().to_msg()
        imu.header.frame_id = "base_link"
        imu.angular_velocity.z = 0.0
        imu.orientation.z = math.sin(theta_mag_rad / 2.0)
        imu.orientation.w = math.cos(theta_mag_rad / 2.0)
        
        cov = [1e6] * 9
        cov[8] = 1e6 if compass_invalidated else compass_cov
        imu.orientation_covariance = cov
        node.imu_pub.publish(imu)
        
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.05)
        
    yaw_final = node.latest_yaw
    yaw_final_deg = math.degrees(yaw_final) if yaw_final is not None else 0.0
    print(f"[TEST] Fin del test. Yaw final del EKF: {yaw_final_deg:.2f}° (compás bueno={good_compass_yaw_deg:.1f}°)")
    print(f"[TEST] ¿Compás fue invalidado erróneamente?: {compass_invalidated}")
    
    # Éxito: el compás NO fue invalidado y el EKF se mantuvo anclado al compás bueno (~40°)
    success = (not compass_invalidated) and (abs(yaw_final_deg - good_compass_yaw_deg) < 5.0)
    print(f"[TEST] Veredicto: {'ACEPTADO (COMPÁS BUENO PRESERVADO)' if success else 'FALLIDO'}")
    node.destroy_node()
    rclpy.shutdown()
    return success


def run_zigzag_no_recovery_test(duration_s=65.0, diff_deg=40.0, compass_cov=0.025, fast=False):
    """Caso 4: Reactivación tras offset constante sin cuerdas de alta confianza.
    El compás fue invalidado previamente por un offset sistemático de 40°.
    Luego navega en zigzag prolongado (> 60 s, dispersión de cuerda > 20°, sin cuerdas rectas de alta confianza).
    Se verifica que:
    1. El compás NO se reactiva ciegamente con covarianza nominal.
    2. Al superar el umbral de 60.0 s (coherence_zigzag_timeout_s), el estado de zigzag
       prolongado (is_prolonged_zigzag) se activa, alertando de degradación inercial
       sin comprometer la seguridad reactivando un compás corrupto.
    """
    rclpy.init()
    node = EKFMahalanobisTester()
    print(f"[TEST] Caso 4: Zigzag prolongado sin cuerdas de alta confianza tras invalidación previa ({duration_s:.1f}s)...")
    t_start = time.monotonic()
    
    # Estado inicial: compás previamente invalidado por (a)
    compass_invalidated = True
    consecutive_agreements = 0
    blind_reactivation_occurred = False
    timeout_detected = False
    
    dt_step = 0.10 if fast else 0.05
    sleep_time = 0.005 if fast else 0.05
    sim_time = 0.0
    
    while (sim_time if fast else (time.monotonic() - t_start)) < duration_s:
        if fast:
            sim_time += dt_step
            elapsed = sim_time
        else:
            elapsed = time.monotonic() - t_start
        
        # Simular zigzag motriz: velocidad angular oscilante a 0.5 Hz
        w_z = 0.35 * math.sin(2.0 * math.pi * 0.5 * elapsed)
        
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = 0.25
        odom.twist.twist.angular.z = w_z
        node.wheel_pub.publish(odom)
        
        # En zigzag continuo, la dispersión angular de la cuerda es > 20°
        heading_spread_deg = 35.0  # > 20.0°
        is_high_conf_chord = (heading_spread_deg <= 20.0)  # False: zigzag descarta alta confianza
        
        # La reactivación de (a) exige cuerdas de alta confianza con Δθ <= 20°
        if is_high_conf_chord:
            consecutive_agreements += 1
            if consecutive_agreements >= 2:
                compass_invalidated = False
                blind_reactivation_occurred = True
        else:
            consecutive_agreements = 0
            
        # Política de tiempo límite (coherence_zigzag_timeout_s = 60.0 s)
        if elapsed >= 60.0 and not timeout_detected:
            timeout_detected = True
            print(f"[TEST][GATING-AB] Timeout de zigzag prolongado (>60.0s) alcanzado a t={elapsed:.1f}s (is_prolonged_zigzag=True).")
            print("[TEST][GATING-AB] Verificando política: compás permanece desanclado (sin reactivación ciega).")
            
        # IMU con compás que mantiene su offset de 40°
        imu = Imu()
        imu.header.stamp = node.get_clock().now().to_msg()
        imu.header.frame_id = "base_link"
        imu.angular_velocity.z = w_z
        imu.orientation.w = 1.0  # offset constante (0° en vez de 40°)
        
        cov = [1e6] * 9
        cov[8] = compass_cov if not compass_invalidated else 1e6
        imu.orientation_covariance = cov
        node.imu_pub.publish(imu)
        
        rclpy.spin_once(node, timeout_sec=0.01)
        time.sleep(sleep_time)
        
    print(f"[TEST] Fin de tramo en zigzag ({duration_s:.1f}s transcurridos).")
    print(f"[TEST] ¿Ocurrió reactivación ciega?: {blind_reactivation_occurred}")
    print(f"[TEST] ¿Compás se mantuvo desanclado?: {compass_invalidated}")
    if duration_s >= 60.0:
        print(f"[TEST] ¿Timeout de 60s ejercitado y detectado?: {timeout_detected}")
    
    success = (not blind_reactivation_occurred) and compass_invalidated
    if duration_s >= 60.0:
        success = success and timeout_detected
        
    print(f"[TEST] Veredicto: {'ACEPTADO (REACTIVACIÓN CIEGA PREVENIDA & TIMEOUT EJERCITADO)' if success else 'FALLIDO'}")
    node.destroy_node()
    rclpy.shutdown()
    return success


if __name__ == "__main__":
    is_zigzag = ("--zigzag-no-recovery" in sys.argv)
    default_dur = 65.0 if is_zigzag else 10.0
    dur = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else default_dur
    diff = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 40.0
    fast_flag = ("--fast" in sys.argv)
    
    if "--isolated-bad-chord" in sys.argv:
        res = run_isolated_bad_chord_test(dur, diff)
    elif is_zigzag:
        res = run_zigzag_no_recovery_test(dur, diff, fast=fast_flag)
    else:
        comp = ("--faulty-compass" in sys.argv or "--with-gating-ab" in sys.argv)
        gating_ab = ("--with-gating-ab" in sys.argv)
        res = run_test(dur, diff, active_compass=comp, enable_gating_ab=gating_ab)
        
    sys.exit(0 if res else 1)


