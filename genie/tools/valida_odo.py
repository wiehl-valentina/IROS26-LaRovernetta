from genie_rover.sdk_client import RoverClient
from genie_rover.odometry import Odometry, OdometryConfig
import yaml, math, time

cfg_yaml = yaml.safe_load(open("configs/frodobot_rover.yaml"))
o = cfg_yaml["odometry"]
cfg = OdometryConfig(
    wheel_radius_m=o["wheel_radius_m"],
    track_width_m=o["track_width_m"],
    left_rpm_indices=tuple(o["left_rpm_indices"]),
    right_rpm_indices=tuple(o["right_rpm_indices"]),
    rotation_sign=o.get("rotation_sign", 1.0),
    gyro_yaw_index=o.get("gyro_yaw_index", 2),
    gyro_sign=o.get("gyro_sign", 1.0),
    gps_correction=False,
)

c = RoverClient(timeout=30)
odo = Odometry(cfg)

try:
    print("PRUEBA 1: avanzar derecho unos 2 metros.")
    input("Marca el punto de partida y apreta ENTER...")

    t0 = time.time()
    while time.time() - t0 < 6.0:
        c.control(0.30, 0.0)
        try:
            odo.update(c.telemetry().raw)
        except Exception:
            pass
        time.sleep(0.1)
    c.stop(); c.stop()

    print(f"\n  odometria: x={odo.pose.x:.3f} m  y={odo.pose.y:.3f} m")
    print(f"  distancia: {odo.distance_travelled:.3f} m")
    print(f"  rotacion:  {math.degrees(odo.pose.theta):+.1f} grados")
    print("\n  >>> MEDI CON CINTA cuanto avanzo y compara con x <<<")

    input("\nENTER para la PRUEBA 2 (giro a la izquierda)...")
    odo.reset()
    t0 = time.time()
    while time.time() - t0 < 2.5:
        c.control(0.0, 0.45)
        try:
            odo.update(c.telemetry().raw)
        except Exception:
            pass
        time.sleep(0.1)
    c.stop(); c.stop()

    print(f"\n  rotacion medida: {math.degrees(odo.pose.theta):+.1f} grados")
    print(f"  deriva: x={odo.pose.x:.3f} y={odo.pose.y:.3f} (deberian ser ~0)")
    print("\n  >>> Giro un cuarto de vuelta? <<<")
finally:
    c.stop(); c.stop()
    print("\n[frenado]")
