from genie_rover.sdk_client import RoverClient
from genie_rover.odometry import Odometry, OdometryConfig
import yaml, math, time

o = yaml.safe_load(open("configs/frodobot_rover.yaml"))["odometry"]
cfg = OdometryConfig(
    wheel_radius_m=o["wheel_radius_m"], track_width_m=o["track_width_m"],
    left_rpm_indices=tuple(o["left_rpm_indices"]),
    right_rpm_indices=tuple(o["right_rpm_indices"]),
    gyro_yaw_index=o.get("gyro_yaw_index", 2),
    gyro_sign=o.get("gyro_sign", 1.0), gps_correction=False)

c = RoverClient(timeout=30)
odo = Odometry(cfg)
print("Avanza 5 s y muestra la pose en cada paso.\n")
try:
    t0 = time.time()
    while time.time() - t0 < 5.0:
        c.control(0.30, 0.0)
        raw = c.telemetry().raw
        odo.update(raw)
        print(f"  rpms={len(raw.get('rpms',[]))} gyros={len(raw.get('gyros',[]))} "
              f"integradas={odo.samples_integrated} "
              f"pose=({odo.pose.x:+.3f},{odo.pose.y:+.3f},"
              f"{math.degrees(odo.pose.theta):+.1f}gr)")
        time.sleep(0.2)
finally:
    c.stop(); c.stop()
    print("\n[frenado]")
    if odo.samples_integrated == 0:
        print(">>> NINGUNA muestra integrada. Ahi esta el problema.")
