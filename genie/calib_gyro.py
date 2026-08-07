from genie_rover.sdk_client import RoverClient
from genie_rover.odometry import calibrate_gyro_axis
import time

c = RoverClient(timeout=30)
print("El rover va a girar en el lugar a la IZQUIERDA, 3 segundos.")
input("Ponelo en el PISO con espacio para girar y apreta ENTER...")

muestras = []
t0 = time.time()
while time.time() - t0 < 3.0:
    c.control(0.0, 0.5)
    try:
        for g in c.telemetry().raw.get("gyros", []):
            muestras.append([float(v) for v in g])
    except Exception:
        pass
    time.sleep(0.15)
c.stop(); c.stop()

i, s = calibrate_gyro_axis(muestras)
print(f"\n{len(muestras)} muestras")
print(f"eje de guinada = columna {i}, signo {s:+.0f}")
print(f"\nPone en el config:\n  gyro_yaw_index: {i}\n  gyro_sign: {s}")
