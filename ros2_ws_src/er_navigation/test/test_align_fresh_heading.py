#!/usr/bin/env python3
"""
test_align_fresh_heading.py — Test offline de alineación con heading discreto (Brief 22 / V.2 y V.4.1)

Verifica que el controlador gps_waypoint_controller:
1. No emite ráfagas repetidas de TURN en ALIGN mientras no llegue un heading nuevo (seq > last_turn_seq).
2. Extiende la fase PAUSE en lugar de sobrevirar con datos de heading obsoletos.
3. Al recibir un heading fresco que reduce el error, transiciona ordenadamente y converge a modo DRIVE.
4. Si se supera el timeout de espera (2.5s), permite una ráfaga de reintento.
"""

import math
import pytest
import rclpy
from rclpy.duration import Duration
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist

from er_navigation.gps_waypoint_controller import GPSWaypointController


@pytest.fixture
def ros_context():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def create_test_controller():
    controller = GPSWaypointController()
    # Desactivar requerimiento estricto del gobernador para test unitario directo
    controller.require_velocity_governor = False
    controller.path_following_enabled = False
    controller.pause_after_turn_s = 0.8
    controller.heading_fresh_wait_timeout_s = 2.5
    controller.heading_max_stale_s = 5.0
    controller.heading_filter_alpha = 1.0  # Respuesta inmediata para tests unitarios sintéticos
    return controller


def test_align_burst_not_repeated_without_fresh_heading(ros_context):
    """Verifica que el controlador no repite ráfagas de TURN si no llegó heading nuevo."""
    controller = create_test_controller()

    controller._safe_velocity_limit = 1.111
    controller._safe_velocity_limit_last_rx = controller.get_clock().now()

    # 1. Configurar posición del rover y meta (a ~50m al Norte, bearing = 0°)
    controller.current_lat = -34.603700
    controller.current_lon = -58.381600
    controller._gps_last_update = controller.get_clock().now()

    target_msg = NavSatFix()
    target_msg.latitude = -34.603200  # Norte
    target_msg.longitude = -58.381600
    controller._on_target(target_msg)

    # 2. Enviar primer heading: 65.0° (meta está a 0°, heading_error = 0 - 65 = -65.0°)
    h_msg = Float32()
    h_msg.data = 65.0
    controller._on_heading(h_msg)

    assert controller._heading_seq == 1
    assert controller.current_heading == 65.0

    published_cmds = []
    controller.cmd_pub.publish = lambda twist: published_cmds.append(
        (twist.linear.x, twist.angular.z)
    )

    # 3. Ciclo 0: debe iniciar ráfaga TURN
    controller._control_loop()

    assert controller._align_phase == "TURN"
    assert controller._last_turn_heading_seq == 1
    assert len(published_cmds) > 0
    cmd_v, cmd_w = published_cmds[-1]
    assert cmd_v == 0.0
    assert abs(cmd_w) > 0.0

    # 4. Simular que expira la ráfaga de giro pasando a PAUSE
    t_pause_start = controller.get_clock().now()
    controller._begin_align_phase("PAUSE", t_pause_start)

    # 5. Simular tiempo en PAUSE transcurriendo 1.2 segundos (elapsed = 1.2s > pause_after_turn_s = 0.8s)
    # SIN haber recibido un nuevo heading (_heading_seq sigue siendo 1)
    controller._align_phase_started_at = controller.get_clock().now() - Duration(nanoseconds=int(1.2 * 1e9))

    controller._control_loop()

    # Brief 22 / V.2: NO DEBE REPETIR EL GIRO
    assert controller._align_phase == "PAUSE", (
        f"El controlador cambió de fase a {controller._align_phase} "
        f"sin nuevo heading (seq={controller._heading_seq}, last={controller._last_turn_heading_seq})"
    )
    cmd_v, cmd_w = published_cmds[-1]
    assert cmd_w == 0.0, f"Emitió giro cmd_w={cmd_w} sin heading nuevo!"

    controller.destroy_node()


def test_align_converges_when_fresh_heading_arrives(ros_context):
    """Verifica que al llegar heading nuevo con error menor, converge ordenadamente hacia DRIVE."""
    controller = create_test_controller()
    controller._safe_velocity_limit = 1.111
    controller._safe_velocity_limit_last_rx = controller.get_clock().now()

    controller.current_lat = -34.603700
    controller.current_lon = -58.381600
    controller._gps_last_update = controller.get_clock().now()

    target_msg = NavSatFix()
    target_msg.latitude = -34.603200  # Norte (bearing 0°)
    target_msg.longitude = -58.381600
    controller._on_target(target_msg)

    # Heading inicial: 65° (error -65°)
    h_msg = Float32()
    h_msg.data = 65.0
    controller._on_heading(h_msg)
    assert controller._heading_seq == 1

    published_cmds = []
    controller.cmd_pub.publish = lambda twist: published_cmds.append(
        (twist.linear.x, twist.angular.z)
    )

    # 1. Primer burst de giro
    controller._control_loop()
    assert controller._align_phase == "TURN"
    assert controller._last_turn_heading_seq == 1

    # 2. Pasa a PAUSE
    controller._begin_align_phase("PAUSE", controller.get_clock().now())
    # Esperamos el tiempo nominal de pausa (1.0s > 0.8s)
    controller._align_phase_started_at = controller.get_clock().now() - Duration(nanoseconds=int(1.0 * 1e9))

    # 3. Llega nuevo reporte del compás: el rover giró físicamente y ahora heading es 30°
    # (error = 0 - 30 = -30°, fuera de coarse_align_threshold=25° para exigir segundo burst)
    h_msg.data = 30.0
    controller._on_heading(h_msg)
    assert controller._heading_seq == 2
    assert controller._heading_seq > controller._last_turn_heading_seq

    # 4. Ahora sí debe permitir el segundo burst con la nueva información
    controller._control_loop()
    assert controller._align_phase == "TURN"
    assert controller._last_turn_heading_seq == 2

    # 5. Pasa a PAUSE de nuevo
    controller._begin_align_phase("PAUSE", controller.get_clock().now())
    controller._align_phase_started_at = controller.get_clock().now() - Duration(nanoseconds=int(1.0 * 1e9))

    # 6. Llega el tercer reporte del compás: heading es 3.0° (error = -3.0° <= 18° align threshold)
    h_msg.data = 3.0
    controller._on_heading(h_msg)
    assert controller._heading_seq == 3

    # 7. Ciclo de control: DEBE ENTRAR EN MODO DRIVE
    controller._control_loop()
    cmd_v, cmd_w = published_cmds[-1]
    assert cmd_v > 0.0, f"Se esperaba cmd_v > 0 en modo DRIVE, pero fue {cmd_v}"
    assert controller._last_turn_heading_seq is None  # Reseteado en DRIVE

    controller.destroy_node()


def test_align_timeout_retry_when_heading_lost(ros_context):
    """Verifica que si no llega heading por más de 2.5s, se dispara un reintento por timeout."""
    controller = create_test_controller()
    controller._safe_velocity_limit = 1.111
    controller._safe_velocity_limit_last_rx = controller.get_clock().now()

    controller.current_lat = -34.603700
    controller.current_lon = -58.381600
    controller._gps_last_update = controller.get_clock().now()

    target_msg = NavSatFix()
    target_msg.latitude = -34.603200
    target_msg.longitude = -58.381600
    controller._on_target(target_msg)

    h_msg = Float32()
    h_msg.data = 65.0
    controller._on_heading(h_msg)

    # Iniciar y completar un burst
    controller._control_loop()
    assert controller._last_turn_heading_seq == 1

    # Pasar a PAUSE
    controller._begin_align_phase("PAUSE", controller.get_clock().now())

    # Simular que transcurrieron 2.6 segundos en PAUSE (> 2.5s timeout)
    # sin que haya llegado ningún heading nuevo
    controller._align_phase_started_at = controller.get_clock().now() - Duration(nanoseconds=int(2.6 * 1e9))

    controller._control_loop()
    # Al superar el timeout de 2.5s, debe permitir reintento de ráfaga
    assert controller._align_phase == "TURN"

    controller.destroy_node()


def test_align_full_sequence_3hz_with_2s_compass(ros_context):
    """Simula el escenario exacto del log (controlador a 3 Hz, compás cada 2.0s).
    
    Verifica que no hay ráfagas repetidas en la ventana de 2s y que el rover
    converge progresivamente a modo DRIVE sin oscilar indefinidamente.
    """
    controller = create_test_controller()
    controller._safe_velocity_limit = 1.111
    controller._safe_velocity_limit_last_rx = controller.get_clock().now()

    controller.current_lat = -34.603700
    controller.current_lon = -58.381600
    controller._gps_last_update = controller.get_clock().now()

    # Meta al Norte (bearing 0°)
    target_msg = NavSatFix()
    target_msg.latitude = -34.603200
    target_msg.longitude = -58.381600
    controller._on_target(target_msg)

    # Estado físico simulado del rover
    # Comienza con heading 65.0° (error -65.0°)
    sim_physical_heading = 65.0
    turn_burst_count = 0
    in_turn_previous = False
    drive_entered = False

    t_sim = 0.0
    dt_ctrl = 0.333  # 3 Hz
    dt_compass = 2.0  # Compás cada 2 segundos
    last_compass_t = -10.0

    # Hook para registrar cuando se emite comando de giro
    def mock_publish(twist):
        nonlocal turn_burst_count, in_turn_previous, drive_entered
        if twist.linear.x > 0.0:
            drive_entered = True
        is_turning = abs(twist.angular.z) > 0.0
        if is_turning and not in_turn_previous:
            turn_burst_count += 1
        in_turn_previous = is_turning

    controller.cmd_pub.publish = mock_publish

    # Usar un mock clock para avanzar el tiempo de simulación
    real_clock = controller.get_clock()
    class MockClock:
        def __init__(self, base):
            self.base = base
            self.current_time = base.now()
            self.clock_type = base.clock_type
        def now(self):
            return self.current_time

    mock_clk = MockClock(real_clock)
    controller.get_clock = lambda: mock_clk

    # Simular 18 ticks (6 segundos a 3 Hz)
    for tick in range(18):
        t_sim = tick * dt_ctrl
        mock_clk.current_time = mock_clk.base.now() + Duration(nanoseconds=int(t_sim * 1e9))

        # Mantener GPS actualizado en el mock time
        controller._gps_last_update = mock_clk.now()

        # ¿Llegó reporte del compás? (a t=0.0s, t=2.0s, t=4.0s)
        if t_sim - last_compass_t >= (dt_compass - 0.01):
            last_compass_t = t_sim
            h_msg = Float32()
            h_msg.data = sim_physical_heading
            controller._on_heading(h_msg)

        # Si el controlador estuvo girando en el tick anterior, la física rota el rover
        # hacia el objetivo (0°) unos 25° por ráfaga
        if in_turn_previous:
            sim_physical_heading = max(0.0, sim_physical_heading - 25.0)

        controller._control_loop()

    # Verificaciones clave:
    # 1. No debe haber explotado en ráfagas repetidas (en 6 segundos, máximo 3 ráfagas correspondientes a los reportes)
    assert turn_burst_count <= 3, f"Se emitieron {turn_burst_count} ráfagas de giro (demasiadas para 6s con compás de 2s)!"
    # 2. Debe haber entrado en DRIVE al converger la alineación
    assert drive_entered, "El rover nunca entró en modo DRIVE durante la simulación de 6 segundos!"

    controller.destroy_node()

