import math
import numpy as np
import pytest


def combine_and_publish(
    confidence: np.ndarray,
    semantic: np.ndarray,
    semantic_layer_enabled: bool = True,
    confidence_max: float = 100.0,
    unknown_evidence_band: float = 5.0,
    semantic_override_threshold: float = 30.0,
    publish_quantization_step: int = 5,
) -> np.ndarray:
    """
    Replica exactamente la lógica de _publish_timer_cb de PersistentMapNode.
    """
    conf_copy = confidence.copy()
    sem_copy = semantic.copy()

    if semantic_layer_enabled:
        has_obstacle = conf_copy >= semantic_override_threshold
        has_semantic = np.abs(sem_copy) >= unknown_evidence_band
        grid_effective = np.where(
            has_obstacle,
            np.maximum(conf_copy, sem_copy),
            np.where(has_semantic, sem_copy, conf_copy),
        )
    else:
        grid_effective = conf_copy

    abs_conf = np.abs(grid_effective)
    unknown_mask = abs_conf < unknown_evidence_band

    scaled = np.clip(
        ((grid_effective + confidence_max) / (2.0 * confidence_max)) * 100.0,
        0.0,
        100.0,
    )
    step = max(1, int(publish_quantization_step))
    occ_grid = (np.round(np.round(scaled, 6) / step) * step).astype(np.int8)
    occ_grid[unknown_mask] = -1
    return occ_grid


def compute_dstar_cost(
    raw_data: np.ndarray,
    occupied_ref_value: int = 78,
    free_ref_value: int = 40,
    unknown_cell_cost: float = 2.5,
    max_finite_cost: float = 15.0,
) -> np.ndarray:
    """
    Replica exactamente la lógica de traducción de OccupancyGrid a costos de D* Lite
    en GlobalPlannerNode._on_map (sin inflación).
    """
    h, w = raw_data.shape
    new_costs = np.full((h, w), float("inf"), dtype=np.float32)
    known = raw_data != -1
    occupied = known & (raw_data >= occupied_ref_value)
    free_ish = known & ~occupied

    new_costs[~known] = float(unknown_cell_cost)

    span = max(1.0, float(occupied_ref_value - free_ref_value))
    norm = np.clip(
        (raw_data[free_ish].astype(np.float32) - free_ref_value) / span,
        0.0,
        1.0,
    )
    new_costs[free_ish] = 1.0 + norm * (max_finite_cost - 1.0)
    return new_costs


def test_five_cases_contract():
    """
    Verifica numéricamente los 5 casos críticos del contrato de costos D* Lite.
    """
    # 5 celdas de prueba:
    # 0: Vereda OSM, sin evidencia cámara (sem=-24, conf=0)
    # 1: Calle OSM, sin evidencia cámara (sem=-6, conf=0)
    # 2: Sin prior ni evidencia (sem=0, conf=0)
    # 3: Obstáculo confirmado cámara, sin prior (sem=0, conf=55)
    # 4: Vereda OSM con obstáculo confirmado cámara (sem=-24, conf=55)
    conf = np.array([[0.0, 0.0, 0.0, 55.0, 55.0]], dtype=np.float32)
    sem = np.array([[-24.0, -6.0, 0.0, 0.0, -24.0]], dtype=np.float32)

    occ = combine_and_publish(conf, sem, semantic_layer_enabled=True, semantic_override_threshold=30.0)
    costs = compute_dstar_cost(occ)

    print("\n" + "=" * 70)
    print("TEST DE LOS 5 CASOS DE CONTRATO NUMÉRICO (Parte B / D)")
    print("=" * 70)
    case_names = [
        "1. Vereda OSM (sem=-24, conf=0)",
        "2. Calle OSM (sem=-6, conf=0)",
        "3. Desconocido (sem=0, conf=0)",
        "4. Obstáculo confirmado (sem=0, conf=55)",
        "5. Vereda con obstáculo (sem=-24, conf=55)",
    ]
    for i, name in enumerate(case_names):
        cost_str = f"{costs[0, i]:.4f}" if not math.isinf(costs[0, i]) else "inf"
        print(f"  {name:45s} -> OccGrid={occ[0, i]:3d} | D* Cost={cost_str}")

    # Caso 1: Vereda = 1.0
    assert occ[0, 0] == 40
    assert math.isclose(costs[0, 0], 1.0, abs_tol=1e-4)

    # Caso 2: Calle ≈ 2.84
    assert occ[0, 1] == 45
    assert math.isclose(costs[0, 1], 2.842105, abs_tol=1e-4)

    # Caso 3: Desconocido = 2.5
    assert occ[0, 2] == -1
    assert math.isclose(costs[0, 2], 2.5, abs_tol=1e-4)

    # Caso 4: Obstáculo = inf
    assert occ[0, 3] == 80
    assert math.isinf(costs[0, 3])

    # Caso 5: Vereda con obstáculo = inf (no diluido)
    assert occ[0, 4] == 80
    assert math.isinf(costs[0, 4])
    print(">>> TODOS LOS 5 CASOS PASARON SATISFACTORIAMENTE.\n")


def test_temporal_decay_vereda():
    """
    Simula 50 ciclos de decaimiento (50 x 5s = 250s ~ 4.16 min) sobre una celda de vereda
    sin observaciones de cámara, demostrando persistencia de costo 1.0 vs fallo a 2.5 en canal único.
    """
    decay_factor = 0.7738
    num_cycles = 50

    # Nueva arquitectura (dos canales separados)
    conf_dual = np.array([[0.0]], dtype=np.float32)
    sem_dual = np.array([[-24.0]], dtype=np.float32)

    # Arquitectura previa (canal único sobre _confidence)
    conf_legacy = np.array([[-24.0]], dtype=np.float32)

    print("=" * 70)
    print(f"TEST TEMPORAL DE DECAIMIENTO: {num_cycles} CICLOS (~4 MINUTOS)")
    print("=" * 70)

    for cycle in range(1, num_cycles + 1):
        # Simular decaimiento en canal dinámico
        conf_dual *= decay_factor
        conf_legacy *= decay_factor

        if cycle in (1, 2, 5, 10, 20, 50):
            occ_dual = combine_and_publish(conf_dual, sem_dual, semantic_layer_enabled=True, semantic_override_threshold=30.0)
            cost_dual = compute_dstar_cost(occ_dual)[0, 0]

            occ_legacy = combine_and_publish(conf_legacy, np.zeros_like(conf_legacy), semantic_layer_enabled=False)
            cost_legacy = compute_dstar_cost(occ_legacy)[0, 0]

            print(
                f"  Ciclo {cycle:2d} ({cycle * 5:3d}s): "
                f"DUAL (nuevo) -> conf={conf_dual[0,0]:.4f}, sem={sem_dual[0,0]:.1f}, occ={occ_dual[0,0]:2d}, D* cost={cost_dual:.2f} | "
                f"LEGACY (viejo) -> conf={conf_legacy[0,0]:.4f}, occ={occ_legacy[0,0]:2d}, D* cost={cost_legacy:.2f}"
            )

    # Verificación final tras 50 ciclos
    occ_final_dual = combine_and_publish(conf_dual, sem_dual, semantic_layer_enabled=True, semantic_override_threshold=30.0)
    cost_final_dual = compute_dstar_cost(occ_final_dual)[0, 0]

    occ_final_legacy = combine_and_publish(conf_legacy, np.zeros_like(conf_legacy), semantic_layer_enabled=False)
    cost_final_legacy = compute_dstar_cost(occ_final_legacy)[0, 0]

    # En la arquitectura nueva, el costo permanece exactamente 1.0
    assert math.isclose(cost_final_dual, 1.0, abs_tol=1e-4)
    # En la arquitectura vieja, el costo degradó a 2.5 (desconocido)
    assert math.isclose(cost_final_legacy, 2.5, abs_tol=1e-4)

    print(f"\n>>> RESULTADO FINAL TRAS 50 CICLOS:")
    print(f"    - Arquitectura Nueva (Dual Channel): Costo D* Lite = {cost_final_dual:.2f} (PRESERVADO)")
    print(f"    - Arquitectura Vieja (Single Channel): Costo D* Lite = {cost_final_legacy:.2f} (DEGRADADO A DESCONOCIDO)")
    print("=" * 70 + "\n")


def test_semantic_layer_disabled_equivalence():
    """
    Verifica que con semantic_layer_enabled=False el comportamiento es 100% idéntico al actual
    sobre entradas sintéticas variadas.
    """
    rng = np.random.RandomState(42)
    synthetic_conf = rng.uniform(-100.0, 100.0, size=(20, 20)).astype(np.float32)
    synthetic_sem = rng.choice([-80.0, -20.0, 0.0], size=(20, 20)).astype(np.float32) * 0.3

    # Publicación con semantic_layer_enabled=False
    occ_disabled = combine_and_publish(synthetic_conf, synthetic_sem, semantic_layer_enabled=False)

    # Publicación puramente legacy (usando solo confidence)
    abs_conf = np.abs(synthetic_conf)
    unknown_mask = abs_conf < 5.0
    scaled = np.clip(((synthetic_conf + 100.0) / 200.0) * 100.0, 0.0, 100.0)
    occ_legacy = (np.round(np.round(scaled, 6) / 5) * 5).astype(np.int8)
    occ_legacy[unknown_mask] = -1

    np.testing.assert_array_equal(occ_disabled, occ_legacy)
    print(">>> VERIFICACIÓN DE EQUIVALENCIA CON semantic_layer_enabled=False: 100% IDÉNTICO.")


def test_neutral_cells_regression():
    """
    PARTE D: Verifica que para celdas sin prior semántico (S = 0.0), el nuevo umbral
    semantic_override_threshold no altera en lo absoluto el comportamiento dinámico.
    """
    sem_zero = np.zeros((1, 100), dtype=np.float32)
    # Barrer valores de C cruzando unknown_evidence_band (5.0) y semantic_override_threshold (30.0)
    conf_sweep = np.linspace(-100.0, 100.0, 100, dtype=np.float32).reshape((1, 100))

    occ_dual_30 = combine_and_publish(conf_sweep, sem_zero, semantic_layer_enabled=True, semantic_override_threshold=30.0)
    occ_dual_45 = combine_and_publish(conf_sweep, sem_zero, semantic_layer_enabled=True, semantic_override_threshold=45.0)
    occ_dual_5 = combine_and_publish(conf_sweep, sem_zero, semantic_layer_enabled=True, semantic_override_threshold=5.0)
    occ_disabled = combine_and_publish(conf_sweep, sem_zero, semantic_layer_enabled=False)

    # Todos deben ser idénticos bit a bit
    np.testing.assert_array_equal(occ_dual_30, occ_disabled)
    np.testing.assert_array_equal(occ_dual_45, occ_disabled)
    np.testing.assert_array_equal(occ_dual_5, occ_disabled)
    print(">>> PARTE D: CELDAS SIN PRIOR (S=0.0) SON 100% IDÉNTICAS ANTE CUALQUIER UMBRAL.")


if __name__ == "__main__":
    test_five_cases_contract()
    test_temporal_decay_vereda()
    test_semantic_layer_disabled_equivalence()
    test_neutral_cells_regression()
