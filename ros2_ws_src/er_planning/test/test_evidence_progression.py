import math
import numpy as np
import pytest

from test_dual_channel_persistent_map import combine_and_publish, compute_dstar_cost


def format_cost(cost: float) -> str:
    return f"{cost:.4f}" if not math.isinf(cost) else "inf"


def test_sequence_1_consecutive_hits():
    """
    Secuencia 1: Hits consecutivos sin decaimiento (+15 en cada paso).
    """
    S_vereda = -24.0
    hit_gain = 15.0
    cost_calle_limpia = 2.842105  # Referencia de calle limpia

    c_dual = 0.0
    c_legacy = S_vereda

    rows = []

    for step in range(0, 8):
        if step > 0:
            c_dual = min(100.0, c_dual + hit_gain)
            c_legacy = min(100.0, c_legacy + hit_gain)

        # Legacy
        occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
        cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]

        # Dual sin corregir (override=5.0)
        occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
        cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]

        # Dual corregido (override=30.0)
        occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
        cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]

        # Dual corregido (override=45.0)
        occ_c45 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=45.0)[0, 0]
        cost_c45 = compute_dstar_cost(np.array([[occ_c45]]))[0, 0]

        action_name = "Inicial" if step == 0 else f"Hit #{step} (+15)"
        rows.append((step, action_name, c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30, cost_c45, occ_c45))

    print("\n" + "=" * 125)
    print("SECUENCIA 1: HITS CONSECUTIVOS SIN DECAIMIENTO (RÁFAGA CONTINUA DE CÁMARA)")
    print("=" * 125)
    header = f"{'Paso':<5} | {'Acción':<15} | {'C_dual':<7} | {'C_leg':<7} | {'Costo Leg':<10} (Occ) | {'Dual Sin Corr':<13} (Occ) | {'Dual (Th=30)':<12} (Occ) | {'Dual (Th=45)':<12} (Occ)"
    print(header)
    print("-" * 125)
    for r in rows:
        p, act, cd, cl, c_l, o_l, c_u, o_u, c_30, o_30, c_45, o_45 = r
        print(f"{p:<5} | {act:<15} | {cd:<7.1f} | {cl:<7.1f} | {format_cost(c_l):<10} ({o_l:3d}) | {format_cost(c_u):<13} ({o_u:3d}) | {format_cost(c_30):<12} ({o_30:3d}) | {format_cost(c_45):<12} ({o_45:3d})")
    print("-" * 125)
    print("UMBRAL SECUENCIA 1 (Hits para superar costo de calle > 2.84):")
    print("  - Legacy (canal único):       2 hits (Hit 1 = 2.8421, Hit 2 = 6.5263 > 2.84)")
    print("  - Dual sin corregir (Th=5):   1 hit  (Hit 1 = 8.3684 > 2.84)")
    print("  - Dual corregido (Th=30):     2 hits (Hit 1 = 1.0000, Hit 2 = 10.2105 > 2.84)")
    print("  - Dual corregido (Th=45):     3 hits (Hit 1 = 1.0000, Hit 2 = 1.0000, Hit 3 = 12.0526 > 2.84)")
    print("=" * 125)


def test_sequence_2_hits_interleaved_decay():
    """
    Secuencia 2: Hits intercalados con decaimiento (+15, decay *0.7738, +15, decay *0.7738...).
    """
    S_vereda = -24.0
    hit_gain = 15.0
    decay_factor = 0.7738

    c_dual = 0.0
    c_legacy = S_vereda

    rows = []

    # Paso 0: Inicial
    occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
    cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]
    occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
    cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]
    occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
    cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]
    occ_c45 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=45.0)[0, 0]
    cost_c45 = compute_dstar_cost(np.array([[occ_c45]]))[0, 0]
    rows.append(("0", "Inicial", c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30, cost_c45, occ_c45))

    for hit_idx in range(1, 10):
        # 1. Aplicar Hit (+15)
        c_dual = min(100.0, c_dual + hit_gain)
        c_legacy = min(100.0, c_legacy + hit_gain)

        occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
        cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]
        occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
        cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]
        occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
        cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]
        occ_c45 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=45.0)[0, 0]
        cost_c45 = compute_dstar_cost(np.array([[occ_c45]]))[0, 0]
        rows.append((f"{hit_idx}a", f"Hit #{hit_idx} (+15)", c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30, cost_c45, occ_c45))

        # 2. Aplicar Decaimiento (5s)
        c_dual *= decay_factor
        c_legacy *= decay_factor

        occ_leg_d = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
        cost_leg_d = compute_dstar_cost(np.array([[occ_leg_d]]))[0, 0]
        occ_uncorr_d = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
        cost_uncorr_d = compute_dstar_cost(np.array([[occ_uncorr_d]]))[0, 0]
        occ_c30_d = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
        cost_c30_d = compute_dstar_cost(np.array([[occ_c30_d]]))[0, 0]
        occ_c45_d = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=45.0)[0, 0]
        cost_c45_d = compute_dstar_cost(np.array([[occ_c45_d]]))[0, 0]
        rows.append((f"{hit_idx}b", f"Decay (x0.7738)", c_dual, c_legacy, cost_leg_d, occ_leg_d, cost_uncorr_d, occ_uncorr_d, cost_c30_d, occ_c30_d, cost_c45_d, occ_c45_d))

    print("\n" + "=" * 125)
    print("SECUENCIA 2: HITS INTERCALADOS CON DECAIMIENTO (PATRÓN REALISTA CADA 5s)")
    print("=" * 125)
    header = f"{'Paso':<5} | {'Acción':<16} | {'C_dual':<8} | {'C_leg':<8} | {'Costo Leg':<10} (Occ) | {'Dual Sin Corr':<13} (Occ) | {'Dual (Th=30)':<12} (Occ) | {'Dual (Th=45)':<12} (Occ)"
    print(header)
    print("-" * 125)
    for r in rows:
        p, act, cd, cl, c_l, o_l, c_u, o_u, c_30, o_30, c_45, o_45 = r
        print(f"{p:<5} | {act:<16} | {cd:<8.4f} | {cl:<8.4f} | {format_cost(c_l):<10} ({o_l:3d}) | {format_cost(c_u):<13} ({o_u:3d}) | {format_cost(c_30):<12} ({o_30:3d}) | {format_cost(c_45):<12} ({o_45:3d})")
    print("-" * 125)
    print("UMBRAL SECUENCIA 2 (Hits espaciados para superar costo de calle > 2.84):")
    print("  - Legacy (canal único):       2 hits espaciados (Hit 2a = 6.5263 > 2.84)")
    print("  - Dual sin corregir (Th=5):   1 hit  espaciado  (Hit 1a = 8.3684 > 2.84)")
    print("  - Dual corregido (Th=30):     3 hits espaciados (Hit 3a = 12.0526 > 2.84; Hit 2a C=26.61 < 30 da 1.00)")
    print("  - Dual corregido (Th=45):     5 hits espaciados (Hit 5a = 13.8947 > 2.84; Hit 4a C=42.54 < 45 da 1.00)")
    print("=" * 125)


def test_sequence_3_hit_then_misses():
    """
    Secuencia 3: Un solo Hit seguido de Misses sucesivos (+15, -10, -10, -10).
    """
    S_vereda = -24.0
    hit_gain = 15.0
    miss_gain = 10.0

    c_dual = 0.0
    c_legacy = S_vereda

    rows = []

    # Paso 0: Inicial
    occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
    cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]
    occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
    cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]
    occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
    cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]
    rows.append((0, "Inicial", c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30))

    # Paso 1: Hit #1 (+15)
    c_dual = min(100.0, c_dual + hit_gain)
    c_legacy = min(100.0, c_legacy + hit_gain)
    occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
    cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]
    occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
    cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]
    occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
    cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]
    rows.append((1, "Hit #1 (+15)", c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30))

    # Pasos 2, 3, 4: Misses (-10)
    for step in range(2, 5):
        c_dual = max(-100.0, c_dual - miss_gain)
        c_legacy = max(-100.0, c_legacy - miss_gain)
        occ_leg = combine_and_publish(np.array([[c_legacy]]), np.zeros((1,1)), semantic_layer_enabled=False)[0, 0]
        cost_leg = compute_dstar_cost(np.array([[occ_leg]]))[0, 0]
        occ_uncorr = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=5.0)[0, 0]
        cost_uncorr = compute_dstar_cost(np.array([[occ_uncorr]]))[0, 0]
        occ_c30 = combine_and_publish(np.array([[c_dual]]), np.array([[S_vereda]]), semantic_layer_enabled=True, semantic_override_threshold=30.0)[0, 0]
        cost_c30 = compute_dstar_cost(np.array([[occ_c30]]))[0, 0]
        rows.append((step, f"Miss #{step-1} (-10)", c_dual, c_legacy, cost_leg, occ_leg, cost_uncorr, occ_uncorr, cost_c30, occ_c30))

    print("\n" + "=" * 115)
    print("SECUENCIA 3: UN SOLO HIT SEGUIDO DE MISSES (+15, -10, -10, -10) - TEST DE REVERSIÓN TRAS 1 HIT")
    print("=" * 115)
    header = f"{'Paso':<5} | {'Acción':<16} | {'C_dual':<8} | {'C_leg':<8} | {'Costo Leg':<10} (Occ) | {'Dual Sin Corr':<13} (Occ) | {'Dual (Th=30)':<12} (Occ)"
    print(header)
    print("-" * 115)
    for r in rows:
        p, act, cd, cl, c_l, o_l, c_u, o_u, c_30, o_30 = r
        print(f"{p:<5} | {act:<16} | {cd:<8.1f} | {cl:<8.1f} | {format_cost(c_l):<10} ({o_l:3d}) | {format_cost(c_u):<13} ({o_u:3d}) | {format_cost(c_30):<12} ({o_30:3d})")
    print("-" * 115)
    print("UMBRAL SECUENCIA 3 (Reversión tras 1 Hit aislado):")
    print("  - Legacy (canal único):       1 miss para volver a costo 1.0000 (Hit 1 sube a 2.8421; Miss 1 vuelve a 1.0000)")
    print("  - Dual sin corregir (Th=5):   2 misses para volver a costo 1.0000 (Hit 1 sube a 8.3684; Miss 1 da 4.6842; Miss 2 da 1.0000)")
    print("  - Dual corregido (Th=30):     0 misses necesarios (1 hit aislado C=15.0 < 30 no altera el costo 1.0000)")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    test_sequence_1_consecutive_hits()
    test_sequence_2_hits_interleaved_decay()
    test_sequence_3_hit_then_misses()
