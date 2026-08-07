import numpy as np

from rover_traversability.policy import PolicyConfig, suggest_command

from conftest import bimodal_mask, half_open_mask, make_mask


def test_all_drivable_goes_forward_straight():
    d = suggest_command(make_mask(value=1.0))
    assert not d.stop
    assert d.reason == "forward"
    assert d.linear > 0
    assert abs(d.angular) < 0.05


def test_left_open_turns_left():
    d = suggest_command(half_open_mask("left"))
    assert not d.stop
    assert d.angular > 0  # positive = LEFT (SDK convention)


def test_right_open_turns_right():
    d = suggest_command(half_open_mask("right"))
    assert not d.stop
    assert d.angular < 0


def test_all_blocked_stops():
    d = suggest_command(make_mask(value=0.0))
    assert d.stop
    assert d.reason == "blocked"
    assert d.linear == 0.0 and d.angular == 0.0


def test_bimodal_picks_an_opening_not_the_wall():
    d = suggest_command(bimodal_mask())
    assert not d.stop
    n = len(d.corridor_scores)
    assert d.best_corridor != n // 2  # not the blocked center
    assert d.reason == "turning_to_corridor"
    assert d.angular != 0.0


def test_nan_mask_is_no_data():
    m = np.full((120, 160), np.nan, dtype=np.float32)
    d = suggest_command(m)
    assert d.stop
    assert d.reason == "no_data"


def test_tiny_mask_is_no_data():
    d = suggest_command(np.ones((4, 4), dtype=np.float32))
    assert d.stop
    assert d.reason == "no_data"


def test_outputs_bounded_by_config():
    cfg = PolicyConfig(max_linear=0.3, max_angular=0.2)
    for mask in (make_mask(), half_open_mask("left"), half_open_mask("right"), bimodal_mask()):
        d = suggest_command(mask, cfg)
        assert -1.0 <= d.linear <= cfg.max_linear + 1e-9
        assert abs(d.angular) <= cfg.max_angular + 1e-9


def test_deterministic():
    m = half_open_mask("left")
    assert suggest_command(m) == suggest_command(m)


def test_goal_bias_steers_toward_goal_when_open():
    open_mask = make_mask(value=1.0)
    d_right = suggest_command(open_mask, goal_offset_deg=30.0)   # goal to the right
    d_left = suggest_command(open_mask, goal_offset_deg=-30.0)   # goal to the left
    assert d_right.angular < 0  # turn right
    assert d_left.angular > 0   # turn left


def test_goal_bias_does_not_override_obstacles():
    """Goal to the right but only the left is drivable: safety wins."""
    d = suggest_command(half_open_mask("left"), goal_offset_deg=40.0)
    assert not d.stop
    assert d.angular > 0  # still goes left, where it can actually drive


def test_partial_veer_keeps_moving():
    """Center open but more space on the left: forward with a left bias."""
    m = np.zeros((120, 160), dtype=np.float32)
    m[:, : int(160 * 2 / 3)] = 1.0
    d = suggest_command(m)
    assert d.reason == "forward"
    assert d.linear > 0
    assert d.angular > 0
