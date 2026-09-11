import math

from rover_traversability.geo import (
    HeadingEstimator,
    gps_bearing_and_distance,
    wrap_angle_deg,
)

# ~1e-5 deg latitude ~= 1.11 m
LAT0, LON0 = -34.9215, -57.9545  # La Plata


def test_due_north():
    bearing, dist = gps_bearing_and_distance(LAT0, LON0, LAT0 + 0.001, LON0)
    assert abs(math.degrees(bearing)) < 0.1
    assert abs(dist - 111.19) < 1.5


def test_due_east():
    bearing, dist = gps_bearing_and_distance(LAT0, LON0, LAT0, LON0 + 0.001)
    assert abs(math.degrees(bearing) - 90.0) < 0.1
    assert 85 < dist < 95  # cos(lat) shortened


def test_wrap_angle():
    assert wrap_angle_deg(190) == -170
    assert wrap_angle_deg(-190) == 170
    assert wrap_angle_deg(180) == 180
    assert wrap_angle_deg(0) == 0
    assert wrap_angle_deg(720 + 45) == 45


def test_heading_cold_start_falls_back_to_orientation():
    est = HeadingEstimator()
    h = est.update(LAT0, LON0, speed_ms=0.0, orientation_deg=123.0)
    assert h == 123.0
    assert not est.has_cog


def test_heading_switches_to_cog_after_motion():
    est = HeadingEstimator(min_move_m=0.5)
    est.update(LAT0, LON0, speed_ms=0.5, orientation_deg=270.0)
    # Move ~11m north: COG should say ~0 deg regardless of the (wrong) compass.
    h = est.update(LAT0 + 0.0001, LON0, speed_ms=0.5, orientation_deg=270.0)
    assert est.has_cog
    assert h is not None and (h < 5 or h > 355)


def test_heading_held_while_stationary():
    est = HeadingEstimator(min_move_m=0.5)
    est.update(LAT0, LON0, speed_ms=0.5)
    est.update(LAT0 + 0.0001, LON0, speed_ms=0.5)
    h = est.update(LAT0 + 0.0001, LON0, speed_ms=0.0, orientation_deg=200.0)
    assert h is not None and (h < 5 or h > 355)  # holds COG, ignores compass


def test_heading_none_without_any_signal():
    est = HeadingEstimator()
    assert est.update(None, None) is None
