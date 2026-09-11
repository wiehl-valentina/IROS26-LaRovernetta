"""GPS bearing/distance helpers and a field-proven heading estimator.

Adapted from the Frodobots ``auto-navigation-mini`` research stack.

Heading note (important): the Mini+ magnetometer is unreliable in the field —
motor/battery interference and site-specific magnetic anomalies made the
telemetry ``orientation`` value untrustworthy at multiple test sites, producing
a "rover spins in circles" failure. The proven alternative is GPS
course-over-ground (COG): the bearing between successive GPS fixes while the
rover is actually moving. ``HeadingEstimator`` implements that, using telemetry
orientation only as a cold-start fallback before the first meters of motion.
"""

from __future__ import annotations

import math

_EARTH_RADIUS_M = 6_371_000.0


def gps_bearing_and_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float]:
    """Return (bearing_from_north_rad, distance_m) between two WGS84 points.

    Equirectangular approximation — accurate to well under 0.5% at the
    sub-kilometer scale of an ERC mission. Bearing: 0 = north, positive
    toward east, range (-pi, pi].
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx_east = dlon * math.cos(mean_lat) * _EARTH_RADIUS_M
    dy_north = dlat * _EARTH_RADIUS_M
    bearing = math.atan2(dx_east, dy_north)
    distance = math.hypot(dx_east, dy_north)
    return bearing, distance


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


class HeadingEstimator:
    """Heading (degrees from north, clockwise) from GPS course-over-ground.

    Feed it every telemetry sample via :meth:`update`. While the rover moves,
    heading comes from the bearing between GPS fixes at least ``min_move_m``
    apart. When stationary, the last known COG is held. Before any motion has
    happened, the telemetry ``orientation`` (magnetometer) is returned as a
    cold-start fallback — a wrong initial heading costs a few meters in a
    wrong direction, after which COG takes over.
    """

    def __init__(self, min_move_m: float = 0.5, min_speed_ms: float = 0.15):
        self.min_move_m = float(min_move_m)
        self.min_speed_ms = float(min_speed_ms)
        self._anchor: tuple[float, float] | None = None  # lat, lon of last COG anchor
        self._last_cog_deg: float | None = None

    @property
    def has_cog(self) -> bool:
        return self._last_cog_deg is not None

    def update(
        self,
        lat: float | None,
        lon: float | None,
        speed_ms: float | None = None,
        orientation_deg: float | None = None,
    ) -> float | None:
        """Ingest one telemetry sample; return the current best heading in degrees.

        Returns None only when there is no GPS motion history AND no
        orientation fallback.
        """
        if lat is not None and lon is not None:
            if self._anchor is None:
                self._anchor = (float(lat), float(lon))
            else:
                bearing_rad, dist_m = gps_bearing_and_distance(
                    self._anchor[0], self._anchor[1], float(lat), float(lon)
                )
                if dist_m >= self.min_move_m and (
                    speed_ms is None or speed_ms >= self.min_speed_ms
                ):
                    self._last_cog_deg = math.degrees(bearing_rad) % 360.0
                    self._anchor = (float(lat), float(lon))

        if self._last_cog_deg is not None:
            return self._last_cog_deg
        if orientation_deg is not None:
            return float(orientation_deg) % 360.0
        return None
