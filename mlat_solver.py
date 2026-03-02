"""
MLAT (Multilateration) Solver
Determines aircraft 3D position from Time Difference of Arrival (TDOA) 
measurements across distributed Mode-S receivers.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Speed of light (m/s)
C = 299_792_458.0

# WGS84 ellipsoid constants
WGS84_A = 6_378_137.0          # semi-major axis (m)
WGS84_F = 1 / 298.257_223_563  # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B / WGS84_A) ** 2


@dataclass
class ReceiverStation:
    """A ground-based Mode-S receiver."""
    station_id: str
    lat: float       # WGS84 degrees
    lon: float       # WGS84 degrees
    alt: float       # meters above ellipsoid
    _ecef: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def ecef(self) -> np.ndarray:
        """ECEF (Earth-Centered, Earth-Fixed) coordinates in meters."""
        if self._ecef is None:
            self._ecef = geodetic_to_ecef(self.lat, self.lon, self.alt)
        return self._ecef


@dataclass
class TDOAMeasurement:
    """Time Difference of Arrival between two receivers for a single Mode-S message."""
    icao: str               # 24-bit ICAO aircraft address
    ref_station_id: str     # reference (master) station
    other_station_id: str   # secondary station
    tdoa_ns: float          # TDOA in nanoseconds (other - ref). Positive = signal reached ref first.
    timestamp_utc: float    # Unix UTC timestamp of message receipt at ref station (seconds)
    mlat_quality: float = 1.0  # weight: 0.0 (bad) – 1.0 (excellent)


@dataclass
class MLATResult:
    """Position solution from multilateration."""
    icao: str
    lat: float          # WGS84 degrees
    lon: float          # WGS84 degrees
    alt_m: float        # meters above WGS84 ellipsoid
    timestamp_utc: float
    residual_rms_m: float     # RMS of range residuals (meters)
    num_stations: int
    converged: bool
    covariance: Optional[np.ndarray] = None  # 3×3 position covariance (m²)

    @property
    def hdop(self) -> float:
        """Horizontal Dilution of Precision (if covariance available)."""
        if self.covariance is None:
            return float("nan")
        # Approximate from horizontal covariance (East-North components)
        # Convert covariance to ENU for HDOP
        return float(np.sqrt(np.trace(self.covariance[:2, :2])))


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS84 geodetic → ECEF (m)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + alt_m) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_geodetic(x: float, y: float, z: float):
    """Convert ECEF → WGS84 geodetic (Bowring iterative method)."""
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))  # initial estimate
    for _ in range(10):  # converges in 3-4 iterations
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N if abs(np.cos(lat)) > 1e-10 else abs(z) / np.sin(lat) - N * (1 - WGS84_E2)
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


# ---------------------------------------------------------------------------
# Core MLAT solver
# ---------------------------------------------------------------------------

class MLATSolver:
    """
    Iterative least-squares multilateration solver using TDOA measurements.

    Algorithm:
        1. Select reference station (typically the one with most measurements).
        2. Form TDOA observation equations (hyperbolic surfaces in ECEF space).
        3. Linearise around current estimate and solve normal equations.
        4. Iterate until convergence or max iterations reached.
        5. Estimate position covariance from Jacobian.

    Requires ≥ 4 stations (N-1 ≥ 3 independent TDOAs) for a 3D solution.
    """

    MIN_STATIONS = 4
    MAX_ITER = 50
    CONVERGENCE_THRESHOLD_M = 0.1   # stop when position update < 0.1 m
    MAX_RESIDUAL_M = 5000.0          # reject solutions with residuals > 5 km

    def __init__(self, stations: dict[str, ReceiverStation]):
        """
        Parameters
        ----------
        stations : dict mapping station_id → ReceiverStation
        """
        self.stations = stations

    # ------------------------------------------------------------------
    def solve(
        self,
        icao: str,
        tdoa_measurements: list[TDOAMeasurement],
        timestamp_utc: float,
        initial_ecef: Optional[np.ndarray] = None,
    ) -> Optional[MLATResult]:
        """
        Solve for aircraft position given a set of TDOA measurements.

        Parameters
        ----------
        icao           : ICAO hex address of aircraft
        tdoa_measurements : list of TDOAMeasurement (same epoch)
        timestamp_utc  : epoch of the observation cluster
        initial_ecef   : optional starting ECEF guess (m)

        Returns
        -------
        MLATResult or None if solution failed
        """
        # ---- Group measurements by reference station ----
        ref_groups = {}
        for m in tdoa_measurements:
            ref_groups.setdefault(m.ref_station_id, []).append(m)

        # Pick the reference with the most observations
        ref_id = max(ref_groups, key=lambda k: len(ref_groups[k]))
        obs = ref_groups[ref_id]

        # Gather unique stations involved
        station_ids = {ref_id} | {m.other_station_id for m in obs}
        if len(station_ids) < self.MIN_STATIONS:
            logger.debug(
                "MLAT: only %d stations for %s, need %d",
                len(station_ids), icao, self.MIN_STATIONS
            )
            return None

        # Resolve station objects
        try:
            ref_ecef = self.stations[ref_id].ecef
            pairs = [
                (self.stations[m.other_station_id].ecef, m.tdoa_ns * 1e-9 * C, m.mlat_quality)
                for m in obs
                if m.other_station_id in self.stations
            ]
        except KeyError as e:
            logger.warning("Unknown station %s", e)
            return None

        if len(pairs) < 3:
            return None

        # ---- Initial position estimate ----
        x0 = initial_ecef if initial_ecef is not None else self._centroid_guess(station_ids)

        # ---- Nonlinear least squares ----
        try:
            result = least_squares(
                self._residuals,
                x0,
                args=(ref_ecef, pairs),
                method="lm",
                max_nfev=1000,
                ftol=1e-9,
                xtol=1e-9,
            )
        except Exception as exc:
            logger.warning("MLAT least_squares failed for %s: %s", icao, exc)
            return None

        if not result.success and result.cost > 1e6:
            logger.debug("MLAT did not converge for %s", icao)

        x_ecef = result.x
        rms = float(np.sqrt(result.cost / len(pairs)))

        if rms > self.MAX_RESIDUAL_M:
            logger.debug("MLAT high residual %.0f m for %s – rejecting", rms, icao)
            return None

        lat, lon, alt = ecef_to_geodetic(*x_ecef)

        # ---- Covariance estimate ----
        cov = self._estimate_covariance(result, pairs)

        converged = result.success or rms < 100.0

        return MLATResult(
            icao=icao,
            lat=lat,
            lon=lon,
            alt_m=alt,
            timestamp_utc=timestamp_utc,
            residual_rms_m=rms,
            num_stations=len(station_ids),
            converged=converged,
            covariance=cov,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _residuals(x: np.ndarray, ref_ecef: np.ndarray, pairs) -> np.ndarray:
        """
        Residual vector for least_squares.
        Each element = observed_range_diff - computed_range_diff  (meters)
        """
        r_ref = float(np.linalg.norm(x - ref_ecef))
        residuals = []
        for station_ecef, obs_range_diff, weight in pairs:
            r_i = float(np.linalg.norm(x - station_ecef))
            computed = r_i - r_ref
            residuals.append(weight * (obs_range_diff - computed))
        return np.array(residuals)

    def _centroid_guess(self, station_ids: set) -> np.ndarray:
        """Centroid of involved stations as initial guess (plus 10 km altitude)."""
        ecefs = [self.stations[sid].ecef for sid in station_ids if sid in self.stations]
        centroid = np.mean(ecefs, axis=0)
        # push outward by 10 km to represent a plausible aircraft altitude
        centroid *= 1 + 10_000 / np.linalg.norm(centroid)
        return centroid

    @staticmethod
    def _estimate_covariance(result, pairs) -> Optional[np.ndarray]:
        """Approximate 3×3 covariance from Jacobian (m²), scaled by residual variance."""
        try:
            J = result.jac
            JTJ = J.T @ J
            # σ² = RSS / (n - p)
            n, p = J.shape
            dof = max(n - p, 1)
            sigma2 = 2.0 * result.cost / dof
            cov_ecef = np.linalg.pinv(JTJ) * sigma2
            return cov_ecef
        except Exception:
            return None
