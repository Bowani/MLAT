"""
Clock Synchronization for Distributed Mode-S Receivers

Accurate TDOA-based MLAT requires all receivers to share a common time
reference with sub-microsecond accuracy. This module implements two
complementary synchronisation strategies:

1. GPS Disciplined Oscillator (GPSDO) — hardware-level, ~10 ns accuracy.
   Receivers with GPS receivers are assumed to report timestamps already
   referenced to UTC.  Clock offsets are estimated from aircraft at known
   positions (over-determined MLAT solutions).

2. Passive Mode-S Synchronisation — uses DF17/DF11 messages received by
   overlapping receiver pairs to estimate relative clock offsets.  Two
   receivers that both hear the same aircraft transmission can compute their
   relative offset (modulo propagation geometry uncertainty).

Reference:
    Schäfer, M. et al. "Bringing Up OpenSky: A Large-scale ADS-B Sensor
    Network for Research." IPSN 2014.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from mlat_solver import ReceiverStation, geodetic_to_ecef, C

logger = logging.getLogger(__name__)


@dataclass
class ClockState:
    """Estimated clock state for one receiver relative to a master reference."""
    station_id: str
    offset_ns: float = 0.0       # additive correction to reported timestamps
    drift_ppb: float = 0.0       # clock drift in parts-per-billion
    last_updated: float = field(default_factory=time.time)
    num_samples: int = 0
    rms_residual_ns: float = float("inf")

    def apply(self, raw_timestamp_ns: int) -> int:
        """Return clock-corrected timestamp."""
        age = time.time() - self.last_updated
        drift_correction = int(self.drift_ppb * age * 1e-9 * raw_timestamp_ns)
        return int(raw_timestamp_ns + self.offset_ns + drift_correction)


class PassiveClockSync:
    """
    Estimates relative clock offsets between receiver pairs using
    Mode-S messages received by both.

    Algorithm:
        For each pair of receivers (A, B) that both heard the same
        Mode-S transmission from aircraft I at position r_I:

            TDOA_AB = t_B - t_A  (observed, raw timestamps)
            TDOA_AB_geo = (|r_I - r_B| - |r_I - r_A|) / c  (expected from geometry)

            clock_offset(B vs A) = TDOA_AB - TDOA_AB_geo

        When r_I is unknown, we collect many such equations and solve for
        offsets jointly (requires ≥ 3 receivers with mutual coverage).

        When r_I is known (e.g., from ADS-B), direct single-epoch offset
        estimation is possible.
    """

    WINDOW_S = 60.0           # rolling window for offset estimation
    MIN_SAMPLES = 10          # minimum coincident observations per pair
    MAX_OFFSET_NS = 1e6       # reject pairs with > 1 ms apparent offset

    def __init__(self, stations: dict[str, ReceiverStation]):
        self.stations = stations
        # pair_id → deque of (offset_estimate_ns, timestamp)
        self._samples: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=500))
        self.clock_states: dict[str, ClockState] = {
            sid: ClockState(station_id=sid) for sid in stations
        }

    def ingest_coincident(
        self,
        station_a: str,
        ts_a_ns: int,
        station_b: str,
        ts_b_ns: int,
        aircraft_ecef: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record a coincident observation (same message, two stations).

        If aircraft_ecef is provided (e.g. from ADS-B), compute a direct
        offset estimate.  Otherwise store the raw TDOA for later batch solve.
        """
        if aircraft_ecef is not None:
            # Direct offset estimate
            r_a = np.linalg.norm(aircraft_ecef - self.stations[station_a].ecef)
            r_b = np.linalg.norm(aircraft_ecef - self.stations[station_b].ecef)
            tdoa_geo_ns = (r_b - r_a) / C * 1e9
            tdoa_obs_ns = float(ts_b_ns - ts_a_ns)
            offset_ba_ns = tdoa_obs_ns - tdoa_geo_ns  # clock(B) - clock(A)
        else:
            # Store raw TDOA; offset solved externally when geometry is known
            offset_ba_ns = float(ts_b_ns - ts_a_ns)

        if abs(offset_ba_ns) > self.MAX_OFFSET_NS:
            return  # unreasonable — skip

        pair = tuple(sorted([station_a, station_b]))
        sign = 1.0 if (station_a, station_b) == pair else -1.0
        self._samples[pair].append((sign * offset_ba_ns, time.time()))

    def update_offsets(self) -> None:
        """Recompute clock offset estimates from accumulated samples."""
        cutoff = time.time() - self.WINDOW_S
        for pair, samples in self._samples.items():
            # Purge old
            fresh = [(v, t) for v, t in samples if t >= cutoff]
            if len(fresh) < self.MIN_SAMPLES:
                continue
            values = np.array([v for v, _ in fresh])
            median_offset = float(np.median(values))
            rms = float(np.sqrt(np.mean((values - median_offset) ** 2)))

            sid_a, sid_b = pair
            # Update relative offset: B relative to A
            # We hold A fixed (or a designated master) and update B
            if sid_a in self.clock_states:
                self.clock_states[sid_b].offset_ns = median_offset + self.clock_states[sid_a].offset_ns
                self.clock_states[sid_b].rms_residual_ns = rms
                self.clock_states[sid_b].num_samples = len(fresh)
                self.clock_states[sid_b].last_updated = time.time()
                logger.debug(
                    "Clock offset %s→%s: %.1f ns (rms=%.1f ns, n=%d)",
                    sid_a, sid_b, median_offset, rms, len(fresh)
                )

    def apply_corrections(self, station_id: str, raw_ts_ns: int) -> int:
        """Return clock-corrected timestamp for a given station."""
        if station_id in self.clock_states:
            return self.clock_states[station_id].apply(raw_ts_ns)
        return raw_ts_ns


class GPSDisciplinedSync:
    """
    Wrapper for receivers equipped with GPS-disciplined oscillators (GPSDO).
    
    GPSDO receivers report timestamps that are already UTC-aligned to ~10 ns.
    This class monitors residuals from over-determined MLAT solutions to 
    detect any systematic offset and correct for it.
    """

    HISTORY_SIZE = 100

    def __init__(self):
        # station_id → deque of offset residuals (ns)
        self._residuals: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.HISTORY_SIZE))
        self.offsets: dict[str, float] = {}   # estimated systematic offset (ns)

    def record_mlat_residual(
        self,
        station_id: str,
        observed_tdoa_ns: float,
        computed_tdoa_ns: float,
    ) -> None:
        """
        Record residual from an over-determined MLAT solution.
        Systematic residuals indicate clock bias.
        """
        residual = observed_tdoa_ns - computed_tdoa_ns
        self._residuals[station_id].append(residual)
        if len(self._residuals[station_id]) >= 10:
            self.offsets[station_id] = float(np.median(list(self._residuals[station_id])))

    def get_offset_ns(self, station_id: str) -> float:
        return self.offsets.get(station_id, 0.0)


# ---------------------------------------------------------------------------
# Utility: theoretical TDOA between two stations for an aircraft position
# ---------------------------------------------------------------------------

def theoretical_tdoa_ns(
    aircraft_ecef: np.ndarray,
    station_ref: ReceiverStation,
    station_other: ReceiverStation,
) -> float:
    """
    Compute the expected TDOA (in nanoseconds) between two stations
    for an aircraft at a known ECEF position.

    Returns t_other - t_ref  (negative if aircraft is closer to 'other').
    """
    d_ref = float(np.linalg.norm(aircraft_ecef - station_ref.ecef))
    d_other = float(np.linalg.norm(aircraft_ecef - station_other.ecef))
    return (d_other - d_ref) / C * 1e9
