#!/usr/bin/env python3
"""
MLAT System – Main Entry Point

Usage:
    python main.py [--config config/settings.json] [--demo]

    --config   Path to JSON configuration file
    --demo     Run offline simulation with synthetic data (no network needed)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mlat.main")

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))
from mlat_solver import MLATSolver, ReceiverStation, TDOAMeasurement, MLATResult
from neuron_integration import MLATOrchestrator, NeuronConfig
from clock_sync import PassiveClockSync, theoretical_tdoa_ns


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> NeuronConfig:
    with open(path) as f:
        raw = json.load(f)
    cfg = NeuronConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # Allow token override via environment variable
    token = os.environ.get("FOURDSSKY_API_TOKEN", "")
    if token:
        cfg.fourdskyapi_token = token
    return cfg


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

async def run_live(config: NeuronConfig) -> None:
    """Connect to the Neuron network and run MLAT in real-time."""

    def on_fix(result: MLATResult) -> None:
        status = "✓" if result.converged else "~"
        print(
            f"[{status}] {result.icao:6s}  "
            f"lat={result.lat:9.5f}  lon={result.lon:10.5f}  "
            f"alt={result.alt_m:6.0f}m  "
            f"rms={result.residual_rms_m:5.0f}m  "
            f"n={result.num_stations}"
        )

    orch = MLATOrchestrator(config)
    orch.on_fix = on_fix

    logger.info("Starting live MLAT on Neuron network (topic %s)", config.hedera_topic_id)
    print("\nAircraft fixes will appear below:")
    print("-" * 75)
    await orch.run()


# ---------------------------------------------------------------------------
# Demo / simulation mode
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Offline demonstration using synthetic receivers and a simulated aircraft.
    Validates that the solver correctly recovers the injected aircraft position.
    """
    import numpy as np
    from mlat_solver import geodetic_to_ecef, ecef_to_geodetic, C

    print("\n" + "=" * 60)
    print("  MLAT Demo – Offline Simulation")
    print("=" * 60)

    # ---- Define 6 fictional receivers around Amsterdam ----
    stations = {
        "RECV-A": ReceiverStation("RECV-A",  52.300,  4.760, 5),
        "RECV-B": ReceiverStation("RECV-B",  52.380,  5.100, 12),
        "RECV-C": ReceiverStation("RECV-C",  52.150,  4.950, 3),
        "RECV-D": ReceiverStation("RECV-D",  52.450,  4.600, 8),
        "RECV-E": ReceiverStation("RECV-E",  52.250,  5.200, 15),
        "RECV-F": ReceiverStation("RECV-F",  52.500,  4.900, 6),
    }

    solver = MLATSolver(stations)
    clock_sync = PassiveClockSync(stations)

    # ---- Inject a "true" aircraft position ----
    TRUE_LAT, TRUE_LON, TRUE_ALT = 52.32, 4.88, 9500   # 9500 m ≈ FL310
    true_ecef = np.array(geodetic_to_ecef(TRUE_LAT, TRUE_LON, TRUE_ALT))
    print(f"\nTrue aircraft position:  {TRUE_LAT:.4f}°N  {TRUE_LON:.4f}°E  {TRUE_ALT} m")

    # ---- Simulate TDOA measurements with ±20 ns receiver noise ----
    np.random.seed(42)
    NOISE_NS = 20.0
    ref_id = "RECV-A"
    ref_ecef = stations[ref_id].ecef
    d_ref = float(np.linalg.norm(true_ecef - ref_ecef))

    measurements = []
    print(f"\nSimulated TDOAs (ref={ref_id}, noise=±{NOISE_NS:.0f} ns):")
    for sid, st in stations.items():
        if sid == ref_id:
            continue
        d_i = float(np.linalg.norm(true_ecef - st.ecef))
        true_tdoa_ns = (d_i - d_ref) / C * 1e9
        noisy_tdoa_ns = true_tdoa_ns + np.random.normal(0, NOISE_NS)
        print(f"  {sid}: true={true_tdoa_ns:+8.1f} ns  noisy={noisy_tdoa_ns:+8.1f} ns")
        measurements.append(TDOAMeasurement(
            icao="TEST01",
            ref_station_id=ref_id,
            other_station_id=sid,
            tdoa_ns=noisy_tdoa_ns,
            timestamp_utc=time.time(),
        ))

    # ---- Run solver ----
    print("\nRunning MLAT solver…")
    result = solver.solve("TEST01", measurements, time.time())

    if result is None:
        print("ERROR: Solver returned no result.")
        return

    print(f"\n--- MLAT Result ---")
    print(f"  Converged:   {result.converged}")
    print(f"  Latitude:    {result.lat:.6f}°  (error: {(result.lat-TRUE_LAT)*111_111:.1f} m)")
    print(f"  Longitude:   {result.lon:.6f}°  (error: {(result.lon-TRUE_LON)*111_111*np.cos(np.radians(TRUE_LAT)):.1f} m)")
    print(f"  Altitude:    {result.alt_m:.1f} m  (error: {result.alt_m-TRUE_ALT:.1f} m)")
    print(f"  RMS residual:{result.residual_rms_m:.2f} m")
    print(f"  Stations:    {result.num_stations}")

    horiz_err = np.sqrt(
        ((result.lat - TRUE_LAT) * 111_111) ** 2 +
        ((result.lon - TRUE_LON) * 111_111 * np.cos(np.radians(TRUE_LAT))) ** 2
    )
    vert_err = abs(result.alt_m - TRUE_ALT)
    print(f"\n  Horizontal error: {horiz_err:.1f} m")
    print(f"  Vertical error:   {vert_err:.1f} m")

    if horiz_err < 500 and vert_err < 1000:
        print("\n  ✓ Solution within expected accuracy bounds.")
    else:
        print("\n  ✗ Solution outside expected bounds – check receiver geometry or noise level.")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLAT System for Neuron / 4DSky")
    parser.add_argument(
        "--config",
        default="config/settings.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run offline simulation (no network required)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if args.demo:
        run_demo()
        return

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        print("Run with --demo for an offline test, or create config/settings.json")
        sys.exit(1)

    config = load_config(args.config)
    asyncio.run(run_live(config))


if __name__ == "__main__":
    main()
