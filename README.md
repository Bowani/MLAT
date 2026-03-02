# MLAT System for Neuron / 4DSky Network

Multilateration (MLAT) engine that localises aircraft using Time Difference of Arrival (TDOA) measurements from geographically distributed Mode-S receivers on the **Neuron** peer-to-peer network, discovered via **Hedera** Hashgraph consensus.

---

## Architecture

```
Hedera HCS Topic (peer announcements)
        │
        ▼
 PeerDiscovery  ──── refreshes every 5 min
        │  receiver lat/lon/alt/ws_endpoint
        ▼
 NeuronStreamManager  ──── WebSocket fan-out to each 4DSky peer
        │  raw Mode-S frames (JSON) with nanosecond GPS timestamps
        ▼
 TDOACorrelator  ──── matches identical payloads within 50 ms window
        │  TDOAMeasurement clusters (≥4 stations per cluster)
        ▼
 MLATSolver  ──── nonlinear least-squares in ECEF space (scipy.optimize)
        │  MLATResult (lat, lon, alt, rms residual, covariance)
        ▼
 on_fix callback  ──── your handler (log, stream, store, visualise)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the offline demo (no network required)
```bash
python main.py --demo
```
This simulates 6 receivers and one aircraft, injects synthetic TDOA
measurements with ±20 ns noise, solves for position, and prints accuracy.

### 3. Live mode (requires 4DSky credentials)
```bash
export FOURDSSKY_API_TOKEN="your-token-here"
python main.py --config config/settings.json
```

---

## Key Modules

| File | Purpose |
|------|---------|
| `src/mlat_solver.py` | Core MLAT algorithm – WGS84, ECEF maths, least-squares solver |
| `src/neuron_integration.py` | Hedera peer discovery, WebSocket streams, TDOA correlator, orchestrator |
| `src/clock_sync.py` | Passive clock synchronisation using coincident Mode-S observations |
| `main.py` | Entry point, CLI, demo runner |
| `config/settings.json` | Runtime configuration |

---

## MLAT Algorithm Details

### Problem formulation
Given N receivers at known positions **r₁…rₙ**, each reporting the time
`tᵢ` at which they received the same Mode-S frame, we observe:

```
TDOA_i = tᵢ - t₁  (relative to reference receiver 1)
```

Each TDOA defines a hyperboloid of possible aircraft positions:

```
|r_aircraft - rᵢ| - |r_aircraft - r₁| = c · TDOA_i
```

Three independent TDOAs (4 receivers) define the intersection in 3D space.

### Solver
- Coordinates: WGS84 → ECEF (meters) for all geometry
- Algorithm: Levenberg-Marquardt via `scipy.optimize.least_squares`
- Initial guess: centroid of participating stations + 10 km altitude
- Warm-start: previous fix reused as initial guess when available
- Convergence: stops when position update < 0.1 m
- Rejection: solutions with RMS residual > 5 km are discarded

### Clock synchronisation
Receiver timestamps must be aligned to < 1 µs for useful MLAT accuracy.
Two methods are supported:
1. **GPS-disciplined receivers** (GPSDO): assumed UTC-aligned to ~10 ns
2. **Passive sync**: uses coincident Mode-S observations with known-position
   aircraft (ADS-B) to estimate and correct relative clock offsets

### Accuracy vs. geometry
| Geometry | Timing noise | Expected error |
|---------|------------|--------------|
| Good (>4 receivers, wide spread) | 50 ns | ~50 m |
| Moderate | 100 ns | ~200 m |
| Poor (near-coplanar) | 100 ns | >1 km |

---

## Extending

**Adding a custom position sink:**
```python
from neuron_integration import MLATOrchestrator, NeuronConfig

config = NeuronConfig(hedera_topic_id="0.0.YOUR_TOPIC")
orch = MLATOrchestrator(config)

def my_handler(result):
    print(f"{result.icao}: {result.lat:.5f}, {result.lon:.5f}, {result.alt_m:.0f}m")

orch.on_fix = my_handler
import asyncio
asyncio.run(orch.run())
```

**Using the solver standalone:**
```python
from src.mlat_solver import MLATSolver, ReceiverStation, TDOAMeasurement

stations = {
    "A": ReceiverStation("A", lat=52.3, lon=4.8, alt=5),
    # ... add more
}
solver = MLATSolver(stations)
result = solver.solve("ICAO24", measurements, epoch_utc)
```
