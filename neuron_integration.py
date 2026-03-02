"""
4DSky Neuron Network Integration
Connects to the Hedera-based peer discovery network, subscribes to Mode-S
data streams from distributed receivers, and emits correlated TDOA clusters
ready for the MLAT solver.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  Hedera Mirror Node / Hashgraph Consensus Service   │
    │  Topic: NEURON_PEER_DISCOVERY_TOPIC_ID              │
    └────────────────────┬────────────────────────────────┘
                         │ peer announcements
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  PeerDiscovery  – discovers active 4DSky receivers  │
    └────────────────────┬────────────────────────────────┘
                         │ receiver metadata (lat/lon/alt/endpoint)
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  NeuronStreamManager – WebSocket fan-out to peers   │
    │  subscribes to Mode-S raw message streams           │
    └────────────────────┬────────────────────────────────┘
                         │ raw Mode-S frames w/ nanosecond timestamps
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  TDOACorrelator – time-windows messages by ICAO,    │
    │  forms TDOA pairs relative to reference station     │
    └────────────────────┬────────────────────────────────┘
                         │ TDOAMeasurement clusters
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  MLATSolver  (mlat_solver.py)                       │
    └─────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from mlat_solver import MLATSolver, ReceiverStation, TDOAMeasurement, MLATResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NeuronConfig:
    """Runtime configuration for the Neuron / 4DSky integration."""

    # Hedera topic containing peer announcements (Base64 topic ID string)
    hedera_topic_id: str = "0.0.1234567"

    # Hedera Mirror Node REST endpoint
    hedera_mirror_url: str = "https://mainnet-public.mirrornode.hedera.com"

    # 4DSky authentication token (set via env var FOURDSSKY_API_TOKEN)
    fourdskyapi_token: str = ""

    # Maximum time window to correlate messages from different receivers (seconds)
    # Mode-S messages must arrive within this window to form a TDOA cluster
    correlation_window_s: float = 0.050   # 50 ms

    # Minimum receivers per TDOA cluster to attempt MLAT
    min_receivers: int = 4

    # Maximum clock drift tolerance between receivers (nanoseconds)
    max_clock_drift_ns: float = 1_000.0   # 1 µs

    # How often to re-query Hedera for new peers (seconds)
    peer_refresh_interval_s: float = 300.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PeerInfo:
    """Metadata for a discovered Neuron / 4DSky peer."""
    peer_id: str
    station_id: str
    lat: float
    lon: float
    alt_m: float
    ws_endpoint: str          # WebSocket URL for raw stream
    last_seen_utc: float = field(default_factory=time.time)
    clock_offset_ns: float = 0.0   # estimated offset vs. UTC


@dataclass
class RawModeS:
    """A single Mode-S message received by one station."""
    station_id: str
    icao: str              # 24-bit hex ICAO address (e.g. "4CA8B5")
    df: int                # Downlink Format (0, 4, 5, 11, 17, 20, 21 …)
    payload_hex: str       # raw message bytes as hex string
    rx_timestamp_ns: int   # nanosecond UTC timestamp at receiver (GPS-disciplined)
    signal_level_dbm: float = -70.0


# ---------------------------------------------------------------------------
# Hedera peer discovery
# ---------------------------------------------------------------------------

class PeerDiscovery:
    """
    Polls the Hedera Mirror Node for peer announcements on the designated
    consensus topic.  Each message on the topic is a JSON object:

        {
          "peer_id":    "neuron-station-42",
          "station_id": "EHAM-01",
          "lat":        52.3086,
          "lon":        4.7639,
          "alt_m":      -3.5,
          "ws_endpoint":"wss://neuron-eham-01.4dsky.net/stream",
          "pubkey":     "..."   // Ed25519, used for message authentication
        }
    """

    def __init__(self, config: NeuronConfig):
        self.config = config
        self._peers: dict[str, PeerInfo] = {}
        self._last_sequence: int = 0

    async def refresh(self) -> dict[str, PeerInfo]:
        """Fetch recent peer announcements from the Hedera mirror node."""
        import aiohttp  # imported lazily to keep optional dependency clear
        url = (
            f"{self.config.hedera_mirror_url}/api/v1/topics/"
            f"{self.config.hedera_topic_id}/messages"
            f"?limit=100&sequencenumber=gt:{self._last_sequence}&order=asc"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Hedera mirror returned HTTP %d", resp.status)
                        return self._peers
                    data = await resp.json()
        except Exception as exc:
            logger.error("Hedera peer refresh failed: %s", exc)
            return self._peers

        for msg in data.get("messages", []):
            try:
                import base64
                payload = json.loads(base64.b64decode(msg["message"]).decode())
                peer = PeerInfo(
                    peer_id=payload["peer_id"],
                    station_id=payload["station_id"],
                    lat=float(payload["lat"]),
                    lon=float(payload["lon"]),
                    alt_m=float(payload["alt_m"]),
                    ws_endpoint=payload["ws_endpoint"],
                )
                self._peers[peer.station_id] = peer
                self._last_sequence = max(self._last_sequence, int(msg["sequence_number"]))
                logger.info("Discovered peer: %s @ %.4f, %.4f", peer.station_id, peer.lat, peer.lon)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.debug("Skipping malformed peer announcement: %s", exc)

        return self._peers

    @property
    def peers(self) -> dict[str, PeerInfo]:
        return self._peers


# ---------------------------------------------------------------------------
# WebSocket stream manager
# ---------------------------------------------------------------------------

class NeuronStreamManager:
    """
    Opens a WebSocket connection to each discovered 4DSky peer and
    subscribes to its raw Mode-S stream. Dispatches decoded RawModeS
    objects to registered callbacks.
    """

    RECONNECT_DELAY_S = 5.0

    def __init__(self, config: NeuronConfig, on_message: Callable[[RawModeS], None]):
        self.config = config
        self.on_message = on_message
        self._tasks: dict[str, asyncio.Task] = {}

    async def update_peers(self, peers: dict[str, PeerInfo]) -> None:
        """Add connections for new peers; remove connections for departed peers."""
        current_ids = set(peers.keys())
        connected_ids = set(self._tasks.keys())

        for sid in current_ids - connected_ids:
            self._tasks[sid] = asyncio.create_task(
                self._connect_peer(peers[sid]), name=f"stream-{sid}"
            )

        for sid in connected_ids - current_ids:
            self._tasks.pop(sid).cancel()

    async def _connect_peer(self, peer: PeerInfo) -> None:
        """Persistent WebSocket connection to one peer with auto-reconnect."""
        import websockets  # optional dependency
        while True:
            try:
                headers = {}
                if self.config.fourdskyapi_token:
                    headers["Authorization"] = f"Bearer {self.config.fourdskyapi_token}"
                async with websockets.connect(
                    peer.ws_endpoint, extra_headers=headers, ping_interval=30
                ) as ws:
                    logger.info("Connected to %s (%s)", peer.station_id, peer.ws_endpoint)
                    await ws.send(json.dumps({"subscribe": "modes_raw"}))
                    async for raw in ws:
                        try:
                            self._parse_and_dispatch(peer.station_id, raw)
                        except Exception as exc:
                            logger.debug("Parse error from %s: %s", peer.station_id, exc)
            except Exception as exc:
                logger.warning("Stream %s disconnected: %s – retrying in %.0fs",
                               peer.station_id, exc, self.RECONNECT_DELAY_S)
                await asyncio.sleep(self.RECONNECT_DELAY_S)

    def _parse_and_dispatch(self, station_id: str, raw: str) -> None:
        """
        Parse a JSON frame from the 4DSky stream protocol.

        Expected wire format (JSON):
        {
          "icao":     "4CA8B5",
          "df":       17,
          "msg":      "8D4CA8B5...",  // full message hex
          "ts_ns":    1713000000123456789,
          "sig_dbm":  -68.2
        }
        """
        frame = json.loads(raw)
        msg = RawModeS(
            station_id=station_id,
            icao=frame["icao"].upper(),
            df=int(frame["df"]),
            payload_hex=frame["msg"].upper(),
            rx_timestamp_ns=int(frame["ts_ns"]),
            signal_level_dbm=float(frame.get("sig_dbm", -70.0)),
        )
        self.on_message(msg)


# ---------------------------------------------------------------------------
# TDOA correlator
# ---------------------------------------------------------------------------

class TDOACorrelator:
    """
    Correlates Mode-S messages received by multiple stations for the same
    aircraft and computes Time Difference of Arrival (TDOA) values.

    A 'cluster' is a set of messages (one per station) that all correspond
    to the same Mode-S transmission from the same aircraft. Messages are
    matched when:
      - Same ICAO address
      - Same DF (downlink format)
      - Identical payload bytes  ← guarantees same transmission
      - Timestamps within `correlation_window_s`
    """

    def __init__(self, config: NeuronConfig, on_cluster: Callable[[str, list[TDOAMeasurement], float], None]):
        self.config = config
        self.on_cluster = on_cluster
        # Buffer: icao → list of (RawModeS, corrected_timestamp_ns)
        self._buffer: dict[str, list[tuple[RawModeS, int]]] = defaultdict(list)
        self._stations: dict[str, ReceiverStation] = {}

    def set_stations(self, stations: dict[str, ReceiverStation]) -> None:
        self._stations = stations

    def ingest(self, msg: RawModeS) -> None:
        """Accept a raw Mode-S message and try to form TDOA clusters."""
        # Apply estimated clock offset correction
        corrected_ns = msg.rx_timestamp_ns  # offset applied externally via calibration
        self._buffer[msg.icao].append((msg, corrected_ns))
        self._try_cluster(msg.icao)
        self._purge_old(msg.icao)

    def _try_cluster(self, icao: str) -> None:
        """Check if any group of buffered messages forms a valid TDOA cluster."""
        entries = self._buffer[icao]
        if len(entries) < self.config.min_receivers:
            return

        window_ns = int(self.config.correlation_window_s * 1e9)

        # Sort by timestamp and look for groups matching by payload within window
        entries.sort(key=lambda e: e[1])

        # Group messages with identical payload bytes (same transmission)
        payload_groups: dict[str, list[tuple[RawModeS, int]]] = defaultdict(list)
        for msg, ts in entries:
            payload_groups[msg.payload_hex].append((msg, ts))

        for payload_hex, group in payload_groups.items():
            if len(group) < self.config.min_receivers:
                continue
            ts_values = [ts for _, ts in group]
            if max(ts_values) - min(ts_values) > window_ns:
                continue  # too spread out in time

            # Check we have distinct stations
            station_ids = [m.station_id for m, _ in group]
            if len(set(station_ids)) < self.config.min_receivers:
                continue

            # Pick reference = station with earliest timestamp (strongest signal heuristic)
            ref_idx = int(np.argmin(ts_values)) if len(group) > 0 else 0
            ref_msg, ref_ts = group[ref_idx]

            measurements = []
            for i, (msg, ts) in enumerate(group):
                if i == ref_idx:
                    continue
                tdoa_ns = float(ts - ref_ts)
                if abs(tdoa_ns) > window_ns:
                    continue
                measurements.append(TDOAMeasurement(
                    icao=icao,
                    ref_station_id=ref_msg.station_id,
                    other_station_id=msg.station_id,
                    tdoa_ns=tdoa_ns,
                    timestamp_utc=ref_ts * 1e-9,
                    mlat_quality=min(1.0, 10 ** ((msg.signal_level_dbm + 60) / 20)),
                ))

            if len(measurements) >= self.config.min_receivers - 1:
                epoch_utc = ref_ts * 1e-9
                self.on_cluster(icao, measurements, epoch_utc)
                # Remove clustered messages from buffer
                used_payloads = {payload_hex}
                self._buffer[icao] = [
                    e for e in self._buffer[icao]
                    if e[0].payload_hex not in used_payloads
                ]

    def _purge_old(self, icao: str) -> None:
        """Remove messages older than twice the correlation window."""
        cutoff_ns = int((time.time() - 2 * self.config.correlation_window_s) * 1e9)
        self._buffer[icao] = [
            e for e in self._buffer[icao] if e[1] >= cutoff_ns
        ]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class MLATOrchestrator:
    """
    Ties together peer discovery, stream management, TDOA correlation, and
    the MLAT solver into a single runnable service.

    Usage:
        config = NeuronConfig(hedera_topic_id="0.0.1234567", ...)
        orch = MLATOrchestrator(config)
        orch.on_fix = lambda result: print(result)
        asyncio.run(orch.run())
    """

    def __init__(self, config: NeuronConfig):
        self.config = config
        self.on_fix: Optional[Callable[[MLATResult], None]] = None

        self._discovery = PeerDiscovery(config)
        self._stream_mgr = NeuronStreamManager(config, on_message=self._on_raw_message)
        self._correlator = TDOACorrelator(config, on_cluster=self._on_tdoa_cluster)
        self._solver: Optional[MLATSolver] = None
        self._stations: dict[str, ReceiverStation] = {}

        # Track last position per aircraft for warm-starting the solver
        self._last_ecef: dict[str, np.ndarray] = {}

    async def run(self) -> None:
        """Main async event loop. Runs indefinitely."""
        logger.info("MLAT Orchestrator starting up")
        while True:
            await self._refresh_peers()
            await asyncio.sleep(self.config.peer_refresh_interval_s)

    async def _refresh_peers(self) -> None:
        peers = await self._discovery.refresh()
        # Build ReceiverStation objects for the solver
        self._stations = {
            sid: ReceiverStation(
                station_id=sid,
                lat=p.lat,
                lon=p.lon,
                alt=p.alt_m,
            )
            for sid, p in peers.items()
        }
        self._solver = MLATSolver(self._stations)
        self._correlator.set_stations(self._stations)
        await self._stream_mgr.update_peers(peers)
        logger.info("Active stations: %d", len(self._stations))

    def _on_raw_message(self, msg: RawModeS) -> None:
        """Callback from stream manager for each received Mode-S message."""
        self._correlator.ingest(msg)

    def _on_tdoa_cluster(self, icao: str, measurements: list[TDOAMeasurement], epoch_utc: float) -> None:
        """Callback from correlator when a cluster is ready."""
        if self._solver is None:
            return
        initial = self._last_ecef.get(icao)
        result = self._solver.solve(icao, measurements, epoch_utc, initial_ecef=initial)
        if result and result.converged:
            import numpy as np
            self._last_ecef[icao] = np.array([0.0, 0.0, 0.0])  # placeholder; real ECEF from result
            logger.info(
                "FIX %s  %.4f, %.4f  %.0f m  rms=%.1f m  stations=%d",
                icao, result.lat, result.lon, result.alt_m,
                result.residual_rms_m, result.num_stations
            )
            if self.on_fix:
                self.on_fix(result)


# ---------------------------------------------------------------------------
# Import guard for numpy (used in TDOACorrelator._try_cluster)
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required: pip install numpy")
