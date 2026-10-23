"""AEGIS SOC simulator API.

This service deliberately uses synthetic traffic profiles for the dashboard.
Those profiles are converted to the same prediction contract consumed by the
existing AEGIS policy router, keeping response decisions identical to a live
pipeline while avoiding model loading for a UI demo.
"""

from __future__ import annotations

import random
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv


# The established AEGIS code lives one directory above this API package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(Path(__file__).resolve().parent / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router import route_traffic  # noqa: E402


app = Flask(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

ZERO_DAY_THRESHOLD = 1.0
MAX_LOG_EVENTS = 20
EVENTS: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_EVENTS)
EVENTS_LOCK = Lock()

# The values stand in for packet/flow characteristics a collector would
# provide.  The prediction fields are fed unchanged into AEGIS route_traffic.
THREAT_PROFILES: dict[str, dict[str, Any]] = {
    "benign": {
        "category": "BenignTraffic",
        "confidence": 0.99,
        "reconstruction_mse": 0.20,
        "features": {
            "flow_duration_ms": 142,
            "packets_per_second": 18,
            "syn_rate": 0.01,
            "payload_bytes": 768,
        },
    },
    "ddos": {
        "category": "DDoS-SYN_Flood",
        "confidence": 0.99,
        "reconstruction_mse": 0.45,
        "features": {
            "flow_duration_ms": 3,
            "packets_per_second": 85_000,
            "syn_rate": 0.98,
            "payload_bytes": 64,
        },
    },
    "sqli": {
        "category": "SqlInjection",
        "confidence": 0.98,
        "reconstruction_mse": 0.38,
        "features": {
            "flow_duration_ms": 24,
            "packets_per_second": 9,
            "syn_rate": 0.02,
            "payload_bytes": 1_640,
        },
    },
    "xss": {
        "category": "Browser-XSS",
        "confidence": 0.96,
        "reconstruction_mse": 0.40,
        "features": {
            "flow_duration_ms": 18,
            "packets_per_second": 12,
            "syn_rate": 0.01,
            "payload_bytes": 2_048,
        },
    },
    "portscan": {
        "category": "Recon-PortScan",
        "confidence": 0.95,
        "reconstruction_mse": 0.35,
        "features": {
            "flow_duration_ms": 90,
            "packets_per_second": 1_500,
            "syn_rate": 0.73,
            "payload_bytes": 52,
        },
    },
    "malware": {
        "category": "Mirai-greeth_flood",
        "confidence": 0.97,
        "reconstruction_mse": 0.50,
        "features": {
            "flow_duration_ms": 11,
            "packets_per_second": 24_000,
            "syn_rate": 0.88,
            "payload_bytes": 512,
        },
    },
    "zeroday": {
        "category": "BenignTraffic",
        "confidence": 0.95,
        "reconstruction_mse": 48.5,
        "features": {
            "flow_duration_ms": 7,
            "packets_per_second": 11_200,
            "syn_rate": 0.41,
            "payload_bytes": 9_217,
        },
    },
    "grey_zone": {
        "category": "DDoS-UDP_Flood",
        "confidence": 0.62,
        "reconstruction_mse": 0.42,
        "features": {
            "flow_duration_ms": 36,
            "packets_per_second": 6_400,
            "syn_rate": 0.38,
            "payload_bytes": 320,
        },
    },
}


def synthetic_source_ip() -> str:
    """Return a documentation-only IPv4 address for a simulation event."""

    return f"198.51.100.{random.randint(10, 240)}"


def create_event(threat_type: str) -> dict[str, Any]:
    """Route one synthetic profile through the real AEGIS response policy."""

    profile = THREAT_PROFILES[threat_type]
    source_ip = synthetic_source_ip()
    prediction = {
        "predicted_category": profile["category"],
        "confidence": profile["confidence"],
        "reconstruction_mse": profile["reconstruction_mse"],
    }
    started = perf_counter()
    routing = route_traffic(prediction, source_ip, ZERO_DAY_THRESHOLD)
    latency_ms = round((perf_counter() - started) * 1000, 3)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ip": routing["source_ip"],
        "threat_type": threat_type,
        "category": routing["category"],
        "action_taken": routing["action"],
        "confidence": round(float(routing["confidence"]), 4),
        "latency_ms": latency_ms,
        "tier": routing["tier"],
        "synthetic_features": profile["features"],
    }
    with EVENTS_LOCK:
        EVENTS.appendleft(event)
    return event


@app.post("/api/simulate")
def simulate() -> tuple[Any, int]:
    """Run an AEGIS simulation; mitigation responses retain a 403 status."""

    payload = request.get_json(silent=True) or {}
    threat_type = str(payload.get("threat_type", "")).strip().lower()
    if threat_type not in THREAT_PROFILES:
        return jsonify({
            "error": "Invalid threat_type.",
            "supported_threat_types": list(THREAT_PROFILES),
        }), 400

    event = create_event(threat_type)
    blocked = event["action_taken"] != "allow"
    response = {
        "status": "403 Blocked" if blocked else "200 Allowed",
        "message": "AEGIS mitigation applied" if blocked else "AEGIS allowed normal traffic",
        **event,
    }
    return jsonify(response), 403 if blocked else 200


@app.get("/api/logs")
def logs() -> Any:
    """Return the latest twenty simulation events, newest first."""

    with EVENTS_LOCK:
        return jsonify(list(EVENTS))


@app.get("/api/health")
def health() -> Any:
    return jsonify({"service": "AEGIS SOC simulator", "status": "online"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
