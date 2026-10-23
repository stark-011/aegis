"""Flask login gateway protected by the AEGIS detection and response pipeline."""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.engines import AegisPredictor
from src.router import route_traffic


app = Flask(__name__)
CORS(app)  # Enables cross-origin requests from the React application.

NODE_BACKEND_URL = "http://localhost:8080"

# Model loading happens once at process startup. Per-request work only calls
# the fitted predictor, which keeps login latency low.
print("[AEGIS] Initializing dual-engine security pipeline...")
predictor = AegisPredictor()
print("[AEGIS] Gateway online and monitoring port 5000.")


def get_safe_json_payload() -> dict[str, Any]:
    """Return a JSON object without allowing malformed/missing JSON to crash Flask.

    ``silent=True`` makes Flask return ``None`` instead of raising a 400/415
    HTML error when the body is absent or malformed. A JSON array/string is
    also not a valid login object, so it is treated as an empty payload.
    """

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    if payload is not None:
        print(
            "[AEGIS] LOGIN payload was valid JSON but not an object; "
            "using an empty payload."
        )
    return {}


def extract_features_from_payload(payload: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    """Map login metadata to AEGIS's 46 CICIoT2023-compatible features."""

    username = str(payload.get("username", ""))
    password = str(payload.get("password", ""))

    # Nominal benign baseline, with simple HTTP metadata proxies populated.
    features = np.zeros(46, dtype=np.float32)
    features[0] = float(len(username) + len(password))
    features[1] = 64.0

    # This gateway-only heuristic simulates a known SQL injection signature.
    sqli_tokens = ["'", "--", "/*", "*/", "or 1=1", "union select", "drop table"]
    combined_payload = f"{username} {password}".lower()
    is_malicious = any(token in combined_payload for token in sqli_tokens)
    return features, is_malicious


def normalize_prediction(prediction: Any) -> dict[str, float | str]:
    """Validate current dict predictions and safely support legacy 3-item outputs."""

    if isinstance(prediction, Mapping):
        try:
            confidence = float(prediction["confidence"])
            category = str(prediction["predicted_category"])
            mse = float(prediction["reconstruction_mse"])
        except KeyError as exc:
            raise ValueError(f"Predictor response is missing key: {exc.args[0]}") from exc
    else:
        # Previous gateway versions expected a tuple. Flatten only an exact
        # three-value result; every other shape is rejected with a clean JSON
        # error by the route-level exception boundary.
        values = np.asarray(prediction, dtype=object).reshape(-1)
        if values.size != 3:
            raise ValueError(
                "Predictor returned an incompatible shape; expected exactly "
                f"three values but received shape {np.asarray(prediction).shape}."
            )
        confidence, category, mse = float(values[0]), str(values[1]), float(values[2])

    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Predictor returned invalid confidence: {confidence!r}")
    if not np.isfinite(mse) or mse < 0.0:
        raise ValueError(f"Predictor returned invalid reconstruction MSE: {mse!r}")
    if not category:
        raise ValueError("Predictor returned an empty category")
    return {
        "confidence": confidence,
        "predicted_category": category,
        "reconstruction_mse": mse,
    }


@app.route("/api/login", methods=["POST"])
def secure_login():
    """Inspect a login attempt, route a threat, then proxy allowed traffic."""

    # This boundary ensures every unexpected route failure is JSON, not Flask's
    # default HTML traceback, so the React client always receives parseable data.
    try:
        request_started = perf_counter()
        client_ip = request.remote_addr or "127.0.0.1"
        payload = get_safe_json_payload()
        features, is_malicious_payload = extract_features_from_payload(payload)

        if is_malicious_payload:
            # Known SQL injection signatures do not need an ML round trip.
            prediction = {
                "predicted_category": "SqlInjection",
                "confidence": 0.98,
                "reconstruction_mse": 0.42,
            }
            print(f"[AEGIS] LOGIN client_ip={client_ip} source=sql_injection_heuristic")
        else:
            raw_prediction = predictor.predict(features.reshape(1, -1))
            prediction = normalize_prediction(raw_prediction)
            print(
                f"[AEGIS] LOGIN client_ip={client_ip} source=model "
                f"category={prediction['predicted_category']} "
                f"confidence={prediction['confidence']:.2%}"
            )

        # src.router.route_traffic accepts the prediction mapping, source IP,
        # and the fitted autoencoder threshold in this exact positional order.
        routing_result = route_traffic(
            prediction,
            client_ip,
            predictor.zero_day_threshold,
        )
        action = str(routing_result.get("action", "allow"))
        category = str(routing_result.get("category", prediction["predicted_category"]))
        latency_ms = (perf_counter() - request_started) * 1000.0
        print(
            f"[AEGIS] LOGIN decision client_ip={client_ip} category={category} "
            f"tier={routing_result.get('tier', 'unknown')} action={action} "
            f"latency_ms={latency_ms:.3f}"
        )

        # Controls that deny, throttle, quarantine, or isolate must not be
        # proxied to the banking backend.
        blocked_actions = {
            "block_ip",
            "rate_limiting",
            "quarantine_zero_day",
            "isolate_node",
        }
        if action in blocked_actions:
            return jsonify({
                "error": "Access Denied by AEGIS Autonomous Defense System",
                "threat_detected": category,
                "action_taken": action,
                "latency_ms": round(latency_ms, 3),
            }), 403

        try:
            response = requests.post(
                f"{NODE_BACKEND_URL}/api/login",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=3.0,
            )
            # Relay the backend response unchanged, including its HTTP status.
            return response.content, response.status_code, response.headers.items()
        except requests.exceptions.ConnectionError:
            print(f"[AEGIS] LOGIN backend_unreachable url={NODE_BACKEND_URL}")
            return jsonify({"error": "Core Banking Server (Port 8080) is unreachable."}), 502
        except requests.exceptions.Timeout:
            print(f"[AEGIS] LOGIN backend_timeout url={NODE_BACKEND_URL}")
            return jsonify({"error": "Core Banking Server (Port 8080) timed out."}), 504

    except Exception as e:
        # print_exc() writes the full stack trace to the Flask process terminal,
        # while the browser receives only a compact, JSON-safe error response.
        print("[AEGIS] LOGIN unhandled_gateway_error")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
