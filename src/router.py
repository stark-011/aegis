"""Policy-only traffic remediation router for AEGIS predictions."""

from __future__ import annotations

from ipaddress import ip_address
from typing import Mapping

import numpy as np

from src.config import (
    BENIGN_LABEL,
    GREY_ZONE_CONFIDENCE_THRESHOLD,
    XGB_CONFIDENCE_THRESHOLD,
)


# Each action is deliberately a replaceable integration seam.  A production
# deployment can exchange these print-only functions for firewall, SOAR, SDN,
# or EDR calls without changing any classification/routing code.
def apply_rate_limiting(ip: str) -> str:
    print(f"[AEGIS] ACTION=RATE_LIMIT source_ip={ip}")
    return "rate_limiting"


def isolate_node(ip: str) -> str:
    print(f"[AEGIS] ACTION=ISOLATE_NODE source_ip={ip}")
    return "isolate_node"


def block_ip(ip: str) -> str:
    print(f"[AEGIS] ACTION=BLOCK_IP source_ip={ip}")
    return "block_ip"


def deploy_decoy(ip: str) -> str:
    print(f"[AEGIS] ACTION=DEPLOY_DECOY source_ip={ip}")
    return "deploy_decoy"


def flag_for_review(ip: str, confidence: float) -> str:
    print(f"[AEGIS] ACTION=FLAG_FOR_REVIEW source_ip={ip} confidence={confidence:.2%}")
    return "flag_for_review"


def quarantine_zero_day(ip: str, mse: float) -> str:
    print(f"[AEGIS] ACTION=QUARANTINE_ZERO_DAY source_ip={ip} mse={mse:.6f}")
    return "quarantine_zero_day"


def _high_confidence_action(category: str, ip: str) -> str:
    """Map known high-confidence attack families to their least broad control."""

    if category.startswith(("DDoS", "DoS")):
        return apply_rate_limiting(ip)
    if category.startswith("Recon") or category == "VulnerabilityScan":
        return deploy_decoy(ip)
    if category.startswith("Mirai"):
        return isolate_node(ip)
    # A new non-benign class has no specific policy yet: deny safely.
    return block_ip(ip)


def route_traffic(
    prediction: Mapping[str, float | str],
    source_ip: str,
    zero_day_threshold: float,
) -> dict[str, float | str]:
    """Apply the five-tier AEGIS response policy to one model prediction.

    Known attacks get high/grey-zone treatment before the reconstruction
    fallback.  This preserves the intended model semantics: the autoencoder
    detects *unknown* attacks rather than overriding known classifier labels.
    """

    try:
        normalized_ip = str(ip_address(source_ip))
    except ValueError as exc:
        raise ValueError(f"Invalid source IP address: {source_ip!r}") from exc

    try:
        confidence = float(prediction["confidence"])
        category = str(prediction["predicted_category"])
        mse = float(prediction["reconstruction_mse"])
    except KeyError as exc:
        raise ValueError(f"Prediction is missing required field: {exc.args[0]}") from exc
    if not all(np.isfinite(value) for value in (confidence, mse, zero_day_threshold)):
        raise ValueError("Confidence, reconstruction MSE, and threshold must be finite")
    if not 0.0 <= confidence <= 1.0 or mse < 0 or zero_day_threshold < 0:
        raise ValueError("Prediction values are outside their valid ranges")

    # Tier 1: confidently recognised attack -- act according to known family.
    if category != BENIGN_LABEL and confidence >= XGB_CONFIDENCE_THRESHOLD:
        action, tier = _high_confidence_action(category, normalized_ip), "high_confidence"
    # Tier 2: plausible but uncertain known attack -- limit impact, retain access.
    elif category != BENIGN_LABEL and confidence >= GREY_ZONE_CONFIDENCE_THRESHOLD:
        action, tier = apply_rate_limiting(normalized_ip), "grey_zone"
    # Tier 3: no supervised decision handled the record; check for novel behavior.
    elif mse > zero_day_threshold:
        category, action, tier = "UNKNOWN_ZERO_DAY", quarantine_zero_day(normalized_ip, mse), "zero_day"
    # Tier 4: low-confidence non-benign classification needs a human decision.
    elif category != BENIGN_LABEL:
        action, tier = flag_for_review(normalized_ip, confidence), "low_confidence"
    # Tier 5: known normal traffic passes through.
    else:
        print(f"[AEGIS] ACTION=ALLOW source_ip={normalized_ip}")
        action, tier = "allow", "benign"

    return {
        "source_ip": normalized_ip,
        "category": category,
        "action": action,
        "tier": tier,
        "confidence": confidence,
        "reconstruction_mse": mse,
    }
