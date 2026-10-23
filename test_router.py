"""Isolated decision-policy tests for the five AEGIS routing tiers."""

import unittest
from unittest.mock import patch

from src.router import route_traffic


SOURCE_IP = "192.0.2.10"
# A deliberately small known threshold lets 0.5 represent normal behaviour
# while 50.0 reliably represents a zero-day reconstruction anomaly.
ZERO_DAY_THRESHOLD = 1.0


class TestAegisRouter(unittest.TestCase):
    """Inject model outputs directly, without requiring packets or model files."""

    def test_high_confidence_attack_is_rate_limited(self) -> None:
        prediction = {
            "predicted_category": "DDoS-SYN_Flood",
            "confidence": 0.99,
            "reconstruction_mse": 0.5,
        }
        with patch("src.router.apply_rate_limiting", return_value="rate_limiting") as action:
            result = route_traffic(prediction, SOURCE_IP, ZERO_DAY_THRESHOLD)

        self.assertEqual(result["action"], "rate_limiting")
        action.assert_called_once_with(SOURCE_IP)
        print("Passed: High Confidence Attack -> rate_limiting")

    def test_grey_zone_attack_is_rate_limited(self) -> None:
        prediction = {
            "predicted_category": "Mirai-Botnet",
            "confidence": 0.65,
            "reconstruction_mse": 0.5,
        }
        with patch("src.router.apply_rate_limiting", return_value="rate_limiting") as action:
            result = route_traffic(prediction, SOURCE_IP, ZERO_DAY_THRESHOLD)

        self.assertEqual(result["action"], "rate_limiting")
        action.assert_called_once_with(SOURCE_IP)
        print("Passed: Grey Zone Attack -> rate_limiting")

    def test_zero_day_anomaly_is_quarantined(self) -> None:
        prediction = {
            "predicted_category": "BenignTraffic",
            "confidence": 0.99,
            "reconstruction_mse": 50.0,
        }
        with patch(
            "src.router.quarantine_zero_day", return_value="quarantine_zero_day"
        ) as action:
            result = route_traffic(prediction, SOURCE_IP, ZERO_DAY_THRESHOLD)

        self.assertEqual(result["category"], "UNKNOWN_ZERO_DAY")
        self.assertEqual(result["action"], "quarantine_zero_day")
        action.assert_called_once_with(SOURCE_IP, 50.0)
        print("Passed: Zero-Day Anomaly -> quarantine_zero_day")

    def test_low_confidence_attack_is_flagged_for_review(self) -> None:
        prediction = {
            "predicted_category": "SqlInjection",
            "confidence": 0.45,
            "reconstruction_mse": 0.5,
        }
        with patch("src.router.flag_for_review", return_value="flag_for_review") as action:
            result = route_traffic(prediction, SOURCE_IP, ZERO_DAY_THRESHOLD)

        self.assertEqual(result["action"], "flag_for_review")
        action.assert_called_once_with(SOURCE_IP, 0.45)
        print("Passed: Low Confidence Attack -> flag_for_review")

    def test_benign_traffic_is_allowed(self) -> None:
        prediction = {
            "predicted_category": "BenignTraffic",
            "confidence": 0.99,
            "reconstruction_mse": 0.5,
        }
        with patch("builtins.print"):
            result = route_traffic(prediction, SOURCE_IP, ZERO_DAY_THRESHOLD)

        self.assertEqual(result["action"], "allow")
        print("Passed: Benign Traffic -> allow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
