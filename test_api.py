"""
Test script for the Wind Power Prediction API.
Tests all endpoints to verify API functionality.
"""

import requests
import json
import sys

# API base URL - using 127.0.0.1 to avoid Windows DNS issues
BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "=" * 50)
    print("TEST 1: Health Check (GET /)")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("   [PASS] Health check passed!")
    except Exception as e:
        print(f"   [FAIL] Health check failed: {e}")
        raise


def test_get_features():
    """Test the features list endpoint."""
    print("\n" + "=" * 50)
    print("TEST 2: Get Required Features (GET /features)")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/features")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Number of features: {data['num_features']}")
    # print(f"First 5 features: {data['features'][:5]}")

    assert response.status_code == 200
    assert data["num_features"] == 75
    print("   [PASS] Features endpoint passed!")

    return data["features"]


def test_model_info():
    """Test the model info endpoint."""
    print("\n" + "=" * 50)
    print("TEST 3: Get Model Info (GET /info)")
    print("=" * 50)

    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("   [PASS] Model info endpoint passed!")


def test_prediction_missing_features():
    """Test prediction with missing features (should fail)."""
    print("\n" + "=" * 50)
    print("TEST 4: Prediction with Missing Features (should fail)")
    print("=" * 50)

    # Incomplete feature set
    request_data = {
        "features": {
            "TEMPERATURE": 15.2,
            "WINDSPEED": 8.5,
            # Missing 73 other features!
        }
    }

    response = requests.post(f"{BASE_URL}/predict", json=request_data)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 400  # Bad request
    print("   [PASS] Correctly rejected incomplete features!")


def test_prediction_success(feature_names):
    """Test successful prediction with all features."""
    print("\n" + "=" * 50)
    print("TEST 5: Successful Prediction")
    print("=" * 50)

    # Create dummy feature values (all set to 0.0 for simplicity)
    # In production, you'd use real sensor data
    features_dict = {name: 0.0 for name in feature_names}

    # Set some realistic values for key features
    features_dict.update(
        {"TEMPERATURE": 15.2, "WINDSPEED": 8.5, "PRESSURE": 1013.2, "HUMIDITY": 65.0}
    )

    request_data = {"features": features_dict}

    response = requests.post(f"{BASE_URL}/predict", json=request_data)

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert "prediction_mw" in result
    assert isinstance(result["prediction_mw"], (int, float))

    print(f"\n   [PASS] Prediction successful!")
    print(f"   Predicted Power: {result['prediction_mw']:.2f} MW")
    print(f"   Timestamp: {result['timestamp']}")


def run_all_tests():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("WIND POWER PREDICTION API - TEST SUITE")
    print("=" * 60)
    print(f"Target API: {BASE_URL}")

    try:
        # Test 1: Health check
        test_health_check()

        # Test 2: Get features list
        feature_names = test_get_features()

        # Test 3: Model info
        test_model_info()

        # Test 4: Prediction with missing features
        test_prediction_missing_features()

        # Test 5: Successful prediction
        test_prediction_success(feature_names)

        print("\n" + "=" * 60)
        print("   [PASS] ALL TESTS PASSED!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n   [ERROR] Cannot connect to API!")
        print(f"   Make sure the API is running: python app_api.py")
        print(f"   Or: uvicorn app_api:app --reload")

    except AssertionError as e:
        print(f"\n   [FAIL] TEST FAILED: {e}")

    except Exception as e:
        print(f"\n   [ERROR] UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    # Force utf-8 output for console if possible, though we removed emojis anyway
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_all_tests()
