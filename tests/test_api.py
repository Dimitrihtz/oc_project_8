import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from api.app import app

client = TestClient(app, raise_server_exceptions=False)

VALID_PAYLOAD = {
    "EXT_SOURCES_MEAN": 0.524,
    "CREDIT_TERM": 0.05,
    "EXT_SOURCE_3": 0.535,
    "GOODS_PRICE_CREDIT_PERCENT": 0.9,
    "INSTAL_AMT_PAYMENT_sum": 318619.5,
    "AMT_ANNUITY": 24903.0,
    "POS_CNT_INSTALMENT_FUTURE_mean": 6.95,
    "DAYS_BIRTH": -15750,
    "EXT_SOURCES_WEIGHTED": 1.5,
    "EXT_SOURCE_2": 0.566,
}


# === Health endpoint ===

def test_health_status_code():
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200


def test_health_model_loaded():
    with TestClient(app) as c:
        response = c.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# === Model loading ===

def test_model_is_loaded():
    with TestClient(app):
        assert app_module.session is not None


def test_model_has_10_features():
    with TestClient(app):
        assert app_module.session.get_inputs()[0].shape[1] == 10


def test_model_has_2_classes():
    with TestClient(app):
        assert len(app_module.session.get_outputs()) == 2


# === Valid prediction ===

def test_predict_status_code():
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200


def test_predict_has_required_fields():
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert "prediction" in data
        assert "probability_default" in data
        assert "credit_decision" in data


def test_predict_binary_output():
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert data["prediction"] in (0, 1)


def test_predict_probability_range():
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        assert 0.0 <= data["probability_default"] <= 1.0


def test_predict_decision_matches_prediction():
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_PAYLOAD)
        data = response.json()
        if data["prediction"] == 1:
            assert data["credit_decision"] == "denied"
        else:
            assert data["credit_decision"] == "approved"


# === Missing fields ===

def test_empty_body_returns_422():
    with TestClient(app) as c:
        response = c.post("/predict", json={})
        assert response.status_code == 422


def test_missing_one_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["EXT_SOURCES_MEAN"]
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_partial_payload_returns_422():
    payload = {
        "EXT_SOURCES_MEAN": 0.5,
        "CREDIT_TERM": 0.05,
    }
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


# === Out-of-range values ===

def test_negative_ext_source_mean_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["EXT_SOURCES_MEAN"] = -0.1
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_ext_source_3_above_1_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["EXT_SOURCE_3"] = 1.5
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_positive_days_birth_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["DAYS_BIRTH"] = 100
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_zero_days_birth_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["DAYS_BIRTH"] = 0
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_negative_annuity_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["AMT_ANNUITY"] = -1000.0
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_zero_annuity_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["AMT_ANNUITY"] = 0.0
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_ext_source_2_above_1_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["EXT_SOURCE_2"] = 1.1
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


# === Wrong types ===

def test_string_for_float_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["EXT_SOURCES_MEAN"] = "not_a_number"
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_string_for_int_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["DAYS_BIRTH"] = "yesterday"
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_none_value_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["AMT_ANNUITY"] = None
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_list_value_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["CREDIT_TERM"] = [0.05, 0.10]
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        assert response.status_code == 422


def test_boolean_coercion_for_float():
    payload = VALID_PAYLOAD.copy()
    payload["EXT_SOURCES_MEAN"] = True
    with TestClient(app) as c:
        response = c.post("/predict", json=payload)
        # Pydantic coerces bool to float (True -> 1.0), which is valid
        assert response.status_code == 200
