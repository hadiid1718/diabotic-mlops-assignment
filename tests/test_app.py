from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_diabetic_payload():
    payload = {"age":65, "urea":7.5, "cr":52.0, "hba1c":11.2, "chol":6.1, "tg":2.8, "hdl":0.9, "ldl":3.5, "vldl":1.2, "bmi":32.5, "gender":"M"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    j = r.json()
    assert 'prediction' in j
    assert 'label' in j

def test_predict_non_diabetic_payload():
    payload = {"age":28, "urea":4.2, "cr":48.0, "hba1c":5.1, "chol":4.0, "tg":1.2, "hdl":1.8, "ldl":2.1, "vldl":0.6, "bmi":22.0, "gender":"F"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200

def test_predict_invalid_gender_returns_422():
    payload = {"age":45, "urea":5.0, "cr":50.0, "hba1c":6.0, "chol":5.0, "tg":1.5, "hdl":1.2, "ldl":2.5, "vldl":0.8, "bmi":25.0, "gender":"X"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 422

def test_missing_fields_returns_422():
    payload = {"age":50, "urea":5.0, "cr":50.0}
    r = client.post('/predict', json=payload)
    assert r.status_code == 422
