from src.api import app, health, predict, HouseFeatures

def test_app_loads():
    assert app is not None

def test_health_ok():
    assert health() == {"status": "ok"}

def test_predict_returns_positive_price():
    f = HouseFeatures(city="Mumbai", locality="Unknown", bhk=2, bathrooms=2, area_sqft=1000)
    out = predict(f)
    assert "predicted_price_inr" in out
    assert out["predicted_price_inr"] > 0
