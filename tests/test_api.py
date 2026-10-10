from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"

def test_batch_detection():
    payload=[{"user_id":"u1","text":"Amazing product best ever 5 stars amazing","rating":5,"velocity":9}]
    r=client.post("/api/reviews/detect/batch",json=payload)
    assert r.status_code==200
    body=r.json()
    assert len(body)==1
    assert 0 <= body[0]["score"] <= 1
    assert "status" in body[0]
    assert "explanation" in body[0]

def test_missing_review():
    r=client.get("/api/reviews/not-a-real-id")
    assert r.status_code==404
