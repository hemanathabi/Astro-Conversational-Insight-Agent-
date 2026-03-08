"""Tests for the Flask API endpoints."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


VALID_PAYLOAD = {
    "session_id": "test-session-001",
    "message": "What is my zodiac sign personality?",
    "user_profile": {
        "name": "Ritika",
        "birth_date": "1995-08-20",
        "birth_time": "14:30",
        "birth_place": "Jaipur, India",
        "preferred_language": "en",
    },
}


def get_error_message(response):
    """Extract error message from flask-restx response."""
    data = response.get_json()
    return data.get("message", data.get("error", ""))


def test_health_endpoint(client):
    """Test GET /health returns 200 with service info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_chat_missing_session_id(client):
    """Test /chat returns 400 when session_id is missing."""
    payload = {
        "message": "Hello",
        "user_profile": {"name": "Test", "birth_date": "2000-01-01"},
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "session_id" in get_error_message(response)


def test_chat_missing_message(client):
    """Test /chat returns 400 when message is missing."""
    payload = {
        "session_id": "test",
        "user_profile": {"name": "Test", "birth_date": "2000-01-01"},
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "message" in get_error_message(response)


def test_chat_missing_profile(client):
    """Test /chat returns 400 when user_profile is missing."""
    payload = {"session_id": "test", "message": "Hello"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "user_profile" in get_error_message(response)


def test_chat_invalid_birth_date(client):
    """Test /chat returns 400 for invalid birth_date format."""
    payload = {
        "session_id": "test",
        "message": "Hello",
        "user_profile": {"name": "Test", "birth_date": "20-08-1995"},
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "YYYY-MM-DD" in get_error_message(response)


def test_chat_invalid_birth_time(client):
    """Test /chat returns 400 for invalid birth_time format."""
    payload = {
        "session_id": "test",
        "message": "Hello",
        "user_profile": {
            "name": "Test",
            "birth_date": "1995-08-20",
            "birth_time": "2:30pm",
        },
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "HH:MM" in get_error_message(response)


def test_chat_empty_body(client):
    """Test /chat returns 400 for empty request body."""
    response = client.post("/chat", data="", content_type="application/json")
    assert response.status_code == 400


def test_chat_missing_name(client):
    """Test /chat returns 400 when name is missing from profile."""
    payload = {
        "session_id": "test",
        "message": "Hello",
        "user_profile": {"birth_date": "1995-08-20"},
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 400
    assert "name" in get_error_message(response)
