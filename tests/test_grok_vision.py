import os
from unittest.mock import MagicMock, patch

import pytest

from src.inference.grok_vision import predict_with_grok_vision


def test_predict_with_grok_vision_no_api_key(monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "")
    result = predict_with_grok_vision(b"fake_image_bytes", "potato")
    assert result is None


@patch("requests.post")
def test_predict_with_grok_vision_success(mock_post, monkeypatch):
    monkeypatch.setenv("GROK_API_KEY", "test-grok-key")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"prediction": "Late Blight", "confidence": 0.92, "explanation": "Dark water-soaked lesions visible."}'
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    result = predict_with_grok_vision(b"fake_image_bytes", "potato", ["Early Blight", "Healthy"])

    assert result is not None
    assert result["prediction"] == "Late Blight"
    assert result["confidence"] == 0.92
    assert "lesions" in result["explanation"]
    mock_post.assert_called_once()
