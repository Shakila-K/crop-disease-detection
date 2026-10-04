"""
Grok Vision API Fallback Module.

Provides fallback disease detection using xAI Grok Vision models when the local
PyTorch classification model has low confidence or fails to identify a trained disease.
"""

import base64
import json
import logging
import os
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    import httpx as requests

logger = logging.getLogger(__name__)

# Default model and URL
DEFAULT_GROK_VISION_MODEL = "grok-4.20-non-reasoning"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def predict_with_grok_vision(
    image_bytes: bytes,
    crop_type: str,
    trained_classes: Optional[List[str]] = None,
    language: str = "en",
) -> Optional[Dict]:
    """
    Query xAI Grok Vision API to detect crop disease from image bytes.

    Args:
        image_bytes: Raw bytes of the image file.
        crop_type: Crop type name (e.g. 'potato', 'tomato').
        trained_classes: List of class names supported by the local model.
        language: Language code for the response ('en', 'si', 'ta', 'hi').

    Returns:
        Dict with keys ('prediction', 'confidence', 'explanation') if successful,
        or None if Grok API call fails or key is missing.
    """
    grok_api_key = os.getenv("GROK_API_KEY", "").strip()
    if not grok_api_key:
        logger.warning("GROK_API_KEY is not configured. Skipping Grok Vision fallback.")
        return None

    model = os.getenv("GROK_VISION_MODEL", DEFAULT_GROK_VISION_MODEL)
    
    # Base64 encode the image
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_image}"

    classes_info = ""
    if trained_classes:
        classes_info = f"\nClasses known by local model: {', '.join(trained_classes)}"

    # Language instruction for non-English responses
    _LANGUAGE_NAMES = {
        "en": "English",
        "si": "Sinhala (සිංහල)",
        "ta": "Tamil (தமிழ்)",
        "hi": "Hindi (हिन्दी)",
    }
    lang_name = _LANGUAGE_NAMES.get(language, "English")
    language_instruction = ""
    if language and language != "en":
        language_instruction = (
            f"\n\nIMPORTANT: You MUST write the 'explanation' field entirely in {lang_name}. "
            f"The 'prediction' field should remain in English for system compatibility. "
            f"The 'confidence' field is a number and requires no translation."
        )

    prompt = f"""You are an expert plant pathologist and agricultural consultant.
Analyze this leaf image for a plant of crop type: '{crop_type}'.{classes_info}

The local detection model had low confidence in classifying this image.
Examine the visual symptoms on the plant tissue (e.g. lesions, leaf spots, blights, rust, mildew, wilting, mosaic patterns, or healthy tissue).

Determine:
1. The most accurate disease name (e.g., 'Late Blight', 'Early Blight', 'Leaf Mold', 'Yellow Leaf Curl Virus', or 'Healthy').
2. Your confidence level (between 0.0 and 1.0).
3. A concise diagnostic explanation describing the visible symptoms that support your diagnosis.{language_instruction}

Return ONLY a valid JSON object with no markdown formatting or fences:
{{
  "prediction": "<Disease Name or Healthy>",
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<Concise diagnostic explanation of visual symptoms>"
}}"""

    # List of vision models to try in case specific version is not enabled on account
    primary_model = os.getenv("GROK_VISION_MODEL", DEFAULT_GROK_VISION_MODEL).strip()
    candidate_models = [primary_model]
    for fallback in ["grok-4.20-non-reasoning", "grok-4.20", "grok-4.3", "grok-4.5", "grok-2-vision", "grok-2-vision-1212"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)


    last_error = None

    for model in candidate_models:
        try:
            logger.info(f"Calling Grok Vision API ({model}) for crop '{crop_type}'...")
            response = requests.post(
                GROK_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {grok_api_key}",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url,
                                        "detail": "high"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                },
                timeout=30,
            )
            
            if response.status_code == 404:
                logger.warning(f"Grok model '{model}' returned 404 Not Found: {response.text}")
                last_error = f"404 Not Found for model '{model}'"
                continue  # Try next candidate model

            response.raise_for_status()

            response_data = response.json()
            raw_content = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            # Strip markdown fences if present
            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]
            raw_content = raw_content.strip()

            parsed = json.loads(raw_content)
            
            prediction = parsed.get("prediction", "Unknown Disease")
            confidence = float(parsed.get("confidence", 0.70))
            explanation = parsed.get("explanation", "Identified via Grok Vision AI analysis.")

            logger.info(f"Grok Vision prediction success using '{model}': {prediction} (confidence: {confidence:.2f})")
            return {
                "prediction": str(prediction),
                "confidence": round(float(confidence), 4),
                "explanation": str(explanation),
            }

        except requests.exceptions.RequestException as e:
            resp_text = getattr(e.response, "text", str(e))
            logger.error(f"Grok Vision API HTTP error for model '{model}': {e} | Details: {resp_text}")
            last_error = str(e)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse Grok Vision API response for model '{model}': {e}")
            last_error = str(e)

    logger.error(f"All Grok Vision candidate models failed. Last error: {last_error}")
    return None

