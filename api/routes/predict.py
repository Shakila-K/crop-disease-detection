import logging
import os
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
from src.inference.grok_vision import predict_with_grok_vision
from ..schemas.prediction import PredictionResponse, ErrorResponse
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# Max image size allowed (10MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"description": "Unauthorized (Invalid or missing JWT)"},
        500: {"model": ErrorResponse},
    },
)
async def predict_disease(
    image: UploadFile = File(..., description="Image file of the crop leaf/fruit"),
    crop_type: str = Form(..., description="Type of crop (e.g., 'potato', 'tomato')"),
    model_type: str = Form("local", description="Detection model type: 'local' (default) or 'advanced'"),
    language: str = Form("en", description="Language code for response: 'en', 'si', 'ta', 'hi'"),
    user_id: str = Depends(get_current_user),
):
    """
    Predict crop disease from an image.
    Supports selecting between 'local' PyTorch model and 'advanced' Grok Vision model.
    Requires a valid JWT token from the Govi backend.
    """
    logger.info(f"User {user_id} requested prediction for crop: {crop_type}, model_type: {model_type}, language: {language}")

    # Validate file size
    image.file.seek(0, 2)
    file_size = image.file.tell()
    image.file.seek(0)
    if file_size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large. Max size is {MAX_IMAGE_SIZE / 1024 / 1024}MB",
        )

    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image",
        )

    # Get predictor from app state (injected in main.py via lifespan)
    from ..main import app
    predictor: Predictor = app.state.predictor

    try:
        image_bytes = await image.read()
        selected_model = model_type.lower().strip()
        lang_code = language.lower().strip() if language else "en"

        if selected_model == "advanced":
            logger.info("Advanced model selected. Directing to Grok Vision API.")
            grok_result = predict_with_grok_vision(
                image_bytes=image_bytes,
                crop_type=crop_type,
                language=lang_code,
            )
            if grok_result:
                return PredictionResponse(
                    crop_type=crop_type,
                    prediction=grok_result["prediction"],
                    confidence=grok_result["confidence"],
                    all_probabilities={},
                    model_version="grok_vision_v2",
                    model_type="advanced",
                    explanation=grok_result.get("explanation"),
                    fallback_used=False,
                )
            else:
                logger.warning("Advanced Grok model unavailable. Falling back to local model.")

        # Default / Local Model Execution
        result = predictor.predict_from_bytes(image_bytes, crop_type.lower())
        result["model_type"] = "local"

        # Check if confidence is below threshold for Grok Vision fallback
        threshold = float(os.getenv("GROK_FALLBACK_THRESHOLD", "0.60"))
        if result.get("confidence", 0.0) < threshold:
            logger.info(
                f"Local model confidence ({result.get('confidence', 0.0):.4f}) is below threshold ({threshold}). "
                f"Triggering Grok Vision API fallback."
            )
            grok_result = predict_with_grok_vision(
                image_bytes=image_bytes,
                crop_type=crop_type,
                trained_classes=list(result.get("all_probabilities", {}).keys()),
                language=lang_code,
            )
            if grok_result:
                result["fallback_used"] = True
                result["fallback_reason"] = (
                    f"Low local model confidence ({result.get('confidence', 0.0):.2f} < {threshold:.2f})"
                )
                result["prediction"] = grok_result["prediction"]
                result["confidence"] = grok_result["confidence"]
                result["explanation"] = grok_result.get("explanation")
                result["model_version"] = f"{result.get('model_version', crop_type)}+grok_vision"

        return PredictionResponse(**result)


    except ValueError as e:
        # Expected error like unsupported crop type
        logger.warning(f"Prediction ValueError: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)},
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An internal error occurred during prediction"},
        )
