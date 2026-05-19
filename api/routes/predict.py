import logging
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
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
    user_id: str = Depends(get_current_user),
):
    """
    Predict crop disease from an image.
    Requires a valid JWT token from the Govi backend.
    """
    logger.info(f"User {user_id} requested prediction for crop: {crop_type}")

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
        result = predictor.predict_from_bytes(image_bytes, crop_type.lower())
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
