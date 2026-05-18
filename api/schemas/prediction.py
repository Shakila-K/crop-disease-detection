from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    crop_type: str = Field(..., description="Type of crop (e.g., 'potato', 'tomato')")


class PredictionResponse(BaseModel):
    crop_type: str = Field(..., description="The crop type that was processed")
    prediction: str = Field(..., description="The predicted disease or 'Healthy'")
    confidence: float = Field(..., description="Confidence score of the prediction (0-1)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    model_version: str = Field(..., description="The version of the model used for prediction")
    grad_cam_url: Optional[str] = Field(None, description="URL to the Grad-CAM visualization image (if enabled)")


class CropInfo(BaseModel):
    crop_name: str
    num_classes: int
    class_names: List[str]


class MetadataResponse(BaseModel):
    available_crops: List[CropInfo]
    model_dir: str


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
