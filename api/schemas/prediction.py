from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    crop_type: str = Field(..., description="Type of crop (e.g., 'potato', 'tomato')")


class PredictionResponse(BaseModel):
    crop_type: str = Field(..., description="The crop type that was processed")
    prediction: str = Field(..., description="The predicted disease or 'Healthy'")
    confidence: float = Field(..., description="Confidence score of the prediction (0-1)")
    all_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probabilities for all classes")
    model_version: str = Field(..., description="The version of the model used for prediction")
    model_type: str = Field("local", description="Model selected for detection: 'local' or 'advanced'")
    grad_cam_url: Optional[str] = Field(None, description="URL to the Grad-CAM visualization image (if enabled)")
    fallback_used: bool = Field(False, description="Whether Grok Vision API fallback was used")
    fallback_reason: Optional[str] = Field(None, description="Reason for triggering fallback")
    explanation: Optional[str] = Field(None, description="Diagnostic explanation (populated when Grok is used)")


class CropInfo(BaseModel):
    crop_name: str
    num_classes: int
    class_names: List[str]


class MetadataResponse(BaseModel):
    available_crops: List[CropInfo]
    model_dir: str


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
