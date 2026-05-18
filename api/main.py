import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.inference.predictor import Predictor
from .routes import predict
from .schemas.prediction import MetadataResponse, CropInfo

logger = logging.getLogger(__name__)

# Configure basic logging for the API
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the ML model into memory
    when the application starts, and clean it up when it shuts down.
    """
    model_dir = "models"
    logger.info(f"Starting Govi Disease Detection API. Loading models from: {model_dir}")
    
    try:
        predictor = Predictor(model_dir=model_dir)
        app.state.predictor = predictor
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # In a real production setup, you might want to retry or let it fail
        # so Kubernetes can restart the pod.
        app.state.predictor = None
        
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API. Cleaning up resources.")
    app.state.predictor = None


# Initialize FastAPI app
app = FastAPI(
    title="Govi Disease Detection API",
    description="Production API for crop disease detection using expandable incremental learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the Govi backend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router, tags=["Prediction"])


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    model_loaded = hasattr(app.state, "predictor") and app.state.predictor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.get("/crops", response_model=MetadataResponse, tags=["System"])
async def get_crops():
    """Get metadata about all supported crops and diseases."""
    if not hasattr(app.state, "predictor") or app.state.predictor is None:
        return {"error": "Predictor not loaded"}
        
    predictor: Predictor = app.state.predictor
    available_crops = predictor.get_available_crops()
    
    crops = [
        CropInfo(
            crop_name=info["crop_name"],
            num_classes=info["num_classes"],
            class_names=info["class_names"]
        ) for info in available_crops
    ]
    
    return MetadataResponse(
        available_crops=crops,
        model_dir="models"
    )

