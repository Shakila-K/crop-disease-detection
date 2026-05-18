# Crop Disease Detection AI

A production-level, expandable AI system for identifying crop diseases from leaf/fruit images, designed for the Govi mobile backend.

## Architecture

This system uses a modular architecture combining a shared **EfficientNet-B0 backbone** with **per-crop classification heads**.

It supports **Incremental Learning** via Experience Replay and Knowledge Distillation, meaning you can add new diseases to a crop without retraining the entire model and suffering from catastrophic forgetting.

## Project Structure

*   `src/`: Core ML package (data, models, training, inference).
*   `scripts/`: User-facing CLI scripts for training and evaluation.
*   `api/`: FastAPI backend for production serving.
*   `models/`: Directory for saved model artifacts and exemplars.
*   `data/raw/`: Directory for your training images.

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Organize your data in `data/raw/<crop_name>/<disease_name>/<images.jpg>`.
    Example:
    ```
    data/raw/potato/
      Early_Blight/
        img1.jpg
      Late_Blight/
        img2.jpg
      Healthy/
        img3.jpg
    ```

## Training

### Phase 1: Initial Training
Train a crop model from scratch. This automatically splits the data (80/10/10), trains the backbone and head, evaluates, and stores exemplars for future incremental learning.

```bash
python scripts/train_initial.py \
    --crop potato \
    --data-dir data/raw/potato \
    --epochs 50
```

### Phase 2: Incremental Training
Add a new disease (e.g., "Septoria_Leaf_Spot") to an existing crop model. Place the new data in a separate folder.

```bash
python scripts/train_incremental.py \
    --crop potato \
    --new-data-dir data/raw/potato_new \
    --existing-head models/heads/potato_v1.pt \
    --epochs 30
```

### Evaluation
Evaluate a trained head on the test set:

```bash
python scripts/evaluate.py \
    --crop potato \
    --data-dir data/raw/potato \
    --head models/heads/potato_v2.pt
```

## API Deployment

The Phase 2 API is built with FastAPI and secured with JWT.

1.  **Set JWT Secret**:
    Ensure the `GOVI_JWT_SECRET` environment variable matches the Govi backend.

2.  **Run Locally**:
    ```bash
    uvicorn api.main:app --reload
    ```

3.  **Run with Docker**:
    ```bash
    docker-compose up --build -d
    ```

The API will be available at `http://localhost:8000`. You can test endpoints via the Swagger UI at `http://localhost:8000/docs`.
