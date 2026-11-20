
import os
import logging

import mlflow
import mlflow.tensorflow
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ==========================================
# CONFIGURATION
# ==========================================

CLASS_NAMES = ["Bengin", "Malignant", "Normal"]
IMAGE_SIZE = (160, 160)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Local default; override this using environment variables in Docker.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "chest-cancer-classifier"
)

MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# FASTAPI INITIALIZATION
# ==========================================

app = FastAPI(
    title="Chest Cancer Classification API",
    description="Chest cancer classification using MLflow Model Registry",
    version="2.0.0"
)

# ==========================================
# LOAD PRODUCTION MODEL FROM MLFLOW
# ==========================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    logger.info("Connecting to MLflow: %s", MLFLOW_TRACKING_URI)
    logger.info("Loading registered model: %s", MODEL_URI)

    model = mlflow.tensorflow.load_model(MODEL_URI)

    logger.info("MLflow Production model loaded successfully")

except Exception as e:
    logger.exception("Failed to load model from MLflow Registry")
    raise RuntimeError(
        f"Could not load MLflow model {MODEL_URI}: {e}"
    ) from e


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Chest Cancer Classification API is running",
        "status": "success",
        "model_source": "MLflow Model Registry",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": "MLflow Model Registry",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed"
        )

    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 10 MB"
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Same preprocessing used during training
        image_array = tf.io.decode_image(
            contents,
            channels=3,
            expand_animations=False
        )

        image_array.set_shape([None, None, 3])

        image_array = tf.image.resize(
            image_array,
            IMAGE_SIZE
        )

        image_array = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
        )

        image_array = tf.expand_dims(image_array, axis=0)

        # Predict using MLflow Production model
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            tf.argmax(predictions).numpy()
        )

        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])

        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
                "model_source": "MLflow Model Registry",
                "model_name": MODEL_NAME,
                "model_stage": MODEL_STAGE,
                "disclaimer": (
                    "This is an experimental AI prediction, "
                    "not a medical diagnosis. Consult a qualified doctor."
                )
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        await file.close()
