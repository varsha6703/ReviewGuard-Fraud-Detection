
import os

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017/"
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "ReviewGuardDB"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "artifacts/best_model"
)

FRAUD_THRESHOLD = float(
    os.getenv("FRAUD_THRESHOLD", "0.70")
)
