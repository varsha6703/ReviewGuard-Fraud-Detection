
from fastapi import FastAPI, HTTPException
from app.schemas import ReviewInput
from app.inference import predict_review
from app.config import FRAUD_THRESHOLD
from app.drift_monitor import check_drift

from app.storage import (
    save_review,
    get_review,
    all_recent_scores,
    save_audit_log
)


app = FastAPI(
    title="ReviewGuard API",
    version="1.0.0",
    description=(
        "Fraudulent review detection, persistence, "
        "batch inference, and drift monitoring"
    )
)


# --------------------------------------------------
# Score a single review using the ML inference module
# --------------------------------------------------
def score_one(review):
    try:
        score, reasons, mode = predict_review(
            review.text,
            review.velocity,
            review.rating
        )

        status = (
            "quarantined"
            if score >= FRAUD_THRESHOLD
            else "approved"
        )

        doc = save_review(
            review.user_id,
            review.text,
            review.rating,
            review.velocity,
            score,
            reasons,
            status
        )
        
        save_audit_log(
            action="review_analyzed",
            review_id=doc["review_id"],
            details={
                "user_id": review.user_id,
                "score": round(float(score), 6),
                "status": status,
                "inference_mode": mode
            }
        )


        return {
            "id": doc["review_id"],
            "score": round(float(score), 6),
            "explanation": reasons,
            "status": status,
            "inference_mode": mode
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Review processing failed: {str(exc)}"
        )


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ReviewGuard API"
    }


# --------------------------------------------------
# Submit and detect a single review
# --------------------------------------------------
@app.post("/api/reviews/submit")
def submit_review(review: ReviewInput):
    return score_one(review)


# --------------------------------------------------
# Detect fraudulent reviews in a batch
# Maximum batch size: 100 reviews
# --------------------------------------------------
@app.post("/api/reviews/detect/batch")
def detect_batch(reviews: list[ReviewInput]):
    if len(reviews) > 100:
        raise HTTPException(
            status_code=413,
            detail="Batch limit is 100 reviews"
        )

    results = []

    for review in reviews:
        results.append(score_one(review))

    return results


# --------------------------------------------------
# Retrieve a review by its ID
# --------------------------------------------------
@app.get("/api/reviews/{review_id}")
def fetch_review(review_id: str):
    try:
        doc = get_review(review_id)

        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Review not found"
            )

        return doc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve review: {str(exc)}"
        )


# --------------------------------------------------
# Check prediction-score drift
# --------------------------------------------------
@app.get("/api/drift/status")
def drift_status():
    try:
        scores = all_recent_scores()
        return check_drift(scores)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Drift monitoring failed: {str(exc)}"
        )
