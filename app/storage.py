
from datetime import datetime, timezone
from uuid import uuid4

from app.config import MONGODB_URI, MONGODB_DATABASE


# In-memory fallback if MongoDB is unavailable
_memory = {}

_client = None
_collection = None


def _mongo_collection():
    """Connect to MongoDB and return the reviews collection."""
    global _client, _collection

    if not MONGODB_URI:
        return None

    if _collection is None:
        from pymongo import MongoClient

        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=1500
        )

        # Verify that MongoDB is reachable
        _client.admin.command("ping")

        _collection = _client[MONGODB_DATABASE]["reviews"]

        # Index for user activity and timestamp queries
        _collection.create_index(
            [("user_id", 1), ("timestamp", -1)]
        )

    return _collection


def save_audit_log(action, review_id=None, details=None):
    """Save an audit event to MongoDB."""

    log = {
        "audit_id": str(uuid4()),
        "action": action,
        "review_id": review_id,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    try:
        col = _mongo_collection()

        if col is None:
            return False

        audit_collection = col.database["audit_logs"]
        audit_collection.insert_one(log)

        return True

    except Exception as exc:
        print(f"Audit log save failed: {exc}")
        return False


def save_review(
    user_id,
    text,
    rating,
    velocity,
    score,
    explanation,
    status
):
    """Save a scored review to MongoDB or memory fallback."""

    doc = {
        "review_id": str(uuid4()),
        "user_id": user_id,
        "text": text,
        "rating": rating,
        "velocity": velocity,
        "score": float(score),
        "explanation": explanation,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    try:
        col = _mongo_collection()

        if col is not None:
            col.insert_one(doc)
        else:
            _memory[doc["review_id"]] = doc

    except Exception as exc:
        # Keep the API usable if MongoDB is unavailable.
        print(f"Review storage failed; using memory fallback: {exc}")
        _memory[doc["review_id"]] = doc
        doc["storage_fallback"] = True

    return doc


def get_review(review_id):
    """Fetch a review using its unique review ID."""

    try:
        col = _mongo_collection()

        if col is not None:
            doc = col.find_one(
                {"review_id": review_id},
                {"_id": 0}
            )

            if doc:
                return doc

    except Exception as exc:
        print(f"Review retrieval failed: {exc}")

    return _memory.get(review_id)


def all_recent_scores(limit=1000):
    """Return recent fraud scores for drift monitoring."""

    try:
        col = _mongo_collection()

        if col is not None:
            docs = list(
                col.find(
                    {"score": {"$type": "number"}},
                    {"_id": 0, "score": 1, "timestamp": 1}
                )
                .sort("timestamp", -1)
                .limit(limit)
            )

            return [
                float(doc["score"])
                for doc in docs
            ]

    except Exception as exc:
        print(f"Fetching recent scores failed: {exc}")

    # Fallback to scores stored in memory
    docs = list(_memory.values())

    return [
        float(doc["score"])
        for doc in docs[-limit:]
        if "score" in doc
    ]
