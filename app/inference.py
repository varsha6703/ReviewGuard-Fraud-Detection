
import os
import re
import numpy as np

from app.config import MODEL_PATH

ONNX_PATH = os.path.join(
    os.path.dirname(MODEL_PATH),
    "onnx_custom",
    "model.onnx"
)

_tokenizer = None
_model = None
_torch = None
_onnx_session = None
_backend = None


def _load_model():
    global _tokenizer, _model, _torch
    global _onnx_session, _backend

    if _backend is not None:
        return True

    try:
        import torch
        from transformers import AutoTokenizer

        _torch = torch
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        if os.path.isfile(ONNX_PATH):
            import onnxruntime as ort

            _onnx_session = ort.InferenceSession(
                ONNX_PATH,
                providers=["CPUExecutionProvider"]
            )
            _backend = "onnx"
            print("ONNX model loaded successfully")
            return True

        from app.modeling import DistilBertWithVelocity

        _model = DistilBertWithVelocity.from_pretrained(MODEL_PATH)
        _model.eval()
        _backend = "pytorch"
        print("PyTorch model loaded successfully")
        return True

    except Exception as error:
        print(f"Model loading failed: {error}")
        _tokenizer = None
        _model = None
        _torch = None
        _onnx_session = None
        _backend = None
        return False


def explain_rules(text, velocity, rating):
    reasons = []

    if velocity >= 6:
        reasons.append("High review-posting velocity")

    if len(re.findall(
        r"\b(amazing|best ever|5 stars|excellent)\b",
        text.lower()
    )) >= 2:
        reasons.append("Promotional or repetitive phrasing")

    if len(text.split()) < 5:
        reasons.append("Unusually short review")

    if rating == 5 and velocity >= 6:
        reasons.append(
            "High rating combined with high posting velocity"
        )

    return reasons or [
        "No strong rule-based warning detected"
    ]


def predict_review(text, velocity, rating):
    if _load_model():
        enc = _tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128 if _backend == "onnx" else 256,
            return_tensors="np" if _backend == "onnx" else "pt"
        )

        velocity_value = np.array(
            [[min(velocity / 10.0, 1.0)]],
            dtype=np.float32
        )

        if _backend == "onnx":
            logits = _onnx_session.run(
                ["logits"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                    "velocity": velocity_value
                }
            )[0]

            logits = logits - np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(logits) / np.exp(logits).sum(
                axis=1, keepdims=True
            )
            score = float(probabilities[0, 1])

        else:
            velocity_tensor = _torch.tensor(
                velocity_value,
                dtype=_torch.float32
            )

            with _torch.no_grad():
                outputs = _model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    velocity=velocity_tensor
                )

                probabilities = _torch.softmax(
                    outputs.logits, dim=1
                )
                score = float(probabilities[0, 1].item())

        return (
            score,
            explain_rules(text, velocity, rating),
            _backend
        )

    words = text.lower()
    score = 0.08
    score += min(velocity / 10.0, 1.0) * 0.58

    repeated_terms = sum(
        words.count(term)
        for term in ["amazing", "best ever", "5 stars"]
    )
    score += min(repeated_terms * 0.08, 0.25)

    if len(text.split()) < 5:
        score += 0.08

    score = float(min(score, 0.99))

    return (
        score,
        explain_rules(text, velocity, rating),
        "rules_fallback"
    )
