# ReviewGuard — Fraudulent Review Detection System

Python/FastAPI implementation scaffold for the Embed Square Solutions assignment.

## Features
- Generates a 50,000-row synthetic review dataset (95% legitimate / 5% fraud)
- Stratified 80/10/10 train/validation/test split
- DistilBERT text model with an additional numeric velocity feature
- Training-only imbalance handling (SMOTE on engineered numeric features is provided as a baseline alternative; do not apply SMOTE directly to token IDs)
- FastAPI endpoints for submission, batch detection, review lookup, and drift status
- MongoDB persistence when `MONGODB_URI` is configured; otherwise local in-memory development storage
- KS-test drift monitor
- ONNX export/quantization helper, benchmark script, tests, and Docker Compose

## Important
This is a runnable project scaffold, not evidence that target metrics have already been achieved. Train and evaluate the model, run benchmarks, and record actual results before submitting. Synthetic data can make metrics unrealistically high; document that limitation.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_dataset.py
python -m app.train --data data/reviews_dataset.csv --output artifacts
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for Swagger UI.

## API examples
### POST `/api/reviews/detect/batch`
```json
[
  {"user_id":"user-101","text":"Amazing product, amazing product, best ever 5 stars","rating":5,"velocity":9},
  {"user_id":"user-202","text":"The product works as described after two weeks.","rating":4,"velocity":1}
]
```
### POST `/api/reviews/submit`
```json
{"user_id":"user-101","text":"Amazing product","rating":5,"velocity":7}
```
### GET `/api/reviews/{review_id}`
### GET `/api/drift/status`

## Model
The model concatenates the DistilBERT pooled representation with a normalized velocity scalar before binary classification. Training uses class-weighted loss to address imbalance by default. This is a visually demonstrable alternative to naive SMOTE on language-token IDs. If using SMOTE, use it only on a valid training representation and document the method.

## Drift
`python -m app.drift_monitor` compares saved reference scores in `data/reference_scores.json` with recent prediction scores. A KS p-value below 0.05 triggers a logged mock alert.

## ONNX
Run:
```bash
python scripts/export_onnx.py --model artifacts/best_model
```
INT8 dynamic quantization is attempted when dependencies/model support it. Benchmark the resulting model on your hardware; latency is hardware- and batch-dependent.

## Testing
```bash
pytest -q
```

## Suggested deliverables
- Source code and `requirements.txt`
- Generated dataset or a documented generation command
- Saved trained model/tokenizer (after training)
- API screenshots / sample JSON
- Metrics and latency benchmark from actual runs
- README assumptions and limitations
