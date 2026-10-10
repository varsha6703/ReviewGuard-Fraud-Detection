
import os
os.environ["USE_TF"] = "0"

import torch
from app.modeling import DistilBertWithVelocity
from transformers import AutoTokenizer

MODEL_PATH = "artifacts/best_model"
OUTPUT_DIR = "artifacts/onnx_custom"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertWithVelocity.from_pretrained(MODEL_PATH)
model.eval()

# Dummy inputs for ONNX export
sample = tokenizer(
    "This product is excellent",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128
)

input_ids = sample["input_ids"]
attention_mask = sample["attention_mask"]
velocity = torch.tensor([[0.3]], dtype=torch.float32)

torch.onnx.export(
    model,
    (input_ids, attention_mask, velocity),
    os.path.join(OUTPUT_DIR, "model.onnx"),
    input_names=["input_ids", "attention_mask", "velocity"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "velocity": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    dynamo=False
)

tokenizer.save_pretrained(OUTPUT_DIR)
print("Custom ONNX export completed:", OUTPUT_DIR)
