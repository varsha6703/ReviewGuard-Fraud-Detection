
import os
os.environ["USE_TF"] = "0"

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
from app.modeling import DistilBertWithVelocity

MODEL_PATH = "artifacts/best_model"
ONNX_PATH = "artifacts/onnx_custom/model.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertWithVelocity.from_pretrained(MODEL_PATH)
model.eval()

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

text = "This product is excellent"
sample = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128
)

velocity = torch.tensor([[0.3]], dtype=torch.float32)

with torch.no_grad():
    pytorch_logits = model(
        input_ids=sample["input_ids"],
        attention_mask=sample["attention_mask"],
        velocity=velocity
    ).logits.numpy()

onnx_logits = session.run(
    ["logits"],
    {
        "input_ids": sample["input_ids"].numpy(),
        "attention_mask": sample["attention_mask"].numpy(),
        "velocity": velocity.numpy()
    }
)[0]

print("PyTorch logits:", pytorch_logits)
print("ONNX logits:", onnx_logits)
print("Outputs match:", np.allclose(
    pytorch_logits, onnx_logits, rtol=1e-3, atol=1e-4
))
print("ONNX inputs:", [i.name for i in session.get_inputs()])
print("ONNX output shape:", onnx_logits.shape)
