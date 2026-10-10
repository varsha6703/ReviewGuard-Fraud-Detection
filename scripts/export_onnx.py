import argparse, os
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="artifacts/best_model")
    p.add_argument("--output",default="artifacts/reviewguard.onnx")
    args=p.parse_args()
    try:
        import torch
        from transformers import AutoTokenizer
        from app.modeling import DistilBertWithVelocity
        model=DistilBertWithVelocity.from_pretrained(args.model).eval()
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        sample=tokenizer("sample review",return_tensors="pt",padding="max_length",truncation=True,max_length=32)
        velocity=torch.tensor([[0.2]],dtype=torch.float32)
        os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
        torch.onnx.export(model,(sample["input_ids"],sample["attention_mask"],velocity),
            args.output,input_names=["input_ids","attention_mask","velocity"],output_names=["logits"],
            dynamic_axes={"input_ids":{0:"batch"},"attention_mask":{0:"batch"},"velocity":{0:"batch"},"logits":{0:"batch"}},
            opset_version=17)
        print("Exported",args.output)
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path=args.output.replace(".onnx","_int8.onnx")
            quantize_dynamic(args.output,quant_path,weight_type=QuantType.QInt8)
            print("Quantized",quant_path)
        except Exception as e:
            print("INT8 quantization skipped:",e)
    except Exception as e:
        raise SystemExit(f"Export failed: {e}")
if __name__=="__main__": main()
