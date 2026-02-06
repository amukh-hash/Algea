
import os
import sys
import torch
# Add backend to path
sys.path.append(os.getcwd())

try:
    from chronos import Chronos2Pipeline
    print("Quantization not supported in this script, loading bf16/cpu")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )
    print(f"Pipeline methods: {dir(pipeline)}")
    if hasattr(pipeline, "predict"):
        print(f"Predict doc: {pipeline.predict.__doc__}")
        
except Exception as e:
    print(f"Error: {e}")
