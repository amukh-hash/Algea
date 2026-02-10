
import torch
from chronos import Chronos2Pipeline

def inspect_pipeline():
    model_id = "amazon/chronos-2"
    print(f"Loading {model_id}...")
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
    
    # create dummy context
    context = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]) # [1, 5]
    
    print("\n=== Test Predict with num_return_sequences=5, do_sample=True ===")
    try:
        out = pipeline.predict(
            context, 
            prediction_length=3, 
            num_return_sequences=5, 
            do_sample=True,
            top_k=20,
            top_p=0.9
        )
        print(f"Shape: {out.shape}")
        print(f"Values: {out}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n=== Test Predict with just do_sample=True ===")
    try:
        out = pipeline.predict(context, prediction_length=3, do_sample=True)
        print(f"Shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")

    if hasattr(pipeline, "tokenizer"):
        print("Pipeline has tokenizer")
    
if __name__ == "__main__":
    inspect_pipeline()
