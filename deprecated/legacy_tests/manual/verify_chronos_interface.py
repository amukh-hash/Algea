
import sys
import os
import torch
import logging

# Ensure backend in path
sys.path.append(os.getcwd())

from backend.app.models.chronos2_teacher import load_chronos_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("--- Starting Chronos 2 Interface Verification ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model (base, no adapter)
    print("Loading model...")
    try:
        wrapper, info = load_chronos_adapter(
            model_id="amazon/chronos-2",
            use_qlora=False,
            device=device,
            adapter_path=None,
            eval_mode=True
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load model: {e}")
        return

    model = wrapper.model
    print(f"Model ID: {type(model)}")
    print(f"Is Pipeline?: {'Pipeline' in str(type(model))}")

    # Inspect methods
    print("\n--- Method Inspection ---")
    methods = ["predict", "generate", "forward"]
    for m in methods:
        has = hasattr(model, m)
        print(f"Has '{m}': {has}")
        if has:
            func = getattr(model, m)
            print(f"  Doc: {func.__doc__}")
            # Try to get help/signature
            try:
                import inspect
                print(f"  Sig: {inspect.signature(func)}")
            except Exception as e:
                print(f"  Sig Error: {e}")

    # Test Inference
    print("\n--- Inference Test ---")
    # Batch=1, Len=20
    context = torch.randn(1, 20).to(device)
    print(f"Input Context: {context.shape}")

    try:
        # Try pipeline-style predict first
        print("Attempting wrapper.generate() with num_samples=5...")
        out = wrapper.generate(context, prediction_length=5, num_samples=5)
        print(f"Generate Output Shape: {out.shape}")
        print("Generate SUCCESS")
    except Exception as e:
        print(f"Generate FAILED: {e}")

    # Try raw predict if available
    if hasattr(model, "predict"):
        print("\nAttempting model.predict() directly...")
        try:
            # Guessing arguments from common Chronos usage
            # predict(context, prediction_length, num_samples, ...)
            out2 = model.predict(context, prediction_length=5, num_samples=5)
            # Output might be tensor or numpy
            if hasattr(out2, "shape"):
                print(f"Predict Output Shape: {out2.shape}")
            else:
                print(f"Predict Output Type: {type(out2)}")
            print("Predict SUCCESS")
        except Exception as e:
            print(f"Predict FAILED: {e}")

if __name__ == "__main__":
    main()
