"""Debug script to trace num_output_patches calculation."""
import sys
sys.path.insert(0, ".")

from backend.app.models.chronos2_teacher import load_chronos_adapter
import torch

device = torch.device("cpu")
wrapper, info = load_chronos_adapter(
    "amazon/chronos-2", 
    use_qlora=False, 
    device=device, 
    lora_config={"rank": 16, "alpha": 32, "dropout": 0.05}
)

print(f"Wrapper type: {type(wrapper)}")
print(f"Has input_patch_embedding: {hasattr(wrapper.model, 'input_patch_embedding')}")

# Test calculation
result = wrapper._infer_num_output_patches(20)
print(f"_infer_num_output_patches(20) = {result}")
