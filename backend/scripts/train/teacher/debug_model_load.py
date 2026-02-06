"""Debug Chronos-2 model loading for daily equity protocol usage."""

import torch
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import sys

model_id = "amazon/chronos-2"

print(f"Loading {model_id}...")
try:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded.")
    
    print("Preparing for kbit...")
    model = prepare_model_for_kbit_training(model)
    print("Prepared.")
    
    print("Finding LoRA targets...")
    import re
    import torch.nn as nn
    
    modules = {name for name, _ in model.named_modules() if isinstance(_, nn.Linear)}
    print(f"Found {len(modules)} linear modules.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
