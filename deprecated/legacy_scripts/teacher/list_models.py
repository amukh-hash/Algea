from huggingface_hub import list_models
import sys

print("Searching for amazon/chronos models...")
try:
    models = list_models(author="amazon", search="chronos")
    found = []
    for m in models:
        print(f"- {m.id}")
        found.append(m.id)
        
    if not found:
        print("No models found matching 'amazon/chronos*'.")
except Exception as e:
    print(f"Error listing models: {e}")
