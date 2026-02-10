
from chronos import Chronos2Pipeline
import inspect, torch

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
sig = inspect.signature(pipeline.predict)
print("pipeline.predict params:", list(sig.parameters.keys()))
