from mlx_vlm import load, generate
from PIL import Image
import os

model_path = "/Volumes/Extreme SSD/models/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path, processor_kwargs={"use_fast": False})

# Create a tiny dummy image
img = Image.new('RGB', (100, 100), color = 'red')

prompt = "What is in this image?"
output = generate(model, processor, prompt, [img], verbose=True)
print("Output:", output)
