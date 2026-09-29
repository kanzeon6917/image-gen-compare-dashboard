from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline

MODEL_ID = "SG161222/RealVisXL_V5.0"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("Loading model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = "a futuristic city at night, highly detailed, cinematic lighting"
negative_prompt = "blurry, low quality, distorted, bad anatomy"

generator = torch.Generator(device="cuda").manual_seed(42)

print("Generating image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=10,
    generator=generator,
    width=1024,
    height=1024,
).images[0]

save_path = output_dir / "sample.png"
image.save(save_path)
print(f"Saved to {save_path}")