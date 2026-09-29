from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
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
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
    width=512,
    height=512,
).images[0]

save_path = output_dir / "sample.png"
image.save(save_path)
print(f"Saved to {save_path}")