from pathlib import Path
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


def generate_images(prompt, negative_prompt, seed, steps, width, height, scales_text):
    scales = [float(x.strip()) for x in scales_text.split(",") if x.strip()]
    results = []
    summaries = []

    for scale in scales:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=scale,
            generator=generator,
            width=int(width),
            height=int(height),
        ).images[0]

        filename = f"seed{seed}_steps{steps}_scale{scale}.png"
        save_path = output_dir / filename
        image.save(save_path)

        results.append(image)
        summaries.append(f"- guidance_scale={scale}, saved={save_path}")

    summary_text = "\n".join(summaries)
    return results, summary_text


with gr.Blocks() as demo:
    gr.Markdown("# Image Generation Compare Dashboard")
    gr.Markdown("同じpromptで guidance scale を変えて比較する最小アプリ")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="a futuristic city at night, highly detailed, cinematic lighting",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, distorted, bad anatomy",
                lines=2,
            )
            seed = gr.Number(label="Seed", value=42, precision=0)
            steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=30, step=1)
            width = gr.Slider(label="Width", minimum=256, maximum=768, value=512, step=64)
            height = gr.Slider(label="Height", minimum=256, maximum=768, value=512, step=64)
            scales_text = gr.Textbox(
                label="Guidance scales (comma separated)",
                value="5.0, 7.5, 10.0",
            )
            run_button = gr.Button("Generate")

        with gr.Column():
            gallery = gr.Gallery(label="Results", columns=3, rows=1, height="auto")
            summary = gr.Markdown()

    run_button.click(
        fn=generate_images,
        inputs=[prompt, negative_prompt, seed, steps, width, height, scales_text],
        outputs=[gallery, summary],
    )

if __name__ == "__main__":
    demo.launch()