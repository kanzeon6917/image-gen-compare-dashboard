import gradio as gr

from comparison import build_comparisons
from generation import ImageGenerator, generate_and_save


def create_app():
    generator = ImageGenerator()
    with gr.Blocks() as demo:
        gr.Markdown("# Image Generation Compare Dashboard")
        gr.Markdown("比較対象を選び、同じ Prompt・Seed で生成結果を比較できます。")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=3,
                    value="a futuristic city at night, highly detailed, cinematic lighting")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=2,
                    value="blurry, low quality, distorted, bad anatomy")
                seed = gr.Number(label="Seed", value=42, precision=0)
                compare_mode = gr.Radio(
                    choices=[("Guidance Scale", "guidance_scale"), ("Steps", "steps")],
                    value="guidance_scale", label="Compare parameter")
                scales_text = gr.Textbox(label="Guidance scales (0–20, comma separated)",
                    value="5.0, 7.5, 10.0")
                steps_text = gr.Textbox(label="Steps (1–100, comma separated)",
                    value="10, 20, 30, 50", visible=False)
                fixed_scale = gr.Slider(label="Fixed Guidance Scale", minimum=0,
                    maximum=20, value=7.5, step=0.5, visible=False)
                fixed_steps = gr.Slider(label="Fixed Steps", minimum=1,
                    maximum=100, value=30, step=1)
                gr.Markdown("比較候補は最大20個。Seed は 0〜4294967295 の整数です。")
                width = gr.Slider(label="Width", minimum=256, maximum=768, value=512, step=64)
                height = gr.Slider(label="Height", minimum=256, maximum=768, value=512, step=64)
                run_button = gr.Button("Generate")
            with gr.Column():
                gallery = gr.Gallery(label="Results", columns=3, height="auto")
                summary = gr.Markdown()

        # Map named settings at the UI boundary instead of passing positional values onward.
        fields = dict(prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                      compare_mode=compare_mode, fixed_scale=fixed_scale,
                      fixed_steps=fixed_steps, scales_text=scales_text,
                      steps_text=steps_text, width=width, height=height)

        def generate_from_ui(data):
            try:
                comparisons = build_comparisons(**{
                    name: data[component] for name, component in fields.items()
                })
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            results = generate_and_save(comparisons, generator)
            return (
                [(str(result.path), result.caption) for result in results],
                "\n".join(f"- {result.caption}, saved={result.path}" for result in results),
            )

        run_button.click(fn=generate_from_ui, inputs=set(fields.values()),
                         outputs=[gallery, summary], concurrency_limit=1)
        compare_mode.change(
            fn=lambda mode: (
                gr.update(visible=mode == "guidance_scale"),
                gr.update(visible=mode == "steps"),
                gr.update(visible=mode == "steps"),
                gr.update(visible=mode == "guidance_scale"),
            ),
            inputs=compare_mode,
            outputs=[scales_text, steps_text, fixed_scale, fixed_steps],
            queue=False,
        )
    return demo


if __name__ == "__main__":
    create_app().launch()
