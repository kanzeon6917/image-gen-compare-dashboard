"""Lazy model loading and storage of images with their generation settings."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

from comparison import GenerationSettings

MODEL_ID = "runwayml/stable-diffusion-v1-5"


@dataclass(frozen=True)
class GenerationResult:
    path: Path
    settings: GenerationSettings

    @property
    def caption(self):
        return f"guidance_scale={self.settings.guidance_scale}, steps={self.settings.steps}"


class ImageGenerator:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._lock = Lock()

    def generate(self, settings):
        # A shared pipeline must not be invoked concurrently.
        with self._lock:
            import torch
            from diffusers import StableDiffusionPipeline

            if self._pipe is None:
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id, torch_dtype=torch.float16, use_safetensors=True,
                ).to("cuda")
                pipe.enable_attention_slicing()
                self._pipe = pipe

            generator = torch.Generator(device="cuda").manual_seed(settings.seed)
            
            return self._pipe(
                prompt=settings.prompt,
                negative_prompt=settings.negative_prompt,
                num_inference_steps=settings.steps,
                guidance_scale=settings.guidance_scale,
                generator=generator,
                width=settings.width, height=settings.height,
            ).images[0]


def generate_and_save(comparisons, generator, output_dir=Path("outputs")):
    run_dir = Path(output_dir) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    
    results = []
    for index, settings in enumerate(comparisons, start=1):
        image = generator.generate(settings)
        path = run_dir / f"{index:02d}_seed{settings.seed}_steps{settings.steps}_scale{settings.guidance_scale}.png"
        image.save(path)
        path.with_suffix(".json").write_text(
            json.dumps({"model_id": generator.model_id, **asdict(settings)},
                       ensure_ascii=False, indent=2), encoding="utf-8",
        )
        results.append(GenerationResult(path, settings))
    
    return results
