"""Generation settings and comparison planning, independent of the UI and GPU."""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance_scale: float
    width: int
    height: int


def integer(value, label, minimum, maximum):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"{label} は整数で入力してください。") from None
    
    if not isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
        raise ValueError(f"{label} は {minimum}〜{maximum} の整数で入力してください。")
    return int(number)


def scale_value(value):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Guidance Scale は数値で入力してください。") from None

    if not isfinite(number) or not 0 <= number <= 20:
        raise ValueError("Guidance Scale は 0〜20 の有限の数値で入力してください。")
    return number


def build_comparisons(*, prompt, negative_prompt, seed, compare_mode,
                      fixed_scale, fixed_steps, scales_text, steps_text, width, height):
    
    if compare_mode not in {"guidance_scale", "steps"}:
        raise ValueError("比較対象を選択してください。")
    
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt を入力してください。")
    
    width = integer(width, "Width", 256, 768)
    height = integer(height, "Height", 256, 768)
    
    if width % 8 or height % 8:
        raise ValueError("Width と Height は8の倍数で入力してください。")
    
    seed = integer(seed, "Seed", 0, 2**32 - 1)
    text = scales_text if compare_mode == "guidance_scale" else steps_text
    parts = str(text or "").split(",")
    
    if any(not part.strip() for part in parts):
        raise ValueError("比較する値をカンマ区切りで入力してください。空の項目は使えません。")
    
    if len(parts) > 20:
        raise ValueError("一度に比較できる値は20個までです。")
    
    if compare_mode == "guidance_scale":
        values = [scale_value(part) for part in parts]
        steps = integer(fixed_steps, "Fixed Steps", 1, 100)
        scale = values[0]
        field = "guidance_scale"
    else:
        values = [integer(part, "Steps", 1, 100) for part in parts]
        steps = values[0]
        scale = scale_value(fixed_scale)
        field = "steps"
    
    base = GenerationSettings(prompt, negative_prompt or "", seed, steps, scale, width, height)
    
    return [replace(base, **{field: value}) for value in values]
