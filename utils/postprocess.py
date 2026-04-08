"""
Post-processing: blend original lineart back to colorized result
"""
import numpy as np
from PIL import Image


def blend_lineart(
    colored: Image.Image,
    lineart: Image.Image,
    alpha: float = 0.15,
) -> Image.Image:
    """
    Overlay lineart on colored image using Multiply blend mode
    
    Args:
        colored: Colorized RGB image
        lineart: White bg + black lines RGB image (same size as colored)
        alpha: Blend strength, 0=no blend, 1=full overlay, recommended 0.1~0.2
    
    Returns:
        Blended RGB PIL Image
    """
    if alpha <= 0:
        return colored

    colored_np = np.array(colored).astype(np.float32)
    line_np = np.array(lineart.resize(colored.size)).astype(np.float32)

    # Multiply blend: darken colored image with lineart
    multiplied = colored_np * (line_np / 255.0)

    result = colored_np * (1 - alpha) + multiplied * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)
