"""
Image preprocessing: denoising + lineart extraction

Denoise methods (via config.denoise_method):
  - "none"              : No denoising
  - "bilateral"         : Bilateral filter, edge-preserving
  - "median"            : Median filter 5×5, good for screentones
  - "nlmeans"           : Non-Local Means, best quality (default)
  - "median_bilateral"  : Combo: median → bilateral

Lineart extraction modes (via config.controlnet_mode):
  - "union"   : LineartAnimeDetector (recommended)
  - "scribble": OpenCV adaptiveThreshold (lightweight)
"""
import cv2
import numpy as np
from PIL import Image


def _to_gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def denoise_none(image: Image.Image) -> Image.Image:
    return image


def denoise_bilateral(image: Image.Image) -> Image.Image:
    gray = _to_gray(image)
    result = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return _to_pil(result)


def denoise_median(image: Image.Image) -> Image.Image:
    gray = _to_gray(image)
    result = cv2.medianBlur(gray, 5)
    return _to_pil(result)


def denoise_nlmeans(image: Image.Image) -> Image.Image:
    gray = _to_gray(image)
    result = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    return _to_pil(result)


def denoise_median_bilateral(image: Image.Image) -> Image.Image:
    gray = _to_gray(image)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return _to_pil(gray)


def denoise_bilateral_strong(image: Image.Image) -> Image.Image:
    """Bilateral filter (strong): 3 iterations"""
    gray = _to_gray(image)
    for _ in range(3):
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=100, sigmaSpace=100)
    return _to_pil(gray)


def denoise_median_large(image: Image.Image) -> Image.Image:
    """Median filter 7×7: stronger screentone removal"""
    gray = _to_gray(image)
    result = cv2.medianBlur(gray, 7)
    return _to_pil(result)


def denoise_gaussian(image: Image.Image) -> Image.Image:
    """Gaussian blur 5×5"""
    gray = _to_gray(image)
    result = cv2.GaussianBlur(gray, (5, 5), 0)
    return _to_pil(result)


def denoise_morphology(image: Image.Image) -> Image.Image:
    """Morphology: binarization + open/close operations"""
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return _to_pil(result)


def denoise_median_nlmeans(image: Image.Image) -> Image.Image:
    """Combo: median → NLMeans"""
    gray = _to_gray(image)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return _to_pil(gray)


_DENOISE_METHODS = {
    "none": denoise_none,
    "bilateral": denoise_bilateral,
    "bilateral_strong": denoise_bilateral_strong,
    "median": denoise_median,
    "median_large": denoise_median_large,
    "gaussian": denoise_gaussian,
    "nlmeans": denoise_nlmeans,
    "morphology": denoise_morphology,
    "median_bilateral": denoise_median_bilateral,
    "median_nlmeans": denoise_median_nlmeans,
}


def apply_denoise(image: Image.Image, method: str = "nlmeans") -> Image.Image:
    """Apply denoising to image"""
    fn = _DENOISE_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown denoise method: '{method}', available: {list(_DENOISE_METHODS)}")
    return fn(image)


def extract_lineart_anime(image: Image.Image) -> Image.Image:
    """Extract anime lineart using LineartAnimeDetector (white bg, black lines)"""
    from controlnet_aux import LineartAnimeDetector
    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    result = detector(image,
                      detect_resolution=max(image.size),
                      image_resolution=min(image.size))
    if result.size != image.size:
        result = result.resize(image.size, Image.LANCZOS)
    return result.convert("RGB")


def extract_lineart_scribble(image: Image.Image, blur_radius: int = 3) -> Image.Image:
    """Extract lineart using OpenCV adaptiveThreshold (white bg, black lines)"""
    gray = np.array(image.convert("L"))
    if blur_radius > 0:
        gray = cv2.GaussianBlur(gray, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
    line = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=7,
    )
    return Image.fromarray(line).convert("RGB")


def extract_lineart(image: Image.Image, mode: str = "union", denoise_method: str = "none") -> Image.Image:
    """Unified lineart extraction interface"""
    if denoise_method != "none":
        image = apply_denoise(image, denoise_method)
    if mode == "union":
        return extract_lineart_anime(image)
    else:
        return extract_lineart_scribble(image)


def preprocess_for_controlnet(
    image: Image.Image,
    target_size: tuple | None = None,
    mode: str = "union",
    denoise_method: str = "none",
) -> tuple:
    """
    Prepare input for ControlNet: denoise → resize → lineart extraction
    
    Returns:
        (lineart_rgb, original_rgb)
    """
    from core import config as _cfg
    _mode = mode or getattr(_cfg, "controlnet_mode", "union")
    _denoise = denoise_method or getattr(_cfg, "denoise_method", "none")

    if target_size:
        image = image.resize(target_size, Image.LANCZOS)

    # Align to 64 multiples
    w, h = image.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    image = image.resize((w, h), Image.LANCZOS)

    # Denoise
    if _denoise != "none":
        print(f"  Denoise method: {_denoise}")
        image = apply_denoise(image, _denoise)

    # Extract lineart
    if _mode == "union":
        lineart = extract_lineart_anime(image)
    else:
        lineart = extract_lineart_scribble(image)

    return lineart, image.convert("RGB")
