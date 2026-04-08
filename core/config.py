# ChromaManga Configuration
import os

# Project root directory (parent of core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inputs_dir = os.path.join(BASE_DIR, "data", "inputs")
outputs_dir = os.path.join(BASE_DIR, "data", "outputs")
outputs_colored_dir = os.path.join(BASE_DIR, "data", "outputs", "colored")
# 复用原项目的模型和 LoRA 文件
models_dir = r"E:\code\graduationProject\Theresa\models"
loras_dir = r"E:\code\graduationProject\Theresa\loras"

# ── Base Model ────────────────────────────────────────────────────────────────
# IMPORTANT: Use Illustrious-based model for LoRA compatibility
# Download from: https://civitai.com/models/795765/illustrious-xl
# base_model_id = "OnomaAIResearch/Illustrious-xl-early-release-v0"  # HuggingFace
# Use local .safetensors file:
base_model_id = os.path.join(models_dir, "Illustrious", "illustriousXL_v01.safetensors")

# ── ControlNet ────────────────────────────────────────────────────────────────
# Modes:
#   "union"   → controlnet-union-sdxl-1.0 with lineart_anime (recommended)
#   "scribble" → controlnet-scribble-sdxl-1.0 with adaptiveThreshold
controlnet_mode = "union"

controlnet_models = {
    "union": os.path.join(models_dir, "controlnet-union-sdxl-1.0"),
    "scribble": os.path.join(models_dir, "controlnet-scribble-sdxl-1.0"),
}

# ── LoRA Configuration ────────────────────────────────────────────────────────
# path   : LoRA file path (.safetensors)
# scale  : LoRA strength 0.0~1.0
# trigger: Trigger word for this LoRA
# prompt : Full prompt for this character (optional)
# refs   : Reference images directory for CLIP ReID (optional)
lora_configs = [
    {
        "path": r"E:\code\graduationProject\Theresa\loras\Eromanga_Sensei-multicharacter-illust.safetensors",
        "scale": 0.8,
        "trigger": "Sagiri",
        "refs": r"E:\code\graduationProject\Theresa\refs\sagiri",
        "prompt": (
            "Sagiri, masterpiece, best quality, anime coloring, flat color, cel shading, "
            "1girl, white hair, blunt bangs, long hair, pigtails, pink bows, "
            "light blue eyes, pink pajamas, white collar, clean lineart, no text"
        ),
    },
    {
        "path": r"E:\code\graduationProject\Theresa\loras\Eromanga_Sensei-multicharacter-illust.safetensors",
        "scale": 0.8,
        "trigger": "Masamune Izumi",
        "refs": r"E:\code\graduationProject\Theresa\refs\masamuune",
        "prompt": (
            "Masamune Izumi, masterpiece, best quality, anime coloring, flat color, cel shading, "
            "1boy, dark blue-green hair, messy hair, dark eyes, blue shirt, clean lineart, no text"
        ),
    },
    {
        "path": r"E:\code\graduationProject\Theresa\loras\Eromanga_Sensei-multicharacter-illust.safetensors",
        "scale": 0.8,
        "trigger": "Elf Yamada",
        "refs": r"E:\code\graduationProject\Theresa\refs\ElfYamada",
        "prompt": (
            "Elf Yamada, masterpiece, best quality, anime coloring, flat color, cel shading, "
            "1girl, blonde hair, long hair, wavy hair, red headband, amber eyes, pink dress, clean lineart, no text"
        ),
    },
    {
        "path": r"E:\code\graduationProject\Theresa\loras\Eromanga_Sensei-multicharacter-illust.safetensors",
        "scale": 0.8,
        "trigger": "Muramasa Senju",
        "refs": r"E:\code\graduationProject\Theresa\refs\MuramasaSenju",
        "prompt": (
            "Muramasa Senju, masterpiece, best quality, anime coloring, flat color, cel shading, "
            "1girl, purple hair, long hair, blunt bangs, purple eyes, yellow cardigan, clean lineart, no text"
        ),
    },
]

# ── Character ReID (CLIP) ─────────────────────────────────────────────────────
reid_enabled = True
reid_cache_path = r"E:\code\graduationProject\Theresa\models\reid_templates.npy"
reid_threshold = 0.75  # Confidence threshold

# ── Prompts ───────────────────────────────────────────────────────────────────
positive_prompt = (
    "masterpiece, best quality, anime coloring, flat color, cel shading, "
    "bright vibrant colors, clean lineart, 1girl, white hair, blue eyes, no text"
)

background_prompt = (
    "masterpiece, best quality, anime coloring, flat color, cel shading, "
    "clean bright background, white background, soft lighting, minimal details, clean lineart, no humans, no text"
)

negative_prompt = (
    "worst quality, low quality, lowres, text, watermark, blurry, jpeg artifacts, "
    "monochrome, grayscale, sketch, dark, gloomy, muddy colors, dirty background, gray tones"
)

# ── Inference Parameters ──────────────────────────────────────────────────────
num_inference_steps = 20      # Recommended: 20-28
guidance_scale = 6.0          # Recommended: 5.0-7.0
controlnet_scale = 1.1        # Lineart control strength
seed = 42                     # -1 for random

# ── Post-processing ───────────────────────────────────────────────────────────
blend_line_alpha = 0.15       # Lineart overlay strength (0~1)
output_size = None            # None=keep original, or (width, height)

# Resolution limits
min_resolution = 512          # Upscale if smaller
max_infer_resolution = 1024   # Downscale before inference if larger
downscale_before_infer = True # Enable inference resolution limit

# ── Upscaling (Real-ESRGAN) ───────────────────────────────────────────────────
upscale_enabled = True
upscale_model_name = "realesrgan-x4plus-anime"

# ── Debug ─────────────────────────────────────────────────────────────────────
save_debug_images = True

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Denoise methods: 
#   "none", "bilateral", "bilateral_strong", "median", "median_large",
#   "gaussian", "nlmeans", "morphology", "median_bilateral", "median_nlmeans"
denoise_method = "none"

# ── Speech Bubble Protection ──────────────────────────────────────────────────
# YOLO model for bubble detection (optional, None for OpenCV auto-detection)
bubble_yolo_model = r"E:\code\graduationProject\Theresa\models\bubble_detetor\comic-speech-bubble-detector.pt"
bubble_ellipse_scale = 1
bubble_text_shrink_x = 0.75
bubble_text_shrink_y = 0.90

# ── Person Detection ──────────────────────────────────────────────────────────
# Anime person detection model for regional colorization
person_detection_model = r"E:\code\graduationProject\Theresa\models\anime_person_detection\model.pt"

# Segmentation backend:
#   "sam2"           → General segmentation, fast but less accurate for anime
#   "anime_seg"      → Anime-specific segmentation, high edge accuracy (recommended)
#   "anime_seg_refine" → anime_seg + extra refinement, highest accuracy but slower
segmentation_backend = "anime_seg"
