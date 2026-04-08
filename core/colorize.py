"""
ChromaManga - AI-Powered Manga Colorization
Regional Prompter approach: One-pass inference with region-specific prompts.

Pipeline:
  1. YOLO person detection
  2. CLIP ReID character identification → region prompts
  3. Build region masks (background + person bboxes)
  4. Encode all region text embeddings
  5. Inject RegionalAttnProcessor, single ControlNet+SDXL inference
  6. Restore speech bubble text, save results

Usage:
    python colorize.py --input inputs/manga.png
"""

import argparse
import os
import sys
import random
import time
import numpy as np
from PIL import Image
import cv2
import torch

# Fix basicsr compatibility with newer torchvision
from unittest.mock import MagicMock
sys.modules.setdefault('torchvision.transforms.functional_tensor', MagicMock())

from . import config
from utils.preprocess import preprocess_for_controlnet
from utils.postprocess import blend_lineart
from utils.character_reid import CharacterReID
from utils.regional_attention import set_regional_attn, reset_attn
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Real-ESRGAN Upscaler
# ─────────────────────────────────────────────────────────────────────────────

_upsampler = None

def get_upsampler():
    """Lazy load Real-ESRGAN upsampler"""
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    model_path = os.path.join(config.models_dir, "realesrgan", "realesrgan-x4plus-anime.pth")

    if not os.path.exists(model_path):
        print("  Downloading Real-ESRGAN anime model...")
        import urllib.request
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        os.makedirs(os.path.join(config.models_dir, "realesrgan"), exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        print("  Download complete")

    _upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        tile=256, tile_pad=10, pre_pad=0,
        half=torch.cuda.is_available(),
    )
    return _upsampler


def upscale_to_target(image: Image.Image, target_size: tuple) -> Image.Image:
    """Upscale with Real-ESRGAN 4x then resize to target"""
    tw, th = target_size
    if image.width >= tw and image.height >= th:
        return image.resize(target_size, Image.LANCZOS)

    print(f"  Upscaling: {image.size} → 4x → resize to {target_size}")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    output, _ = get_upsampler().enhance(img_bgr, outscale=4)
    result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return result.resize(target_size, Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# Speech Bubble Text Extraction & Restoration
# ─────────────────────────────────────────────────────────────────────────────

def detect_bubble_boxes(image: Image.Image, model_path: str):
    """YOLO bubble detection, returns bbox list"""
    model = YOLO(model_path)
    det = model(image, verbose=False)[0]
    boxes = []
    if det.boxes is not None:
        for b in det.boxes.xyxy.cpu().numpy():
            boxes.append(tuple(map(int, b[:4])))
    return boxes


def _ellipse_mask(bx1, by1, bx2, by2):
    """Generate ellipse mask within bbox"""
    rh, rw = by2 - by1, bx2 - bx1
    if rh <= 0 or rw <= 0:
        return np.zeros((max(rh, 1), max(rw, 1)), dtype=np.float32)
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    cy, cx = (rh - 1) / 2.0, (rw - 1) / 2.0
    ry, rx = rh / 2.0, rw / 2.0
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return (ellipse <= 1.0).astype(np.float32)


def extract_text_by_threshold(img_gray, boxes, threshold=210,
                              shrink_x=0.60, shrink_y=0.85):
    """Extract text mask from bubble regions"""
    h, w = img_gray.shape
    mask = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = (x2 - x1) * shrink_x / 2
        bh = (y2 - y1) * shrink_y / 2
        bx1 = max(0, int(cx - bw))
        by1 = max(0, int(cy - bh))
        bx2 = min(w, int(cx + bw))
        by2 = min(h, int(cy + bh))
        if bx1 >= bx2 or by1 >= by2:
            continue

        roi = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        text_weight = np.clip((threshold - roi) / threshold, 0, 1)
        emask = _ellipse_mask(bx1, by1, bx2, by2)
        text_weight *= emask
        mask[by1:by2, bx1:bx2] = np.maximum(mask[by1:by2, bx1:bx2], text_weight)

    return mask


def stamp_text(colored_np, text_mask_float, boxes, shrink_x=0.60, shrink_y=0.85):
    """Inpaint bubble regions then stamp black text"""
    h, w = colored_np.shape[:2]
    result = colored_np.copy()

    # Build inpaint mask
    base_inpaint = (text_mask_float * 255).astype(np.uint8)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    base_inpaint = cv2.dilate(base_inpaint, k_dilate, iterations=2)

    # Detect residual artifacts in colored image
    colored_gray = cv2.cvtColor(colored_np, cv2.COLOR_RGB2GRAY)
    residual_mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = (x2 - x1) * shrink_x / 2
        bh = (y2 - y1) * shrink_y / 2
        bx1 = max(0, int(cx - bw))
        by1 = max(0, int(cy - bh))
        bx2 = min(w, int(cx + bw))
        by2 = min(h, int(cy + bh))
        if bx1 >= bx2 or by1 >= by2:
            continue

        emask = _ellipse_mask(bx1, by1, bx2, by2)
        roi = colored_gray[by1:by2, bx1:bx2]
        median_val = np.median(roi)
        dark_thresh = max(100, median_val - 30)
        dark_pixels = (roi < dark_thresh).astype(np.uint8) * 255
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_pixels = cv2.dilate(dark_pixels, k_small, iterations=1)
        dark_pixels = (dark_pixels.astype(np.float32) * emask).astype(np.uint8)
        residual_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
            residual_mask[by1:by2, bx1:bx2], dark_pixels)

    # Merge masks
    inpaint_mask = cv2.bitwise_or(base_inpaint, residual_mask)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw = (x2 - x1) * shrink_x / 2
        bh = (y2 - y1) * shrink_y / 2
        bx1 = max(0, int(cx - bw))
        by1 = max(0, int(cy - bh))
        bx2 = min(w, int(cx + bw))
        by2 = min(h, int(cy + bh))
        emask_uint8 = (_ellipse_mask(bx1, by1, bx2, by2) * 255).astype(np.uint8)
        region_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
            region_mask[by1:by2, bx1:bx2], emask_uint8)
    inpaint_mask = cv2.bitwise_and(inpaint_mask, region_mask)

    # Inpaint
    if np.any(inpaint_mask > 0):
        result = cv2.inpaint(result, inpaint_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

    # Stamp black text
    m3 = np.stack([text_mask_float] * 3, axis=-1)
    result = (result.astype(np.float32) * (1 - m3)).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        UniPCMultistepScheduler,
    )

    print("Loading ControlNet...")
    _cn_id = config.controlnet_models.get(config.controlnet_mode,
             config.controlnet_models["scribble"])
    if not os.path.isdir(_cn_id):
        _cn_id = "xinsir/controlnet-union-sdxl-1.0" if config.controlnet_mode == "union" \
                 else "xinsir/controlnet-scribble-sdxl-1.0"
    print(f"  Mode: {config.controlnet_mode}  Path: {_cn_id}")
    controlnet = ControlNetModel.from_pretrained(
        _cn_id, torch_dtype=torch.float16, cache_dir=config.models_dir
    )

    print(f"Loading base model: {config.base_model_id}")
    if os.path.isfile(config.base_model_id) and config.base_model_id.endswith(".safetensors"):
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            config.base_model_id, controlnet=controlnet, torch_dtype=torch.float16,
            original_config="https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml",
        )
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config.base_model_id, controlnet=controlnet, torch_dtype=torch.float16,
            cache_dir=config.models_dir, safety_checker=None,
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Load LoRAs
    loaded_paths = {}
    for i, lora_cfg in enumerate(config.lora_configs):
        lora_path = lora_cfg["path"]
        if not os.path.exists(lora_path):
            continue
        norm_path = os.path.normpath(lora_path)
        if norm_path not in loaded_paths:
            adapter_name = f"lora_{i}"
            print(f"  Loading LoRA [{adapter_name}]: {os.path.basename(lora_path)}")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded_paths[norm_path] = adapter_name
        else:
            print(f"  Skipping duplicate LoRA file (already loaded as {loaded_paths[norm_path]}): {os.path.basename(lora_path)}")

    # Activate all adapters
    if loaded_paths:
        adapter_names = list(loaded_paths.values())
        weights = []
        for name in adapter_names:
            idx = int(name.split("_")[1])
            weights.append(config.lora_configs[idx]["scale"])
        pipe.set_adapters(adapter_names, adapter_weights=weights)
        print(f"  LoRA activated: {adapter_names}  weights: {weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Enable xformers or SDPA
    if torch.cuda.is_available():
        try:
            import xformers
            pipe.enable_xformers_memory_efficient_attention()
            print("  xformers acceleration enabled")
        except (ImportError, ModuleNotFoundError):
            from packaging import version
            if version.parse(torch.__version__) >= version.parse("2.0"):
                pipe.unet.set_attn_processor(
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION
                    if hasattr(torch.nn.attention, "SDPBackend") else None
                )
                print("  PyTorch 2.0 SDPA acceleration enabled")

    # Initialize ReID
    reid = None
    if getattr(config, "reid_enabled", False):
        print("\nInitializing Character ReID (CLIP)...")
        reid = CharacterReID(
            config.lora_configs,
            cache_path=getattr(config, "reid_cache_path", None),
        )
        reid.release()

    print(f"Pipeline ready, device: {device}")
    return pipe, reid


# ─────────────────────────────────────────────────────────────────────────────
# Region Mask Building
# ─────────────────────────────────────────────────────────────────────────────

def build_region_masks(image_size: tuple, person_bboxes: list, latent_scale: int = 8):
    """Build region masks in latent space"""
    W, H = image_size
    H_lat, W_lat = H // latent_scale, W // latent_scale

    bg_mask = torch.ones(1, 1, H_lat, W_lat)
    person_masks = []
    
    for bbox in person_bboxes:
        x1, y1, x2, y2 = bbox
        lx1 = max(0, x1 // latent_scale)
        ly1 = max(0, y1 // latent_scale)
        lx2 = min(W_lat, x2 // latent_scale)
        ly2 = min(H_lat, y2 // latent_scale)

        pmask = torch.zeros(1, 1, H_lat, W_lat)
        pmask[0, 0, ly1:ly2, lx1:lx2] = 1.0
        person_masks.append(pmask)
        bg_mask[0, 0, ly1:ly2, lx1:lx2] = 0.0

    return bg_mask, person_masks


def visualize_region_masks(image: Image.Image, bg_mask: torch.Tensor,
                           person_masks: list, person_labels: list) -> Image.Image:
    """Visualize region masks overlaid on image"""
    vis = np.array(image.convert("RGB")).copy().astype(np.float32)
    H, W = vis.shape[:2]

    colors = [
        (80, 180, 255),   # Blue: person1
        (80, 255, 120),   # Green: person2
        (255, 200, 80),   # Yellow: person3
        (200, 80, 255),   # Purple: person4
    ]

    for i, (pmask, label) in enumerate(zip(person_masks, person_labels)):
        m = torch.nn.functional.interpolate(pmask.float(), size=(H, W), mode="nearest")
        m = m.squeeze().numpy()
        color = colors[i % len(colors)]
        for c, v in enumerate(color):
            vis[:, :, c] = np.where(m > 0.5, vis[:, :, c] * 0.5 + v * 0.5, vis[:, :, c])

        ys, xs = np.where(m > 0.5)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis.astype(np.uint8), label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return Image.fromarray(vis.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Text Encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_prompts(pipe, prompts: list[str], device: str):
    """Batch encode prompts, returns list of (cond, neg, pooled, neg_pooled)"""
    embeddings = []
    for prompt in prompts:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        
        (
            neg_embeds,
            _,
            neg_pooled,
            _,
        ) = pipe.encode_prompt(
            prompt=config.negative_prompt,
            prompt_2=config.negative_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        embeddings.append((prompt_embeds, neg_embeds,
                           pooled_prompt_embeds, neg_pooled))
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Main Colorization Function
# ─────────────────────────────────────────────────────────────────────────────

def colorize_regional(pipe, input_path: str, output_path: str, reid=None):
    save_debug = getattr(config, "save_debug_images", True)
    debug_dir = os.path.join(
        os.path.dirname(output_path), "debug",
        os.path.splitext(os.path.basename(output_path))[0]
    )
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"\n[Regional] Input: {input_path}")
        print(f"[DEBUG] Intermediate results: {debug_dir}")
    else:
        print(f"\n[Regional] Input: {input_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    image = Image.open(input_path).convert("RGB")
    original_size = image.size

    # Resolution check
    min_res = getattr(config, "min_resolution", 512)
    if min_res and max(image.size) < min_res:
        scale = min_res / max(image.size)
        new_w, new_h = int(image.width * scale), int(image.height * scale)
        print(f"  Upscaling low-res: {original_size} → ({new_w},{new_h})")
        image = image.resize((new_w, new_h), Image.LANCZOS)

    if save_debug:
        image.save(os.path.join(debug_dir, "00_input.png"))

    # Bubble detection
    print("Detecting speech bubbles...")
    bubble_model = getattr(config, "bubble_yolo_model", None)
    bubble_boxes = []
    if bubble_model and os.path.exists(bubble_model):
        bubble_boxes = detect_bubble_boxes(image, bubble_model)
        print(f"  Detected {len(bubble_boxes)} bubbles")
    else:
        print("  Warning: bubble_yolo_model not configured, skipping")

    if save_debug and bubble_boxes:
        bbox_vis = np.array(image).copy()
        for x1, y1, x2, y2 in bubble_boxes:
            cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (255, 80, 80), 2)
        Image.fromarray(bbox_vis).save(os.path.join(debug_dir, "01_bubble_bbox.png"))

    # Extract text mask
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    shrink_x = getattr(config, "bubble_text_shrink_x", 0.60)
    shrink_y = getattr(config, "bubble_text_shrink_y", 0.85)
    text_mask_float = extract_text_by_threshold(
        img_gray, bubble_boxes, threshold=210,
        shrink_x=shrink_x, shrink_y=shrink_y,
    )
    if save_debug:
        Image.fromarray((text_mask_float * 255).astype(np.uint8)).save(
            os.path.join(debug_dir, "01b_text_mask.png"))

    # Extract lineart
    print("Extracting lineart...")
    lineart, _ = preprocess_for_controlnet(
        image, config.output_size,
        denoise_method=getattr(config, "denoise_method", "none"),
    )
    if save_debug:
        lineart.save(os.path.join(debug_dir, "02_lineart.png"))

    # Inference resolution limit
    lineart_original_size = lineart.size
    downscale_enabled = getattr(config, "downscale_before_infer", False)
    MAX_INFER_SIDE = getattr(config, "max_infer_resolution", 1024)
    if downscale_enabled and min(lineart.size) > MAX_INFER_SIDE:
        scale = MAX_INFER_SIDE / min(lineart.size)
        infer_w = (int(lineart.width * scale) // 64) * 64
        infer_h = (int(lineart.height * scale) // 64) * 64
        print(f"  Inference resolution limit: {lineart.size} → ({infer_w},{infer_h})")
        lineart = lineart.resize((infer_w, infer_h), Image.LANCZOS)
    else:
        print(f"  Inference resolution: {lineart.size}")

    # YOLO person detection
    print("Detecting persons...")
    yolo = YOLO(config.person_detection_model)
    results = yolo(image, verbose=False)[0]
    persons_orig = []
    if results.boxes is not None:
        for box in results.boxes.xyxy.cpu().numpy():
            persons_orig.append(tuple(map(int, box[:4])))
    print(f"  Detected {len(persons_orig)} persons")

    # Scale bboxes to inference resolution
    scale_x = lineart.width / image.width
    scale_y = lineart.height / image.height
    persons = []
    for (x1, y1, x2, y2) in persons_orig:
        persons.append((
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y),
        ))

    # ReID and prompt assignment
    bbox_vis = np.array(image).copy()
    vis_colors = [(255, 80, 80), (80, 180, 255), (80, 255, 120), (255, 200, 80)]
    region_prompts = []
    region_labels = []
    threshold = getattr(config, "reid_threshold", 0.75)

    for i, (bbox_orig, bbox_infer) in enumerate(zip(persons_orig, persons)):
        x1, y1, x2, y2 = bbox_orig
        pad = 16
        W_img, H_img = image.size
        rx1 = max(0, x1 - pad); ry1 = max(0, y1 - pad)
        rx2 = min(W_img, x2 + pad); ry2 = min(H_img, y2 + pad)
        crop = image.crop((rx1, ry1, rx2, ry2))

        c = vis_colors[i % len(vis_colors)]

        if reid and reid.is_available():
            lora_idx, trigger, sim = reid.identify(crop)
            print(f"  [ReID] Person{i+1}: {trigger}  similarity: {sim:.4f}")
            if lora_idx is not None and sim >= threshold:
                prompt = config.lora_configs[lora_idx].get("prompt") or config.positive_prompt
                label = trigger
            else:
                prompt = config.positive_prompt
                label = f"P{i+1}(unknown)"
        else:
            prompt = config.positive_prompt
            label = f"P{i+1}"

        region_prompts.append(prompt)
        region_labels.append(label)
        cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), c, 3)
        cv2.putText(bbox_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        print(f"  Person{i+1} [{label}] prompt: {prompt[:60]}...")

    if reid:
        reid.release()

    if save_debug:
        Image.fromarray(bbox_vis).save(os.path.join(debug_dir, "03_bboxes_reid.png"))

    # Build region masks
    print("Building region masks...")
    bg_mask, person_masks = build_region_masks(lineart.size, persons)
    if save_debug:
        region_vis = visualize_region_masks(image, 
            *build_region_masks(image.size, persons_orig), region_labels)
        region_vis.save(os.path.join(debug_dir, "04_region_mask.png"))

    # Encode prompts
    bg_prompt = getattr(config, "background_prompt", config.positive_prompt)
    all_prompts = [bg_prompt] + region_prompts
    print(f"Encoding prompts ({len(all_prompts)} regions)...")
    print(f"  Background: {bg_prompt[:60]}...")
    all_embeds = encode_prompts(pipe, all_prompts, device)

    base_cond, base_neg, base_pooled, base_neg_pooled = all_embeds[0]
    person_cond_embeds = [e[0] for e in all_embeds[1:]]

    # Inject Regional Attention
    if persons:
        latent_h = lineart.height // 8
        latent_w = lineart.width // 8
        print(f"Injecting Regional Attention Processor... (latent: {latent_h}×{latent_w})")
        set_regional_attn(pipe, person_masks, person_cond_embeds, base_cond, latent_h, latent_w)

    # Inference
    print("Running inference...")
    t0 = time.time()

    _seed = config.seed if config.seed != -1 else random.randint(0, 2**32 - 1)
    gen = torch.Generator(device=device).manual_seed(_seed)
    print(f"  Size: {lineart.size}  Seed: {_seed}")

    result = pipe(
        prompt_embeds=base_cond,
        negative_prompt_embeds=base_neg,
        pooled_prompt_embeds=base_pooled,
        negative_pooled_prompt_embeds=base_neg_pooled,
        image=lineart,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        controlnet_conditioning_scale=config.controlnet_scale,
        generator=gen,
        width=lineart.width,
        height=lineart.height,
    ).images[0]

    print(f"  Inference time: {time.time() - t0:.1f}s")

    if persons:
        reset_attn(pipe)

    if save_debug:
        result.save(os.path.join(debug_dir, "05_colored_raw.png"))

    # Resize back to lineart original size
    if result.size != lineart_original_size:
        use_upscale = getattr(config, "upscale_enabled", False) and \
                      getattr(config, "downscale_before_infer", False)
        if use_upscale:
            result = upscale_to_target(result, lineart_original_size)
        else:
            print(f"  Resizing back: {result.size} → {lineart_original_size}")
            result = result.resize(lineart_original_size, Image.LANCZOS)
        lineart = lineart.resize(lineart_original_size, Image.LANCZOS)

    # Blend lineart
    result = blend_lineart(result, lineart, alpha=config.blend_line_alpha)
    if save_debug:
        result.save(os.path.join(debug_dir, "06_lineart_blend.png"))

    # Resize back to original size
    if result.size != original_size:
        print(f"  Resizing to original: {result.size} → {original_size}")
        result = result.resize(original_size, Image.LANCZOS)

    # Restore speech bubble text
    if bubble_boxes:
        print("Restoring speech bubble text...")
        result_np = np.array(result)
        result_np = stamp_text(result_np, text_mask_float, bubble_boxes,
                               shrink_x=shrink_x, shrink_y=shrink_y)
        result = Image.fromarray(result_np)
    if save_debug:
        result.save(os.path.join(debug_dir, "07_text_restored.png"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path}")
    if save_debug:
        print(f"[DEBUG] Intermediate results: {debug_dir}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChromaManga - AI Manga Colorization")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    stem = os.path.splitext(os.path.basename(args.input))[0]
    output = args.output or os.path.join(
        config.outputs_colored_dir, f"{stem}_colored.png"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    pipe, reid = build_pipeline()
    colorize_regional(pipe, args.input, output, reid=reid)


if __name__ == "__main__":
    main()
