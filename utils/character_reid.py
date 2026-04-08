"""
CLIP-based character re-identification
Match person crops to character reference templates using cosine similarity
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# CLIP singleton
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        try:
            import clip
            import torch
        except ImportError:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [ReID] Loading CLIP ViT-L/14 ({_clip_device})...")
        _clip_model, _clip_preprocess = clip.load("ViT-L/14", device=_clip_device)
        _clip_model.eval()
        print("  [ReID] CLIP loaded")
    return _clip_model, _clip_preprocess, _clip_device


def extract_feature(image: Image.Image) -> np.ndarray:
    """Extract CLIP feature vector, grayscale preprocessing to eliminate color bias"""
    import torch
    model, preprocess, device = _get_clip()
    gray = image.convert("L").convert("RGB")
    tensor = preprocess(gray).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().squeeze()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_template_from_dir(ref_dir: str) -> np.ndarray | None:
    """Extract features from all images in directory, average as character template"""
    imgs = [f for f in os.listdir(ref_dir) if Path(f).suffix.lower() in IMAGE_EXTS]
    if not imgs:
        return None
    feats = []
    for name in imgs:
        try:
            img = Image.open(os.path.join(ref_dir, name)).convert("RGB")
            feats.append(extract_feature(img))
        except Exception as e:
            print(f"  [ReID] Skip {name}: {e}")
    if not feats:
        return None
    avg = np.mean(feats, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg


class CharacterReID:
    """
    Character re-identification using CLIP
    Build template library from lora_configs refs, identify characters and return LoRA index
    """

    def __init__(self, lora_configs: list, cache_path: str | None = None):
        self.lora_configs = lora_configs
        self.cache_path = cache_path
        self.templates: dict[int, np.ndarray] = {}
        self._build()

    def _build(self):
        # Try load from cache
        if self.cache_path and os.path.exists(self.cache_path):
            cached = np.load(self.cache_path, allow_pickle=True).item()
            self.templates = cached
            names = {i: self.lora_configs[i].get("trigger", f"lora_{i}")
                     for i in self.templates}
            print(f"  [ReID] Loaded templates from cache: {names}")
            return

        # Build from refs directories
        for i, lora_cfg in enumerate(self.lora_configs):
            refs = lora_cfg.get("refs", None)
            if not refs or not os.path.isdir(refs):
                continue
            trigger = lora_cfg.get("trigger", f"lora_{i}")
            print(f"  [ReID] Building template [{trigger}], refs: {refs}")
            tmpl = build_template_from_dir(refs)
            if tmpl is not None:
                self.templates[i] = tmpl
                print(f"  [ReID] ✓ [{trigger}] template built")
            else:
                print(f"  [ReID] ✗ [{trigger}] refs directory empty, skipped")

        if not self.templates:
            print("  [ReID] Warning: No character templates, ReID unavailable")
            return

        # Save cache
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, self.templates)
            print(f"  [ReID] Template cache saved: {self.cache_path}")

    def is_available(self) -> bool:
        return len(self.templates) > 0

    def identify(self, image: Image.Image) -> tuple[int | None, str, float]:
        """
        Identify character in image, return best matching LoRA index
        
        Returns:
            (lora_index, trigger_name, similarity)
            lora_index is None if identification failed or no templates
        """
        if not self.is_available():
            return None, "unknown", 0.0

        feat = extract_feature(image)
        best_idx, best_sim = None, -1.0
        for lora_idx, tmpl in self.templates.items():
            sim = _cosine_sim(feat, tmpl)
            if sim > best_sim:
                best_sim = sim
                best_idx = lora_idx

        trigger = self.lora_configs[best_idx].get("trigger", f"lora_{best_idx}") if best_idx is not None else "unknown"
        return best_idx, trigger, best_sim

    def release(self):
        """Release CLIP model to free GPU memory"""
        global _clip_model, _clip_preprocess, _clip_device
        if _clip_model is not None:
            import torch
            _clip_model.cpu()
            del _clip_model
            _clip_model = None
            _clip_preprocess = None
            _clip_device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  [ReID] CLIP model released")
