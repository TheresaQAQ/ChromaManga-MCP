"""Mock 工具集：在没有 GPU / 模型文件的机器上让 Agent 跑通完整循环。

**输出格式严格对齐真实 MCP Server (`mcp_server/chromamanga_mcp_server.py`)**：
- 工具名 / 参数（含默认值与 description）/ 返回文本字段名 / 错误消息 与 real 100% 一致
- task_id 用标准 uuid4，与 real 模式无法从字符串区分
- 仅"图片文件内容"是假的（复制 examples/ 下的示例图）
- run_inference 用 asyncio.sleep(3) 模拟耗时，避免真的依赖 SDXL/ControlNet/LoRA

切换方式：`.env.agent` 里设 `AGENT_MCP_MODE=mock`。
切回 real 时：mock 这边的字段名/格式不变，前端和 SYSTEM_PROMPT 无需改动。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

from . import config

logger = logging.getLogger(__name__)


_STEPS_ORDER = [
    "lineart_extraction",
    "person_detection",
    "character_identification",
    "bubble_detection",
    "mask_generation",
    "inference",
    "postprocess",
]

_EXAMPLES_DIR = config.PROJECT_ROOT / "examples" / "c1a670e964087446163d32eeab823613_regional"
# stage 文件名 → 仓库内的真实示例图（mock 用作各阶段产物以便前端缩略图能命中真实文件）
_STAGE_SOURCE: dict[str, Path] = {
    "01_lineart":          _EXAMPLES_DIR / "02_lineart.png",
    "02_person_detection": _EXAMPLES_DIR / "03_bboxes_reid.png",
    "04_region_mask":      _EXAMPLES_DIR / "04_region_mask.png",
    "05_colored_raw":      _EXAMPLES_DIR / "05_colored_raw.png",
    "06_lineart_blend":    _EXAMPLES_DIR / "06_lineart_blend.png",
    "07_final":            _EXAMPLES_DIR / "07_text_restored.png",
}


@dataclass
class _MockTask:
    """对齐 real 的 `core.task_manager.Task` 关键字段（字段名一致）。"""

    STEPS_ORDER = _STEPS_ORDER  # type: list[str]

    task_id: str
    image_path: str
    task_name: str
    image_size: tuple[int, int]
    output_dir: Path
    status: str = "created"
    current_step: str | None = None
    steps_completed: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def get_progress(self) -> float:
        return len(self.steps_completed) / len(_STEPS_ORDER)

    def mark_step_complete(self, step: str, result: Any) -> None:
        if step not in self.steps_completed:
            self.steps_completed.append(step)
        self.results[step] = result
        next_step = next((s for s in _STEPS_ORDER if s not in self.steps_completed), None)
        self.current_step = next_step
        self.status = "completed" if next_step is None else "processing"

    def reset_to_step(self, step: str) -> list[str]:
        if step not in _STEPS_ORDER:
            return []
        target_idx = _STEPS_ORDER.index(step)
        cleared: list[str] = []
        for s in _STEPS_ORDER[target_idx:]:
            if s in self.steps_completed:
                self.steps_completed.remove(s)
                cleared.append(s)
            self.results.pop(s, None)
        self.current_step = step
        self.status = "processing"
        return cleared


_TASKS: dict[str, _MockTask] = {}

_MOCK_CONFIG = {
    "base_model": "illustriousXL_v01.safetensors",
    "controlnet_mode": "union",
    "lora_configs": [
        {"index": 0, "name": "Sagiri", "scale": 0.8},
        {"index": 1, "name": "Masamune Izumi", "scale": 0.8},
        {"index": 2, "name": "Elf Yamada", "scale": 0.8},
        {"index": 3, "name": "Muramasa Senju", "scale": 0.8},
    ],
    "inference_params": {
        "num_inference_steps": 20,
        "guidance_scale": 6.0,
        "controlnet_scale": 1.1,
        "seed": 42,
    },
    "reid_enabled": True,
    "reid_threshold": 0.75,
}


def _get_task(task_id: str) -> _MockTask | None:
    return _TASKS.get(task_id)


def _copy_as_stage(task: _MockTask, stage_name: str) -> str:
    """把 examples/ 里该阶段的示例图复制到任务输出目录；找不到时回退到原图。"""
    dst = task.output_dir / f"{stage_name}.png"
    if dst.exists():
        return str(dst.resolve())

    src = _STAGE_SOURCE.get(stage_name)
    if src and src.exists():
        try:
            shutil.copyfile(src, dst)
            return str(dst.resolve())
        except OSError as e:
            logger.warning("复制示例图 %s 失败: %s", stage_name, e)

    try:
        shutil.copyfile(task.image_path, dst)
    except OSError as e:
        logger.warning("复制原图也失败 (%s): %s", stage_name, e)
        return task.image_path
    return str(dst.resolve())


def _read_image_size(image_path: str) -> tuple[int, int]:
    """读取图片尺寸，仅依赖 PIL（轻量级，无需 GPU）。"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (1024, 1536)


# ─── Task Management ───────────────────────────────────────────────────

@tool
def create_task(image_path: str, task_name: str | None = None) -> str:
    """Create a new manga colorization task by providing an image path."""
    if not os.path.exists(image_path):
        return f"✗ Failed to create task: Image not found: {image_path}"

    task_id = str(uuid.uuid4())
    output_dir = config.PROJECT_ROOT / "data" / "outputs" / "colored" / "debug" / f"{Path(image_path).stem}_colored"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = _read_image_size(image_path)
    name = task_name or Path(image_path).stem

    task = _MockTask(
        task_id=task_id,
        image_path=str(Path(image_path).resolve()),
        task_name=name,
        image_size=image_size,
        output_dir=output_dir,
    )
    _TASKS[task_id] = task

    return (
        f"✓ Task created successfully\n"
        f"\n"
        f"Task ID: {task_id}\n"
        f"Task Name: {name}\n"
        f"Image: {os.path.basename(image_path)}\n"
        f"Size: {image_size[0]}x{image_size[1]}\n"
        f"Output Directory: {output_dir}\n"
        f"\n"
        f"Next step: Run extract_lineart to begin the colorization pipeline.\n"
    )


@tool
def get_task_status(task_id: str) -> str:
    """Get the current status and progress of a colorization task."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    progress = task.get_progress()
    response = (
        f"Task Status: {task.task_id}\n"
        f"\n"
        f"Status: {task.status}\n"
        f"Progress: {progress:.1%} ({len(task.steps_completed)}/{len(_STEPS_ORDER)} steps)\n"
        f"Current Step: {task.current_step or 'Completed'}\n"
        f"\n"
        f"Steps Completed:\n"
    )
    for step in task.steps_completed:
        response += f"  ✓ {step}\n"
    if task.error:
        response += f"\nError: {task.error}"
    return response


@tool
def get_task_result(task_id: str, include_intermediates: bool = True) -> str:
    """Get the final colorized image and all intermediate results."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    response = f"Task Results: {task.task_id}\n\nStatus: {task.status}\n"
    if "postprocess" in task.results:
        final_path = task.results["postprocess"].get("final_image")
        response += f"\n✓ Final Image: {final_path}\n"
    if include_intermediates and task.results:
        response += "\nIntermediate Results:\n"
        for step, result in task.results.items():
            response += f"  • {step}: {result}\n"
    return response


# ─── Pipeline Steps ───────────────────────────────────────────────────

@tool
def extract_lineart(
    task_id: str,
    method: Literal["lineart_anime", "scribble"] = "lineart_anime",
) -> str:
    """Extract lineart from manga image for ControlNet guidance. This is usually the first step after creating a task."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    task.status = "processing"
    task.current_step = "lineart_extraction"

    lineart_path = _copy_as_stage(task, "01_lineart")
    task.mark_step_complete(
        "lineart_extraction",
        {"lineart_image": lineart_path, "method": method},
    )

    return (
        f"✓ Lineart extraction completed\n"
        f"\n"
        f"Method: {method}\n"
        f"Output: {lineart_path}\n"
        f"\n"
        f"Next step: Run detect_persons to detect characters in the image.\n"
    )


@tool
def detect_persons(task_id: str, confidence_threshold: float = 0.5) -> str:
    """Detect persons in the manga using YOLO model. Returns bounding boxes for each detected person."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    task.current_step = "person_detection"

    # 模拟两个高置信度人物，confidence_threshold 仅作过滤演示
    candidates = [
        {"id": 0, "bbox": [120, 80, 520, 900], "confidence": 0.92},
        {"id": 1, "bbox": [540, 120, 880, 950], "confidence": 0.88},
    ]
    persons = [p for p in candidates if p["confidence"] >= confidence_threshold]

    annotated_path = _copy_as_stage(task, "02_person_detection")

    task.mark_step_complete(
        "person_detection",
        {
            "persons_count": len(persons),
            "persons": persons,
            "annotated_image": annotated_path,
        },
    )

    response = f"✓ Person detection completed\n\nDetected: {len(persons)} person(s)\n"
    for p in persons:
        response += f"  • Person {p['id']}: confidence {p['confidence']:.2f}\n"
    response += f"\nAnnotated image: {annotated_path}\n"
    response += "\nNext step: Run identify_characters to identify which character each person is."
    return response


@tool
def identify_characters(task_id: str, person_ids: list[int] | None = None) -> str:
    """Use CLIP ReID to identify which character each detected person is. Requires detect_persons to be run first."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    if "person_detection" not in task.results:
        return "✗ Please run detect_persons first"

    task.current_step = "character_identification"

    persons = task.results["person_detection"]["persons"]
    if person_ids:
        persons = [p for p in persons if p["id"] in person_ids]

    name_pool = [
        ("Sagiri", 0, 0.890),
        ("Masamune Izumi", 1, 0.823),
        ("Elf Yamada", 2, 0.781),
        ("Muramasa Senju", 3, 0.755),
    ]
    identifications = []
    for i, p in enumerate(persons):
        name, lora_idx, sim = name_pool[i % len(name_pool)]
        identifications.append({
            "person_id": p["id"],
            "character": name,
            "lora_index": lora_idx,
            "similarity": sim,
        })

    task.mark_step_complete("character_identification", {"identifications": identifications})

    response = "✓ Character identification completed\n\n"
    for ident in identifications:
        response += f"  • Person {ident['person_id']}: {ident['character']} (similarity: {ident['similarity']:.3f})\n"
    response += "\nNext step: Run detect_bubbles to detect speech bubbles."
    return response


@tool
def detect_bubbles(task_id: str, use_yolo: bool = True) -> str:
    """Detect speech bubbles to protect text areas during colorization."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    task.current_step = "bubble_detection"

    bubbles = [
        {"id": 0, "bbox": [100, 60, 280, 200]},
        {"id": 1, "bbox": [440, 50, 620, 180]},
        {"id": 2, "bbox": [200, 850, 380, 980]},
    ]
    method = "yolo" if use_yolo else "opencv"

    task.mark_step_complete(
        "bubble_detection",
        {"bubbles_count": len(bubbles), "bubbles": bubbles, "method": method},
    )

    return (
        f"✓ Speech bubble detection completed\n"
        f"\n"
        f"Detected: {len(bubbles)} bubble(s)\n"
        f"Method: {'YOLO' if use_yolo else 'OpenCV'}\n"
        f"\n"
        f"Next step: Run generate_masks to create region masks.\n"
    )


@tool
def generate_masks(
    task_id: str,
    segmentation_backend: Literal["anime_seg", "sam2", "anime_seg_refine"] = "anime_seg",
) -> str:
    """Generate region masks for each person and background. Required before running inference."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    if "person_detection" not in task.results:
        return "✗ Please run detect_persons first"

    task.current_step = "mask_generation"

    persons = task.results["person_detection"]["persons"]
    person_labels: list[str] = []
    if "character_identification" in task.results:
        for ident in task.results["character_identification"]["identifications"]:
            person_labels.append(ident["character"])
    else:
        person_labels = [f"Person{i+1}" for i in range(len(persons))]

    vis_path = _copy_as_stage(task, "04_region_mask")

    task.mark_step_complete(
        "mask_generation",
        {
            "regions_count": len(persons) + 1,
            "person_labels": person_labels,
            "visualization": vis_path,
            "backend": segmentation_backend,
        },
    )

    return (
        f"✓ Region masks generated\n"
        f"\n"
        f"Total regions: {len(persons) + 1} (1 background + {len(persons)} persons)\n"
        f"Backend: {segmentation_backend}\n"
        f"Visualization: {vis_path}\n"
        f"\n"
        f"Next step: Run run_inference to colorize the image (this will take 10-20 minutes).\n"
    )


@tool
async def run_inference(
    task_id: str,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    controlnet_scale: float | None = None,
    seed: int | None = None,
) -> str:
    """Run SDXL + ControlNet inference to generate colored image. This is the main colorization step and takes 10-20 minutes depending on image size."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    required = ["lineart_extraction", "person_detection", "character_identification", "mask_generation"]
    for step in required:
        if step not in task.results:
            return f"✗ Please complete {step} first"

    task.current_step = "inference"

    p = _MOCK_CONFIG["inference_params"]
    n_steps = num_inference_steps if num_inference_steps is not None else p["num_inference_steps"]
    cfg = guidance_scale if guidance_scale is not None else p["guidance_scale"]
    cn = controlnet_scale if controlnet_scale is not None else p["controlnet_scale"]
    sd = seed if seed is not None else p["seed"]

    # mock 用 3 秒模拟 real 的 1~3 分钟，让 UI spinner 可见
    t0 = time.time()
    await asyncio.sleep(3)
    inference_time = time.time() - t0

    colored_path = _copy_as_stage(task, "05_colored_raw")

    task.mark_step_complete(
        "inference",
        {
            "colored_image": colored_path,
            "inference_time": inference_time,
            "seed_used": sd,
            "parameters": {
                "num_inference_steps": n_steps,
                "guidance_scale": cfg,
                "controlnet_scale": cn,
            },
        },
    )

    return (
        f"\n"
        f"✓ Inference completed\n"
        f"\n"
        f"Output: {colored_path}\n"
        f"Time: {inference_time:.1f}s\n"
        f"Seed: {sd}\n"
        f"\n"
        f"Next step: Run postprocess to apply final touches (blend lineart, restore text, upscale).\n"
    )


@tool
def postprocess(
    task_id: str,
    blend_lineart: bool = True,
    blend_alpha: float = 0.15,
    restore_bubbles: bool = True,
    upscale: bool = True,
) -> str:
    """Apply post-processing: blend lineart, restore speech bubbles, upscale. This is the final step."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    if "inference" not in task.results:
        return "✗ Please run run_inference first"

    task.current_step = "postprocess"

    final_path = _copy_as_stage(task, "07_final")

    main_output = (
        config.PROJECT_ROOT / "data" / "outputs" / "colored"
        / f"{Path(task.image_path).stem}_colored.png"
    )
    main_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(final_path, main_output)
    except OSError as e:
        logger.warning("写入 main_output 失败: %s", e)

    task.mark_step_complete(
        "postprocess",
        {
            "final_image": final_path,
            "main_output": str(main_output),
            "options": {
                "blend_lineart": blend_lineart,
                "blend_alpha": blend_alpha,
                "restore_bubbles": restore_bubbles,
                "upscale": upscale,
            },
        },
    )

    return (
        f"✓ Post-processing completed\n"
        f"\n"
        f"Final image: {final_path}\n"
        f"Main output: {main_output}\n"
        f"\n"
        f"Options applied:\n"
        f"  • Lineart blend: {blend_lineart} (alpha: {blend_alpha})\n"
        f"  • Restore bubbles: {restore_bubbles}\n"
        f"  • Upscale: {upscale}\n"
        f"\n"
        f"🎉 Colorization complete! \n"
        f"\n"
        f"All intermediate results saved to: {task.output_dir}\n"
    )


# ─── Configuration Management ─────────────────────────────────────────

@tool
def get_config() -> str:
    """Get current configuration including model settings, LoRA configs, and inference parameters."""
    c = _MOCK_CONFIG
    response = "Current Configuration:\n\n"
    response += f"Base Model: {c['base_model']}\n"
    response += f"ControlNet Mode: {c['controlnet_mode']}\n"
    response += "\nLoRA Characters:\n"
    for lora in c["lora_configs"]:
        response += f"  {lora['index']}: {lora['name']} (scale: {lora['scale']})\n"
    response += "\nInference Parameters:\n"
    p = c["inference_params"]
    response += f"  • Steps: {p['num_inference_steps']}\n"
    response += f"  • Guidance Scale: {p['guidance_scale']}\n"
    response += f"  • ControlNet Scale: {p['controlnet_scale']}\n"
    response += f"  • Seed: {p['seed']}\n"
    response += f"\nReID: {'Enabled' if c['reid_enabled'] else 'Disabled'} (threshold: {c['reid_threshold']})\n"
    return response


@tool
def update_inference_params(
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    controlnet_scale: float | None = None,
    seed: int | None = None,
) -> str:
    """Update inference parameters like guidance_scale, steps, etc. Use this when user wants to adjust generation quality."""
    updated: list[str] = []
    if num_inference_steps is not None:
        _MOCK_CONFIG["inference_params"]["num_inference_steps"] = num_inference_steps
        updated.append(f"num_inference_steps = {num_inference_steps}")
    if guidance_scale is not None:
        _MOCK_CONFIG["inference_params"]["guidance_scale"] = guidance_scale
        updated.append(f"guidance_scale = {guidance_scale}")
    if controlnet_scale is not None:
        _MOCK_CONFIG["inference_params"]["controlnet_scale"] = controlnet_scale
        updated.append(f"controlnet_scale = {controlnet_scale}")
    if seed is not None:
        _MOCK_CONFIG["inference_params"]["seed"] = seed
        updated.append(f"seed = {seed}")

    response = "✓ Inference parameters updated:\n\n"
    for item in updated:
        response += f"  • {item}\n"
    response += "\nThese parameters will be used for the next inference run."
    return response


@tool
def update_prompt(
    character_index: int | None = None,
    character_prompt: str | None = None,
    background_prompt: str | None = None,
    negative_prompt: str | None = None,
) -> str:
    """Update character or background prompts. Use this when user wants to change colors, style, or appearance of specific characters."""
    updated: list[str] = []
    if character_index is not None and character_prompt is not None:
        if 0 <= character_index < len(_MOCK_CONFIG["lora_configs"]):
            char_name = _MOCK_CONFIG["lora_configs"][character_index]["name"]
            updated.append(f"Character {character_index} ({char_name}) prompt updated")
    if background_prompt is not None:
        updated.append("Background prompt updated")
    if negative_prompt is not None:
        updated.append("Negative prompt updated")

    response = "✓ Prompts updated:\n\n"
    for item in updated:
        response += f"  • {item}\n"
    response += "\nThese prompts will be used for the next inference run."
    response += "\nTip: Use reset_to_step('inference') to regenerate with new prompts."
    return response


# ─── Utility Tools ────────────────────────────────────────────────────

@tool
def analyze_image(
    task_id: str,
    image_type: Literal["original", "lineart", "colored", "final"] = "colored",
) -> str:
    """Analyze image quality metrics (brightness, contrast, saturation, etc.) to help decide if adjustments are needed."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Task not found: {task_id}"

    image_path: str | None = None
    if image_type == "original":
        image_path = task.image_path
    elif image_type == "lineart" and "lineart_extraction" in task.results:
        image_path = task.results["lineart_extraction"]["lineart_image"]
    elif image_type == "colored" and "inference" in task.results:
        image_path = task.results["inference"]["colored_image"]
    elif image_type == "final" and "postprocess" in task.results:
        image_path = task.results["postprocess"]["final_image"]

    if not image_path or not os.path.exists(image_path):
        return f"✗ Image not available: {image_type}"

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            size = img.size
    except Exception:
        size = task.image_size

    brightness = 0.58  # mock 静态值，避免 numpy 依赖

    response = f"Image Analysis: {image_type}\n\n"
    response += f"Brightness: {brightness:.2f}\n"
    response += f"Size: {size[0]}x{size[1]}\n"
    response += "\nSuggestions:\n"
    if brightness < 0.4:
        response += "  • Image appears dark, consider increasing guidance_scale\n"
    elif brightness > 0.7:
        response += "  • Image appears bright, colors look good\n"
    return response


@tool
def reset_to_step(
    task_id: str,
    step: Literal[
        "lineart_extraction",
        "person_detection",
        "character_identification",
        "bubble_detection",
        "mask_generation",
        "inference",
        "postprocess",
    ],
) -> str:
    """Reset task to a specific step to re-run from there. Useful when you want to adjust parameters and regenerate without starting from scratch."""
    task = _get_task(task_id)
    if task is None:
        return f"✗ Reset failed: Task not found: {task_id}"

    cleared = task.reset_to_step(step)

    response = f"✓ Task reset to: {step}\n\nCleared steps:\n"
    for s in cleared:
        response += f"  • {s}\n"
    response += f"\nCurrent step: {task.current_step}"
    response += "\nYou can now re-run the pipeline from this step."
    return response


# ─── 入口 ─────────────────────────────────────────────────────────────

def get_mock_tools() -> list:
    """返回 15 个 mock 工具，供 `agent/mcp_bridge.py` 在 AGENT_MCP_MODE=mock 下使用。"""
    return [
        create_task,
        get_task_status,
        get_task_result,
        extract_lineart,
        detect_persons,
        identify_characters,
        detect_bubbles,
        generate_masks,
        run_inference,
        postprocess,
        get_config,
        update_inference_params,
        update_prompt,
        analyze_image,
        reset_to_step,
    ]
