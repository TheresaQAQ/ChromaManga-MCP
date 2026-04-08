#!/usr/bin/env python3
"""
ChromaManga MCP Server
Provides manga colorization tools via Model Context Protocol
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# MCP SDK imports
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import mcp.server.stdio

# ChromaManga imports
from core import config
from core.colorize import build_pipeline
from core.task_manager import TaskManager

# Initialize MCP Server
app = Server("chromamanga")

# Global state
task_manager: Optional[TaskManager] = None
pipeline = None
reid = None


# ─────────────────────────────────────────────────────────────────────────────
# Startup: Load Models
# ─────────────────────────────────────────────────────────────────────────────

async def initialize_models():
    """Load models on startup"""
    global pipeline, reid, task_manager
    
    print("Initializing ChromaManga MCP Server...", file=sys.stderr)
    print("Loading models (this may take a minute)...", file=sys.stderr)
    
    try:
        pipeline, reid = build_pipeline()
        
        # IMPORTANT: Release ReID CLIP model to save memory
        # It will be lazy-loaded when needed
        if reid:
            print("✓ ReID initialized (CLIP model will be lazy-loaded)", file=sys.stderr)
        
        task_manager = TaskManager()
        print("✓ Models loaded successfully", file=sys.stderr)
        print("✓ MCP Server ready", file=sys.stderr)
    except Exception as e:
        print(f"✗ Failed to load models: {e}", file=sys.stderr)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Tools Definition
# ─────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools"""
    return [
        # Task Management
        Tool(
            name="create_task",
            description="Create a new manga colorization task by providing an image path",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the manga image file"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Optional name for this task"
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="get_task_status",
            description="Get the current status and progress of a colorization task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to query"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="get_task_result",
            description="Get the final colorized image and all intermediate results",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID"
                    },
                    "include_intermediates": {
                        "type": "boolean",
                        "description": "Include intermediate step results",
                        "default": True
                    }
                },
                "required": ["task_id"]
            }
        ),
        
        # Pipeline Steps
        Tool(
            name="extract_lineart",
            description="Extract lineart from manga image for ControlNet guidance. This is usually the first step after creating a task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["lineart_anime", "scribble"],
                        "description": "Extraction method. lineart_anime is recommended for manga."
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="detect_persons",
            description="Detect persons in the manga using YOLO model. Returns bounding boxes for each detected person.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Detection confidence threshold (0-1)",
                        "default": 0.5
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="identify_characters",
            description="Use CLIP ReID to identify which character each detected person is. Requires detect_persons to be run first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "person_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific person IDs to identify. If not provided, identifies all detected persons."
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="detect_bubbles",
            description="Detect speech bubbles to protect text areas during colorization",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "use_yolo": {
                        "type": "boolean",
                        "description": "Use YOLO model (more accurate) or OpenCV (faster)",
                        "default": True
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="generate_masks",
            description="Generate region masks for each person and background. Required before running inference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "segmentation_backend": {
                        "type": "string",
                        "enum": ["anime_seg", "sam2", "anime_seg_refine"],
                        "description": "Segmentation model. anime_seg is recommended for manga.",
                        "default": "anime_seg"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="run_inference",
            description="Run SDXL + ControlNet inference to generate colored image. This is the main colorization step and takes 10-20 minutes depending on image size.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of diffusion steps (15-28). More steps = better quality but slower.",
                        "default": 20
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "CFG scale (5-8). Higher = more prompt adherence, may oversaturate colors.",
                        "default": 6.0
                    },
                    "controlnet_scale": {
                        "type": "number",
                        "description": "ControlNet strength (0.8-1.2). Higher = stricter lineart following.",
                        "default": 1.1
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed. Use -1 for random.",
                        "default": 42
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="postprocess",
            description="Apply post-processing: blend lineart, restore speech bubbles, upscale. This is the final step.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "blend_lineart": {
                        "type": "boolean",
                        "description": "Overlay original lineart on colored image",
                        "default": True
                    },
                    "blend_alpha": {
                        "type": "number",
                        "description": "Lineart overlay strength (0-1)",
                        "default": 0.15
                    },
                    "restore_bubbles": {
                        "type": "boolean",
                        "description": "Restore original text in speech bubbles",
                        "default": True
                    },
                    "upscale": {
                        "type": "boolean",
                        "description": "Upscale to original resolution with Real-ESRGAN",
                        "default": True
                    }
                },
                "required": ["task_id"]
            }
        ),
        
        # Configuration Management
        Tool(
            name="get_config",
            description="Get current configuration including model settings, LoRA configs, and inference parameters",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="update_inference_params",
            description="Update inference parameters like guidance_scale, steps, etc. Use this when user wants to adjust generation quality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of inference steps"
                    },
                    "guidance_scale": {
                        "type": "number",
                        "description": "Guidance scale (CFG)"
                    },
                    "controlnet_scale": {
                        "type": "number",
                        "description": "ControlNet strength"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed"
                    }
                }
            }
        ),
        Tool(
            name="update_prompt",
            description="Update character or background prompts. Use this when user wants to change colors, style, or appearance of specific characters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "character_index": {
                        "type": "integer",
                        "description": "LoRA index (0=Sagiri, 1=Masamune, 2=Elf, 3=Muramasa)"
                    },
                    "character_prompt": {
                        "type": "string",
                        "description": "New prompt for this character"
                    },
                    "background_prompt": {
                        "type": "string",
                        "description": "New background prompt"
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "New negative prompt"
                    }
                }
            }
        ),
        
        # Utility Tools
        Tool(
            name="analyze_image",
            description="Analyze image quality metrics (brightness, contrast, saturation, etc.) to help decide if adjustments are needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "image_type": {
                        "type": "string",
                        "enum": ["original", "lineart", "colored", "final"],
                        "description": "Which image to analyze",
                        "default": "colored"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="reset_to_step",
            description="Reset task to a specific step to re-run from there. Useful when you want to adjust parameters and regenerate without starting from scratch.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "step": {
                        "type": "string",
                        "enum": [
                            "lineart_extraction",
                            "person_detection",
                            "character_identification",
                            "bubble_detection",
                            "mask_generation",
                            "inference",
                            "postprocess"
                        ],
                        "description": "Step to reset to"
                    }
                },
                "required": ["task_id", "step"]
            }
        ),
    ]



# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    global task_manager, pipeline, reid
    
    try:
        # Task Management Tools
        if name == "create_task":
            return await handle_create_task(arguments)
        
        elif name == "get_task_status":
            return await handle_get_task_status(arguments)
        
        elif name == "get_task_result":
            return await handle_get_task_result(arguments)
        
        # Pipeline Step Tools
        elif name == "extract_lineart":
            return await handle_extract_lineart(arguments)
        
        elif name == "detect_persons":
            return await handle_detect_persons(arguments)
        
        elif name == "identify_characters":
            return await handle_identify_characters(arguments)
        
        elif name == "detect_bubbles":
            return await handle_detect_bubbles(arguments)
        
        elif name == "generate_masks":
            return await handle_generate_masks(arguments)
        
        elif name == "run_inference":
            return await handle_run_inference(arguments)
        
        elif name == "postprocess":
            return await handle_postprocess(arguments)
        
        # Configuration Tools
        elif name == "get_config":
            return await handle_get_config(arguments)
        
        elif name == "update_inference_params":
            return await handle_update_inference_params(arguments)
        
        elif name == "update_prompt":
            return await handle_update_prompt(arguments)
        
        # Utility Tools
        elif name == "analyze_image":
            return await handle_analyze_image(arguments)
        
        elif name == "reset_to_step":
            return await handle_reset_to_step(arguments)
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


# ─────────────────────────────────────────────────────────────────────────────
# Task Management Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_create_task(args: dict) -> list[TextContent]:
    """Create a new colorization task"""
    image_path = args["image_path"]
    task_name = args.get("task_name")
    
    try:
        task = task_manager.create_task(image_path, task_name)
        
        response = f"""✓ Task created successfully

Task ID: {task.task_id}
Task Name: {task.task_name}
Image: {os.path.basename(image_path)}
Size: {task.image_size[0]}x{task.image_size[1]}
Output Directory: {task.output_dir}

Next step: Run extract_lineart to begin the colorization pipeline.
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"✗ Failed to create task: {str(e)}"
        )]


async def handle_get_task_status(args: dict) -> list[TextContent]:
    """Get task status"""
    task_id = args["task_id"]
    
    try:
        task = task_manager.get_task(task_id)
        progress = task.get_progress()
        
        response = f"""Task Status: {task.task_id}

Status: {task.status}
Progress: {progress:.1%} ({len(task.steps_completed)}/{len(task.STEPS_ORDER)} steps)
Current Step: {task.current_step or 'Completed'}

Steps Completed:
"""
        for step in task.steps_completed:
            response += f"  ✓ {step}\n"
        
        if task.error:
            response += f"\nError: {task.error}"
        
        return [TextContent(type="text", text=response)]
    
    except KeyError as e:
        return [TextContent(type="text", text=f"✗ Task not found: {task_id}")]


async def handle_get_task_result(args: dict) -> list[TextContent]:
    """Get task results"""
    task_id = args["task_id"]
    include_intermediates = args.get("include_intermediates", True)
    
    try:
        task = task_manager.get_task(task_id)
        
        response = f"""Task Results: {task.task_id}

Status: {task.status}
"""
        
        if "postprocess" in task.results:
            final_path = task.results["postprocess"].get("final_image")
            response += f"\n✓ Final Image: {final_path}\n"
        
        if include_intermediates and task.results:
            response += "\nIntermediate Results:\n"
            for step, result in task.results.items():
                response += f"  • {step}: {result}\n"
        
        return [TextContent(type="text", text=response)]
    
    except KeyError:
        return [TextContent(type="text", text=f"✗ Task not found: {task_id}")]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Step Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_extract_lineart(args: dict) -> list[TextContent]:
    """Extract lineart from image"""
    task_id = args["task_id"]
    method = args.get("method", "lineart_anime")
    
    try:
        task = task_manager.get_task(task_id)
        task.status = "processing"
        task.current_step = "lineart_extraction"
        
        # Import here to avoid circular dependency
        from utils.preprocess import preprocess_for_controlnet
        
        # Extract lineart
        lineart_image, _ = preprocess_for_controlnet(
            task.original_image,
            mode="union" if method == "lineart_anime" else "scribble"
        )
        
        # Save lineart
        lineart_path = os.path.join(task.output_dir, "01_lineart.png")
        lineart_image.save(lineart_path)
        
        # Store result
        result = {
            "lineart_image": lineart_path,
            "method": method
        }
        task.mark_step_complete("lineart_extraction", result)
        
        response = f"""✓ Lineart extraction completed

Method: {method}
Output: {lineart_path}

Next step: Run detect_persons to detect characters in the image.
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        task = task_manager.get_task(task_id)
        task.status = "failed"
        task.error = str(e)
        return [TextContent(type="text", text=f"✗ Lineart extraction failed: {str(e)}")]


async def handle_detect_persons(args: dict) -> list[TextContent]:
    """Detect persons in image"""
    task_id = args["task_id"]
    confidence = args.get("confidence_threshold", 0.5)
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "person_detection"
        
        from ultralytics import YOLO
        
        # Load person detection model
        model = YOLO(config.person_detection_model)
        
        # Run detection
        results = model(task.original_image, conf=confidence, verbose=False)
        
        # Extract bounding boxes
        persons = []
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                # Convert to integers to match colorize.py behavior
                bbox = [int(x) for x in box.xyxy[0].cpu().numpy().tolist()]
                conf = float(box.conf[0])
                persons.append({
                    "id": i,
                    "bbox": bbox,
                    "confidence": conf
                })
        
        # Save annotated image
        annotated = results[0].plot()
        annotated_path = os.path.join(task.output_dir, "02_person_detection.png")
        Image.fromarray(annotated).save(annotated_path)
        
        # Store result
        result = {
            "persons_count": len(persons),
            "persons": persons,
            "annotated_image": annotated_path
        }
        task.mark_step_complete("person_detection", result)
        
        response = f"""✓ Person detection completed

Detected: {len(persons)} person(s)
"""
        for p in persons:
            response += f"  • Person {p['id']}: confidence {p['confidence']:.2f}\n"
        
        response += f"\nAnnotated image: {annotated_path}\n"
        response += "\nNext step: Run identify_characters to identify which character each person is."
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        return [TextContent(type="text", text=f"✗ Person detection failed: {str(e)}")]


async def handle_identify_characters(args: dict) -> list[TextContent]:
    """Identify characters using CLIP ReID"""
    task_id = args["task_id"]
    person_ids = args.get("person_ids")
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "character_identification"
        
        # Get person detection results
        if "person_detection" not in task.results:
            return [TextContent(
                type="text",
                text="✗ Please run detect_persons first"
            )]
        
        persons = task.results["person_detection"]["persons"]
        
        # Filter persons if specific IDs provided
        if person_ids:
            persons = [p for p in persons if p["id"] in person_ids]
        
        # Identify each person
        identifications = []
        for person in persons:
            # Crop person region
            bbox = person["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = task.original_image.crop((x1, y1, x2, y2))
            
            # Identify using ReID (CLIP will be lazy-loaded)
            lora_idx, character_name, similarity = reid.identify(person_crop)
            
            identifications.append({
                "person_id": person["id"],
                "character": character_name,
                "lora_index": lora_idx,
                "similarity": float(similarity)
            })
        
        # Release CLIP model to free memory
        if reid:
            reid.release()
            print("  [ReID] CLIP model released", file=sys.stderr)
        
        # Store result
        result = {"identifications": identifications}
        task.mark_step_complete("character_identification", result)
        
        response = f"""✓ Character identification completed

"""
        for ident in identifications:
            response += f"  • Person {ident['person_id']}: {ident['character']} (similarity: {ident['similarity']:.3f})\n"
        
        response += "\nNext step: Run detect_bubbles to detect speech bubbles."
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [TextContent(type="text", text=f"✗ Character identification failed: {str(e)}")]



async def handle_detect_bubbles(args: dict) -> list[TextContent]:
    """Detect speech bubbles"""
    task_id = args["task_id"]
    use_yolo = args.get("use_yolo", True)
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "bubble_detection"
        
        from core.colorize import detect_bubble_boxes
        
        bubbles = []
        if use_yolo and config.bubble_yolo_model and os.path.exists(config.bubble_yolo_model):
            # Use YOLO model
            boxes = detect_bubble_boxes(task.original_image, config.bubble_yolo_model)
            bubbles = [{"id": i, "bbox": list(box)} for i, box in enumerate(boxes)]
        
        # Store result
        result = {
            "bubbles_count": len(bubbles),
            "bubbles": bubbles,
            "method": "yolo" if use_yolo else "opencv"
        }
        task.mark_step_complete("bubble_detection", result)
        
        response = f"""✓ Speech bubble detection completed

Detected: {len(bubbles)} bubble(s)
Method: {'YOLO' if use_yolo else 'OpenCV'}

Next step: Run generate_masks to create region masks.
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        return [TextContent(type="text", text=f"✗ Bubble detection failed: {str(e)}")]


async def handle_generate_masks(args: dict) -> list[TextContent]:
    """Generate region masks"""
    task_id = args["task_id"]
    backend = args.get("segmentation_backend", "anime_seg")
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "mask_generation"
        
        # Check prerequisites
        if "person_detection" not in task.results:
            return [TextContent(
                type="text",
                text="✗ Please run detect_persons first"
            )]
        
        persons = task.results["person_detection"]["persons"]
        
        # Import mask building function
        from core.colorize import build_region_masks, visualize_region_masks
        
        # Get lineart size (or use original image size)
        if "lineart_extraction" in task.results:
            lineart_path = task.results["lineart_extraction"]["lineart_image"]
            lineart_img = Image.open(lineart_path)
            image_size = lineart_img.size
        else:
            image_size = task.original_image.size
        
        # Extract bboxes
        person_bboxes = [p["bbox"] for p in persons]
        
        # Build masks in latent space
        bg_mask, person_masks = build_region_masks(image_size, person_bboxes)
        
        # Visualize masks
        person_labels = []
        if "character_identification" in task.results:
            identifications = task.results["character_identification"]["identifications"]
            for ident in identifications:
                person_labels.append(ident["character"])
        else:
            person_labels = [f"Person{i+1}" for i in range(len(persons))]
        
        # Create visualization
        region_vis = visualize_region_masks(
            task.original_image, bg_mask, person_masks, person_labels
        )
        vis_path = os.path.join(task.output_dir, "04_region_mask.png")
        region_vis.save(vis_path)
        
        # Store result
        result = {
            "regions_count": len(person_masks) + 1,  # +1 for background
            "bg_mask": bg_mask,
            "person_masks": person_masks,
            "person_labels": person_labels,
            "visualization": vis_path,
            "backend": backend
        }
        task.mark_step_complete("mask_generation", result)
        
        response = f"""✓ Region masks generated

Total regions: {len(person_masks) + 1} (1 background + {len(persons)} persons)
Backend: {backend}
Visualization: {vis_path}

Next step: Run run_inference to colorize the image (this will take 10-20 minutes).
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [TextContent(type="text", text=f"✗ Mask generation failed: {str(e)}")]


async def handle_run_inference(args: dict) -> list[TextContent]:
    """Run SDXL inference"""
    task_id = args["task_id"]
    num_steps = args.get("num_inference_steps", config.num_inference_steps)
    guidance = args.get("guidance_scale", config.guidance_scale)
    cn_scale = args.get("controlnet_scale", config.controlnet_scale)
    seed = args.get("seed", config.seed)
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "inference"
        
        # Check prerequisites
        required_steps = ["lineart_extraction", "person_detection", "character_identification", "mask_generation"]
        for step in required_steps:
            if step not in task.results:
                return [TextContent(
                    type="text",
                    text=f"✗ Please complete {step} first"
                )]
        
        response_start = f"""⏳ Starting inference...

Parameters:
  • Steps: {num_steps}
  • Guidance Scale: {guidance}
  • ControlNet Scale: {cn_scale}
  • Seed: {seed}

This will take approximately 10-20 minutes depending on image size.
Processing...
"""
        
        print(response_start, file=sys.stderr)
        
        # Import necessary functions
        from core.colorize import encode_prompts, build_region_masks
        from utils.regional_attention import set_regional_attn, reset_attn
        import random
        import time
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load lineart
        lineart_path = task.results["lineart_extraction"]["lineart_image"]
        lineart = Image.open(lineart_path)
        
        # Get person detection results
        persons = task.results["person_detection"]["persons"]
        person_bboxes = [tuple(p["bbox"]) for p in persons]
        
        # Get character identifications
        identifications = task.results["character_identification"]["identifications"]
        
        # Build region prompts
        region_prompts = []
        region_labels = []
        
        for ident in identifications:
            lora_idx = ident["lora_index"]
            character = ident["character"]
            prompt = config.lora_configs[lora_idx].get("prompt", config.positive_prompt)
            region_prompts.append(prompt)
            region_labels.append(character)
            print(f"  Person {ident['person_id']} [{character}]: {prompt[:60]}...", file=sys.stderr)
        
        # Build region masks
        bg_mask, person_masks = build_region_masks(lineart.size, person_bboxes)
        
        # Encode prompts
        bg_prompt = config.background_prompt
        all_prompts = [bg_prompt] + region_prompts
        print(f"Encoding prompts ({len(all_prompts)} regions)...", file=sys.stderr)
        
        all_embeds = encode_prompts(pipeline, all_prompts, device)
        
        base_cond, base_neg, base_pooled, base_neg_pooled = all_embeds[0]
        person_cond_embeds = [e[0] for e in all_embeds[1:]]
        
        # Inject Regional Attention
        if person_masks:
            latent_h = lineart.height // 8
            latent_w = lineart.width // 8
            print(f"Injecting Regional Attention... (latent: {latent_h}×{latent_w})", file=sys.stderr)
            set_regional_attn(pipeline, person_masks, person_cond_embeds, base_cond, latent_h, latent_w)
        
        # Run inference
        print("Running inference...", file=sys.stderr)
        t0 = time.time()
        
        _seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=device).manual_seed(_seed)
        
        result = pipeline(
            prompt_embeds=base_cond,
            negative_prompt_embeds=base_neg,
            pooled_prompt_embeds=base_pooled,
            negative_pooled_prompt_embeds=base_neg_pooled,
            image=lineart,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=cn_scale,
            generator=gen,
            width=lineart.width,
            height=lineart.height,
        ).images[0]
        
        inference_time = time.time() - t0
        print(f"Inference completed in {inference_time:.1f}s", file=sys.stderr)
        
        # Reset attention
        if person_masks:
            reset_attn(pipeline)
        
        # Save colored image
        colored_path = os.path.join(task.output_dir, "05_colored_raw.png")
        result.save(colored_path)
        
        # Store result
        result_data = {
            "colored_image": colored_path,
            "inference_time": inference_time,
            "seed_used": _seed,
            "parameters": {
                "num_inference_steps": num_steps,
                "guidance_scale": guidance,
                "controlnet_scale": cn_scale
            }
        }
        task.mark_step_complete("inference", result_data)
        
        response = f"""
✓ Inference completed

Output: {colored_path}
Time: {inference_time:.1f}s
Seed: {_seed}

Next step: Run postprocess to apply final touches (blend lineart, restore text, upscale).
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        task = task_manager.get_task(task_id)
        task.status = "failed"
        task.error = str(e)
        return [TextContent(type="text", text=f"✗ Inference failed: {str(e)}")]


async def handle_postprocess(args: dict) -> list[TextContent]:
    """Apply post-processing"""
    task_id = args["task_id"]
    blend_lineart = args.get("blend_lineart", True)
    blend_alpha = args.get("blend_alpha", config.blend_line_alpha)
    restore_bubbles = args.get("restore_bubbles", True)
    upscale = args.get("upscale", config.upscale_enabled)
    
    try:
        task = task_manager.get_task(task_id)
        task.current_step = "postprocess"
        
        # Check prerequisites
        if "inference" not in task.results:
            return [TextContent(
                type="text",
                text="✗ Please run run_inference first"
            )]
        
        from utils.postprocess import blend_lineart as blend_lineart_func
        from core.colorize import stamp_text, extract_text_by_threshold, upscale_to_target
        import cv2
        
        # Load colored image
        colored_path = task.results["inference"]["colored_image"]
        result = Image.open(colored_path)
        
        # Load lineart
        lineart_path = task.results["lineart_extraction"]["lineart_image"]
        lineart = Image.open(lineart_path)
        
        # Ensure same size
        if result.size != lineart.size:
            lineart = lineart.resize(result.size, Image.LANCZOS)
        
        # Blend lineart
        if blend_lineart:
            print(f"Blending lineart (alpha={blend_alpha})...", file=sys.stderr)
            result = blend_lineart_func(result, lineart, alpha=blend_alpha)
            
            blend_path = os.path.join(task.output_dir, "06_lineart_blend.png")
            result.save(blend_path)
        
        # Resize to original size
        original_size = task.original_image.size
        if result.size != original_size:
            print(f"Resizing to original: {result.size} → {original_size}", file=sys.stderr)
            
            if upscale and result.size[0] < original_size[0]:
                # Use Real-ESRGAN for upscaling
                print("Upscaling with Real-ESRGAN...", file=sys.stderr)
                result = upscale_to_target(result, original_size)
            else:
                result = result.resize(original_size, Image.LANCZOS)
        
        # Restore speech bubble text
        if restore_bubbles and "bubble_detection" in task.results:
            bubbles = task.results["bubble_detection"]["bubbles"]
            if bubbles:
                print(f"Restoring {len(bubbles)} speech bubbles...", file=sys.stderr)
                
                # Extract text mask
                img_gray = cv2.cvtColor(np.array(task.original_image), cv2.COLOR_RGB2GRAY)
                bubble_boxes = [tuple(b["bbox"]) for b in bubbles]
                
                shrink_x = getattr(config, "bubble_text_shrink_x", 0.75)
                shrink_y = getattr(config, "bubble_text_shrink_y", 0.90)
                
                text_mask_float = extract_text_by_threshold(
                    img_gray, bubble_boxes, threshold=210,
                    shrink_x=shrink_x, shrink_y=shrink_y
                )
                
                # Stamp text
                result_np = np.array(result)
                result_np = stamp_text(result_np, text_mask_float, bubble_boxes,
                                     shrink_x=shrink_x, shrink_y=shrink_y)
                result = Image.fromarray(result_np)
        
        # Save final result
        final_path = os.path.join(task.output_dir, "07_final.png")
        result.save(final_path)
        
        # Also save to main output directory
        main_output = os.path.join(
            config.outputs_colored_dir,
            f"{Path(task.image_path).stem}_colored.png"
        )
        result.save(main_output)
        
        # Store result
        result_data = {
            "final_image": final_path,
            "main_output": main_output,
            "options": {
                "blend_lineart": blend_lineart,
                "blend_alpha": blend_alpha,
                "restore_bubbles": restore_bubbles,
                "upscale": upscale
            }
        }
        task.mark_step_complete("postprocess", result_data)
        
        response = f"""✓ Post-processing completed

Final image: {final_path}
Main output: {main_output}

Options applied:
  • Lineart blend: {blend_lineart} (alpha: {blend_alpha})
  • Restore bubbles: {restore_bubbles}
  • Upscale: {upscale}

🎉 Colorization complete! 

All intermediate results saved to: {task.output_dir}
"""
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [TextContent(type="text", text=f"✗ Post-processing failed: {str(e)}")]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_get_config(args: dict) -> list[TextContent]:
    """Get current configuration"""
    config_data = {
        "base_model": os.path.basename(config.base_model_id),
        "controlnet_mode": config.controlnet_mode,
        "lora_configs": [
            {
                "index": i,
                "name": lora["trigger"],
                "scale": lora["scale"]
            }
            for i, lora in enumerate(config.lora_configs)
        ],
        "inference_params": {
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "controlnet_scale": config.controlnet_scale,
            "seed": config.seed
        },
        "reid_enabled": config.reid_enabled,
        "reid_threshold": config.reid_threshold
    }
    
    response = f"""Current Configuration:

Base Model: {config_data['base_model']}
ControlNet Mode: {config_data['controlnet_mode']}

LoRA Characters:
"""
    for lora in config_data['lora_configs']:
        response += f"  {lora['index']}: {lora['name']} (scale: {lora['scale']})\n"
    
    response += f"""
Inference Parameters:
  • Steps: {config_data['inference_params']['num_inference_steps']}
  • Guidance Scale: {config_data['inference_params']['guidance_scale']}
  • ControlNet Scale: {config_data['inference_params']['controlnet_scale']}
  • Seed: {config_data['inference_params']['seed']}

ReID: {'Enabled' if config_data['reid_enabled'] else 'Disabled'} (threshold: {config_data['reid_threshold']})
"""
    
    return [TextContent(type="text", text=response)]


async def handle_update_inference_params(args: dict) -> list[TextContent]:
    """Update inference parameters"""
    updated = []
    
    if "num_inference_steps" in args:
        config.num_inference_steps = args["num_inference_steps"]
        updated.append(f"num_inference_steps = {args['num_inference_steps']}")
    
    if "guidance_scale" in args:
        config.guidance_scale = args["guidance_scale"]
        updated.append(f"guidance_scale = {args['guidance_scale']}")
    
    if "controlnet_scale" in args:
        config.controlnet_scale = args["controlnet_scale"]
        updated.append(f"controlnet_scale = {args['controlnet_scale']}")
    
    if "seed" in args:
        config.seed = args["seed"]
        updated.append(f"seed = {args['seed']}")
    
    response = "✓ Inference parameters updated:\n\n"
    for item in updated:
        response += f"  • {item}\n"
    
    response += "\nThese parameters will be used for the next inference run."
    
    return [TextContent(type="text", text=response)]


async def handle_update_prompt(args: dict) -> list[TextContent]:
    """Update prompts"""
    updated = []
    
    if "character_index" in args and "character_prompt" in args:
        idx = args["character_index"]
        prompt = args["character_prompt"]
        if 0 <= idx < len(config.lora_configs):
            config.lora_configs[idx]["prompt"] = prompt
            char_name = config.lora_configs[idx]["trigger"]
            updated.append(f"Character {idx} ({char_name}) prompt updated")
    
    if "background_prompt" in args:
        config.background_prompt = args["background_prompt"]
        updated.append("Background prompt updated")
    
    if "negative_prompt" in args:
        config.negative_prompt = args["negative_prompt"]
        updated.append("Negative prompt updated")
    
    response = "✓ Prompts updated:\n\n"
    for item in updated:
        response += f"  • {item}\n"
    
    response += "\nThese prompts will be used for the next inference run."
    response += "\nTip: Use reset_to_step('inference') to regenerate with new prompts."
    
    return [TextContent(type="text", text=response)]


# ─────────────────────────────────────────────────────────────────────────────
# Utility Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_analyze_image(args: dict) -> list[TextContent]:
    """Analyze image quality"""
    task_id = args["task_id"]
    image_type = args.get("image_type", "colored")
    
    try:
        task = task_manager.get_task(task_id)
        
        # Get image path based on type
        image_path = None
        if image_type == "original":
            image_path = task.image_path
        elif image_type == "lineart" and "lineart_extraction" in task.results:
            image_path = task.results["lineart_extraction"]["lineart_image"]
        elif image_type == "colored" and "inference" in task.results:
            image_path = task.results["inference"]["colored_image"]
        elif image_type == "final" and "postprocess" in task.results:
            image_path = task.results["postprocess"]["final_image"]
        
        if not image_path or not os.path.exists(image_path):
            return [TextContent(
                type="text",
                text=f"✗ Image not available: {image_type}"
            )]
        
        # Simple analysis (would be more sophisticated in production)
        img = Image.open(image_path)
        img_array = np.array(img)
        
        brightness = np.mean(img_array) / 255.0
        
        response = f"""Image Analysis: {image_type}

Brightness: {brightness:.2f}
Size: {img.size[0]}x{img.size[1]}

Suggestions:
"""
        if brightness < 0.4:
            response += "  • Image appears dark, consider increasing guidance_scale\n"
        elif brightness > 0.7:
            response += "  • Image appears bright, colors look good\n"
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        return [TextContent(type="text", text=f"✗ Analysis failed: {str(e)}")]


async def handle_reset_to_step(args: dict) -> list[TextContent]:
    """Reset task to a specific step"""
    task_id = args["task_id"]
    step = args["step"]
    
    try:
        task = task_manager.get_task(task_id)
        cleared_steps = task.reset_to_step(step)
        
        response = f"""✓ Task reset to: {step}

Cleared steps:
"""
        for s in cleared_steps:
            response += f"  • {s}\n"
        
        response += f"\nCurrent step: {task.current_step}"
        response += "\nYou can now re-run the pipeline from this step."
        
        return [TextContent(type="text", text=response)]
    
    except Exception as e:
        return [TextContent(type="text", text=f"✗ Reset failed: {str(e)}")]


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Main entry point"""
    # Initialize models
    await initialize_models()
    
    # Run MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
