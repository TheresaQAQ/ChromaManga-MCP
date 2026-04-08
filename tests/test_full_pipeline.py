#!/usr/bin/env python3
"""
Complete pipeline test for ChromaManga MCP Server
Tests all steps from image upload to final colorization
"""

import sys
import asyncio
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import MCP server handlers
from mcp_server.chromamanga_mcp_server import (
    initialize_models,
    handle_create_task,
    handle_extract_lineart,
    handle_detect_persons,
    handle_identify_characters,
    handle_detect_bubbles,
    handle_generate_masks,
    handle_run_inference,
    handle_postprocess,
    handle_get_task_result,
    task_manager
)


async def test_full_pipeline(image_path: str):
    """Test complete colorization pipeline"""
    
    print("=" * 80)
    print("ChromaManga MCP Server - Full Pipeline Test")
    print("=" * 80)
    
    # Step 0: Initialize models
    print("\n[0/8] Initializing models...")
    try:
        await initialize_models()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return None
    
    # Step 1: Create task
    print("\n[1/8] Creating task...")
    try:
        result = await handle_create_task({
            "image_path": image_path,
            "task_name": "full_pipeline_test"
        })
        print(result[0].text)
        
        # Extract task_id
        task_id = None
        for line in result[0].text.split('\n'):
            if line.startswith("Task ID:"):
                task_id = line.split(": ")[1].strip()
                break
        
        if not task_id:
            print("✗ Failed to get task_id")
            return None
        
        print(f"✓ Task created: {task_id}")
        
    except Exception as e:
        print(f"✗ Create task failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 2: Extract lineart
    print("\n[2/8] Extracting lineart...")
    try:
        result = await handle_extract_lineart({
            "task_id": task_id,
            "method": "lineart_anime"
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Extract lineart failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 3: Detect persons
    print("\n[3/8] Detecting persons...")
    try:
        result = await handle_detect_persons({
            "task_id": task_id,
            "confidence_threshold": 0.5
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Detect persons failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 4: Identify characters
    print("\n[4/8] Identifying characters...")
    try:
        result = await handle_identify_characters({
            "task_id": task_id
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Identify characters failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 5: Detect bubbles
    print("\n[5/8] Detecting speech bubbles...")
    try:
        result = await handle_detect_bubbles({
            "task_id": task_id,
            "use_yolo": True
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Detect bubbles failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 6: Generate masks
    print("\n[6/8] Generating region masks...")
    try:
        result = await handle_generate_masks({
            "task_id": task_id,
            "segmentation_backend": "anime_seg"
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Generate masks failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 7: Run inference
    print("\n[7/8] Running inference (this will take 10-20 minutes)...")
    try:
        result = await handle_run_inference({
            "task_id": task_id,
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "controlnet_scale": 1.1,
            "seed": 42
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Step 8: Postprocess
    print("\n[8/8] Applying post-processing...")
    try:
        result = await handle_postprocess({
            "task_id": task_id,
            "blend_lineart": True,
            "blend_alpha": 0.15,
            "restore_bubbles": True,
            "upscale": True
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Postprocess failed: {e}")
        import traceback
        traceback.print_exc()
        return task_id
    
    # Get final result
    print("\n" + "=" * 80)
    print("Getting final results...")
    try:
        result = await handle_get_task_result({
            "task_id": task_id,
            "include_intermediates": True
        })
        print(result[0].text)
    except Exception as e:
        print(f"✗ Get result failed: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Full pipeline test completed!")
    print("=" * 80)
    
    return task_id


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_full_pipeline.py <image_path>")
        print("\nExample:")
        print("  python test_full_pipeline.py inputs/553d7fe3f77db5f73264cad52de45117.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Run test
    task_id = asyncio.run(test_full_pipeline(image_path))
    
    if task_id:
        print(f"\nTask ID: {task_id}")
        print(f"Check output directory for results")
    else:
        print("\nTest failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
