#!/usr/bin/env python3
"""
Verify ChromaManga MCP Server integration
Check if all imports and functions are available
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    errors = []
    
    # Check MCP SDK
    try:
        import mcp
        print("  ✓ mcp")
    except ImportError as e:
        errors.append(f"mcp: {e}")
        print(f"  ✗ mcp: {e}")
    
    # Check ChromaManga modules
    try:
        from core import config
        print("  ✓ config")
    except ImportError as e:
        errors.append(f"config: {e}")
        print(f"  ✗ config: {e}")
    
    try:
        from core.colorize import build_pipeline, encode_prompts, build_region_masks
        print("  ✓ colorize (build_pipeline, encode_prompts, build_region_masks)")
    except ImportError as e:
        errors.append(f"colorize: {e}")
        print(f"  ✗ colorize: {e}")
    
    try:
        from utils.preprocess import preprocess_for_controlnet
        print("  ✓ utils.preprocess")
    except ImportError as e:
        errors.append(f"utils.preprocess: {e}")
        print(f"  ✗ utils.preprocess: {e}")
    
    try:
        from utils.postprocess import blend_lineart
        print("  ✓ utils.postprocess")
    except ImportError as e:
        errors.append(f"utils.postprocess: {e}")
        print(f"  ✗ utils.postprocess: {e}")
    
    try:
        from utils.regional_attention import set_regional_attn, reset_attn
        print("  ✓ utils.regional_attention")
    except ImportError as e:
        errors.append(f"utils.regional_attention: {e}")
        print(f"  ✗ utils.regional_attention: {e}")
    
    try:
        from utils.character_reid import CharacterReID
        print("  ✓ utils.character_reid")
    except ImportError as e:
        errors.append(f"utils.character_reid: {e}")
        print(f"  ✗ utils.character_reid: {e}")
    
    try:
        from task_manager import TaskManager
        print("  ✓ task_manager")
    except ImportError as e:
        errors.append(f"task_manager: {e}")
        print(f"  ✗ task_manager: {e}")
    
    # Check dependencies
    try:
        import torch
        print(f"  ✓ torch ({torch.__version__})")
    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"  ✗ torch: {e}")
    
    try:
        from PIL import Image
        print("  ✓ PIL")
    except ImportError as e:
        errors.append(f"PIL: {e}")
        print(f"  ✗ PIL: {e}")
    
    try:
        import numpy as np
        print(f"  ✓ numpy ({np.__version__})")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"  ✗ numpy: {e}")
    
    try:
        import cv2
        print(f"  ✓ opencv ({cv2.__version__})")
    except ImportError as e:
        errors.append(f"opencv: {e}")
        print(f"  ✗ opencv: {e}")
    
    try:
        from ultralytics import YOLO
        print("  ✓ ultralytics")
    except ImportError as e:
        errors.append(f"ultralytics: {e}")
        print(f"  ✗ ultralytics: {e}")
    
    return errors


def check_mcp_server():
    """Check if MCP server can be imported"""
    print("\nChecking MCP server...")
    
    try:
        from chromamanga_mcp_server import (
            app,
            initialize_models,
            handle_create_task,
            handle_extract_lineart,
            handle_detect_persons,
            handle_identify_characters,
            handle_detect_bubbles,
            handle_generate_masks,
            handle_run_inference,
            handle_postprocess,
        )
        print("  ✓ All MCP server handlers imported")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import MCP server: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config():
    """Check if config is valid"""
    print("\nChecking config...")
    
    try:
        import config
        
        # Check critical attributes
        attrs = [
            'base_model_id',
            'controlnet_mode',
            'lora_configs',
            'num_inference_steps',
            'guidance_scale',
            'controlnet_scale',
            'models_dir',
            'outputs_colored_dir',
        ]
        
        for attr in attrs:
            if hasattr(config, attr):
                print(f"  ✓ {attr}")
            else:
                print(f"  ✗ {attr} not found")
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def main():
    print("=" * 60)
    print("ChromaManga MCP Server Integration Verification")
    print("=" * 60)
    
    # Check imports
    errors = check_imports()
    
    # Check MCP server
    mcp_ok = check_mcp_server()
    
    # Check config
    config_ok = check_config()
    
    # Summary
    print("\n" + "=" * 60)
    if not errors and mcp_ok and config_ok:
        print("✓ All checks passed!")
        print("\nYou can now:")
        print("1. Run: python test_full_pipeline.py <image_path>")
        print("2. Or start MCP server: python chromamanga_mcp_server.py")
        print("3. Or use in Kiro after restarting")
    else:
        print("✗ Some checks failed")
        if errors:
            print("\nMissing dependencies:")
            for error in errors:
                print(f"  - {error}")
            print("\nRun: pip install -r requirements.txt")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
