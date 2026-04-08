#!/usr/bin/env python3
"""
Comprehensive import test for ChromaManga project
Tests all modules after directory restructuring
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_core_modules():
    """Test core module imports"""
    print("\n=== Testing Core Modules ===")
    
    try:
        from core import config
        print("✓ core.config")
    except Exception as e:
        print(f"✗ core.config: {e}")
        return False
    
    try:
        from core.colorize import build_pipeline, colorize_regional
        print("✓ core.colorize")
    except Exception as e:
        print(f"✗ core.colorize: {e}")
        return False
    
    try:
        from core.task_manager import TaskManager
        print("✓ core.task_manager")
    except Exception as e:
        print(f"✗ core.task_manager: {e}")
        return False
    
    return True

def test_utils_modules():
    """Test utils module imports"""
    print("\n=== Testing Utils Modules ===")
    
    try:
        from utils.preprocess import preprocess_for_controlnet
        print("✓ utils.preprocess")
    except Exception as e:
        print(f"✗ utils.preprocess: {e}")
        return False
    
    try:
        from utils.postprocess import blend_lineart
        print("✓ utils.postprocess")
    except Exception as e:
        print(f"✗ utils.postprocess: {e}")
        return False
    
    try:
        from utils.character_reid import CharacterReID
        print("✓ utils.character_reid")
    except Exception as e:
        print(f"✗ utils.character_reid: {e}")
        return False
    
    try:
        from utils.regional_attention import set_regional_attn, reset_attn
        print("✓ utils.regional_attention")
    except Exception as e:
        print(f"✗ utils.regional_attention: {e}")
        return False
    
    return True

def test_mcp_server():
    """Test MCP server imports"""
    print("\n=== Testing MCP Server ===")
    
    try:
        # Check if MCP SDK is installed
        import mcp
        print("✓ mcp SDK installed")
    except ImportError:
        print("⚠ mcp SDK not installed (run: pip install -r mcp_server/mcp_requirements.txt)")
        return True  # Not a critical error for basic functionality
    
    try:
        from mcp_server.chromamanga_mcp_server import app
        print("✓ mcp_server.chromamanga_mcp_server")
    except Exception as e:
        print(f"✗ mcp_server.chromamanga_mcp_server: {e}")
        return False
    
    return True

def test_scripts():
    """Test script imports"""
    print("\n=== Testing Scripts ===")
    
    # We don't actually import scripts, just check they exist and have correct structure
    scripts = [
        "scripts/batch_colorize.py",
        "scripts/experiments/quick_denoise_test.py",
        "scripts/experiments/compare_denoise_methods.py",
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"✓ {script} exists")
        else:
            print(f"✗ {script} not found")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("ChromaManga Import Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Core Modules", test_core_modules()))
    results.append(("Utils Modules", test_utils_modules()))
    results.append(("MCP Server", test_mcp_server()))
    results.append(("Scripts", test_scripts()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! The project structure is correct.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
