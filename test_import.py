#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    from core import config
    print("✓ core.config imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.config: {e}")

try:
    from core.colorize import build_pipeline
    print("✓ core.colorize.build_pipeline imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.colorize: {e}")

try:
    from core.task_manager import TaskManager
    print("✓ core.task_manager.TaskManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import core.task_manager: {e}")

print("\nAll imports successful! The module structure is working correctly.")
