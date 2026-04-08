#!/usr/bin/env python3
"""
Test configuration paths
"""

import os
from core import config

print("=" * 60)
print("Configuration Paths Test")
print("=" * 60)

paths_to_check = [
    ("BASE_DIR", config.BASE_DIR),
    ("inputs_dir", config.inputs_dir),
    ("outputs_dir", config.outputs_dir),
    ("outputs_colored_dir", config.outputs_colored_dir),
    ("models_dir", config.models_dir),
    ("loras_dir", config.loras_dir),
]

print("\nConfigured Paths:")
for name, path in paths_to_check:
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"{exists} {name}: {path}")

print("\n" + "=" * 60)

# Check if input images exist
print("\nInput Images:")
if os.path.exists(config.inputs_dir):
    images = [f for f in os.listdir(config.inputs_dir) 
              if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images in {config.inputs_dir}")
    for img in images[:5]:  # Show first 5
        print(f"  • {img}")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")
else:
    print(f"✗ Input directory not found: {config.inputs_dir}")

print("\n" + "=" * 60)
print("✓ Configuration test complete")
