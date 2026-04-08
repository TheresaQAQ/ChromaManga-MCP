"""
Batch colorization script
Process all images in inputs/ directory
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.colorize import build_pipeline, colorize_regional
from core import config

def main():
    # Find all images in inputs/
    image_patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(config.inputs_dir, pattern)))
    
    if not image_files:
        print("No images found in inputs/ directory")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Build pipeline once
    print("\nBuilding pipeline...")
    pipe, reid = build_pipeline()
    
    # Process each image
    for i, input_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(image_files)}: {os.path.basename(input_path)}")
        print(f"{'='*60}")
        
        stem = Path(input_path).stem
        output_path = os.path.join(config.outputs_colored_dir, f"{stem}_colored.png")
        
        try:
            colorize_regional(pipe, input_path, output_path, reid=reid)
            print(f"✓ Saved: {output_path}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Results saved to: {config.outputs_colored_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
