"""
Compare traditional lineart extraction methods with LineartAnimeDetector
"""
import cv2
import numpy as np
from PIL import Image
import os

def extract_canny(image: Image.Image) -> Image.Image:
    """Canny edge detection"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Invert: white bg, black lines
    edges = 255 - edges
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def extract_sobel(image: Image.Image) -> Image.Image:
    """Sobel edge detection"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    # Invert: white bg, black lines
    sobel = 255 - sobel
    return Image.fromarray(cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB))

def extract_laplacian(image: Image.Image) -> Image.Image:
    """Laplacian edge detection"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.abs(laplacian) / np.abs(laplacian).max() * 255)
    # Invert: white bg, black lines
    laplacian = 255 - laplacian
    return Image.fromarray(cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB))

def extract_adaptive_threshold(image: Image.Image) -> Image.Image:
    """Adaptive threshold (same as scribble mode)"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=7
    )
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

def extract_otsu(image: Image.Image) -> Image.Image:
    """Otsu's binarization"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

def create_comparison_grid(input_path: str, lineart_anime_path: str, output_path: str):
    """Create comparison grid: traditional methods vs LineartAnimeDetector"""
    
    # Load images
    input_img = Image.open(input_path).convert("RGB")
    lineart_anime = Image.open(lineart_anime_path).convert("RGB")
    
    print(f"Input image size: {input_img.size}")
    
    # Extract using traditional methods
    print("Extracting with Canny...")
    canny = extract_canny(input_img)
    
    print("Extracting with Sobel...")
    sobel = extract_sobel(input_img)
    
    print("Extracting with Laplacian...")
    laplacian = extract_laplacian(input_img)
    
    print("Extracting with Adaptive Threshold...")
    adaptive = extract_adaptive_threshold(input_img)
    
    print("Extracting with Otsu...")
    otsu = extract_otsu(input_img)
    
    # Resize all to same size
    w, h = input_img.size
    
    # Create grid: 3 rows x 3 columns
    # Row 1: Original, Canny, Sobel
    # Row 2: Laplacian, Adaptive Threshold, Otsu
    # Row 3: LineartAnimeDetector (span 3 columns)
    
    padding = 10
    label_height = 40
    
    # Calculate grid dimensions
    cell_w = w
    cell_h = h + label_height
    grid_w = cell_w * 3 + padding * 4
    grid_h = cell_h * 2 + padding * 3 + (h + label_height) + padding
    
    # Create white canvas
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    
    # Helper function to add image with label
    def add_image_with_label(img, label, x, y):
        # Add label background
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        
        # Draw label background
        draw.rectangle([x, y, x + cell_w, y + label_height], fill=(240, 240, 240))
        
        # Draw label text
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Center text
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (cell_w - text_w) // 2
        text_y = y + (label_height - (bbox[3] - bbox[1])) // 2
        
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        
        # Paste image
        canvas.paste(img, (x, y + label_height))
    
    # Row 1
    y = padding
    add_image_with_label(input_img, "Original", padding, y)
    add_image_with_label(canny, "Canny Edge", padding * 2 + cell_w, y)
    add_image_with_label(sobel, "Sobel Edge", padding * 3 + cell_w * 2, y)
    
    # Row 2
    y = padding * 2 + cell_h
    add_image_with_label(laplacian, "Laplacian", padding, y)
    add_image_with_label(adaptive, "Adaptive Threshold", padding * 2 + cell_w, y)
    add_image_with_label(otsu, "Otsu Binarization", padding * 3 + cell_w * 2, y)
    
    # Row 3: LineartAnimeDetector (full width)
    y = padding * 3 + cell_h * 2
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    
    # Draw label background
    label_w = grid_w - padding * 2
    draw.rectangle([padding, y, padding + label_w, y + label_height], fill=(200, 220, 255))
    
    # Draw label text
    label = "LineartAnimeDetector (Deep Learning)"
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = padding + (label_w - text_w) // 2
    text_y = y + (label_height - (bbox[3] - bbox[1])) // 2
    
    draw.text((text_x, text_y), label, fill=(0, 0, 100), font=font)
    
    # Paste LineartAnimeDetector result (centered)
    lineart_x = (grid_w - w) // 2
    canvas.paste(lineart_anime, (lineart_x, y + label_height))
    
    # Save
    canvas.save(output_path, quality=95)
    print(f"\nComparison saved to: {output_path}")
    print(f"Grid size: {grid_w} x {grid_h}")

def create_side_by_side_comparison(input_path: str, lineart_anime_path: str, output_dir: str):
    """Create individual side-by-side comparisons for each method"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    input_img = Image.open(input_path).convert("RGB")
    lineart_anime = Image.open(lineart_anime_path).convert("RGB")
    
    methods = {
        "01_canny": ("Canny Edge", extract_canny),
        "02_sobel": ("Sobel Edge", extract_sobel),
        "03_laplacian": ("Laplacian", extract_laplacian),
        "04_adaptive": ("Adaptive Threshold", extract_adaptive_threshold),
        "05_otsu": ("Otsu Binarization", extract_otsu),
    }
    
    w, h = input_img.size
    padding = 20
    label_height = 50
    
    for filename, (label, extract_fn) in methods.items():
        print(f"Creating comparison: {label}...")
        
        # Extract lineart
        traditional = extract_fn(input_img)
        
        # Create side-by-side canvas
        canvas_w = w * 2 + padding * 3
        canvas_h = h + label_height + padding * 2
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        
        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_label = ImageFont.truetype("arial.ttf", 20)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
        
        # Title
        title = f"{label} vs LineartAnimeDetector"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        text_w = bbox[2] - bbox[0]
        text_x = (canvas_w - text_w) // 2
        draw.text((text_x, 10), title, fill=(0, 0, 0), font=font_title)
        
        # Left label
        draw.text((padding + 10, label_height - 5), label, fill=(100, 0, 0), font=font_label)
        
        # Right label
        draw.text((padding * 2 + w + 10, label_height - 5), "LineartAnimeDetector", fill=(0, 0, 100), font=font_label)
        
        # Paste images
        canvas.paste(traditional, (padding, label_height + padding))
        canvas.paste(lineart_anime, (padding * 2 + w, label_height + padding))
        
        # Save
        output_path = os.path.join(output_dir, f"{filename}.png")
        canvas.save(output_path, quality=95)
        print(f"  Saved: {output_path}")

if __name__ == "__main__":
    input_path = r"E:\code\graduationProject\ChromaManga\c1a670e964087446163d32eeab823613_regional\00_input.png"
    lineart_anime_path = r"E:\code\graduationProject\ChromaManga\c1a670e964087446163d32eeab823613_regional\02_lineart.png"
    
    # Check if files exist
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    if not os.path.exists(lineart_anime_path):
        print(f"Error: LineartAnime file not found: {lineart_anime_path}")
        exit(1)
    
    print("=" * 60)
    print("Lineart Extraction Methods Comparison")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Reference: {lineart_anime_path}")
    print()
    
    # Create grid comparison
    output_grid = "outputs/lineart_comparison_grid.png"
    os.makedirs("outputs", exist_ok=True)
    print("Creating grid comparison...")
    create_comparison_grid(input_path, lineart_anime_path, output_grid)
    
    print()
    print("-" * 60)
    print()
    
    # Create side-by-side comparisons
    output_dir = "outputs/lineart_side_by_side"
    print("Creating side-by-side comparisons...")
    create_side_by_side_comparison(input_path, lineart_anime_path, output_dir)
    
    print()
    print("=" * 60)
    print("All comparisons completed!")
    print("=" * 60)
    print(f"Grid comparison: {output_grid}")
    print(f"Side-by-side comparisons: {output_dir}/")
