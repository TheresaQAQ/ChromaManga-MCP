"""
去噪算法对比工具
对比不同去噪方法的效果，生成对比图像
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入现有的去噪函数
from utils.preprocess import (
    denoise_none,
    denoise_median,
    denoise_bilateral,
    denoise_nlmeans,
    denoise_median_bilateral,
    denoise_median_nlmeans,
    denoise_bilateral_strong,
    denoise_median_large,
    denoise_gaussian,
    denoise_morphology
)


def setup_chinese_font():
    """设置中文字体"""
    try:
        # Windows系统字体
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if os.path.exists(font_path):
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，标签可能显示为方框")


def compare_denoise_methods(input_path, output_dir="outputs/denoise_comparison"):
    """
    对比所有去噪方法的效果
    
    Args:
        input_path: 输入图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始图像
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    # 定义要对比的方法
    methods = {
        "原图": denoise_none,
        "中值滤波": denoise_median,
        "双边滤波": denoise_bilateral,
        "非局部均值": denoise_nlmeans,
        "中值+双边": denoise_median_bilateral,
        "中值+非局部均值": denoise_median_nlmeans,
        "双边滤波(强)": denoise_bilateral_strong,
        "中值滤波(大)": denoise_median_large,
        "高斯模糊": denoise_gaussian,
        "形态学": denoise_morphology
    }
    
    # 应用所有去噪方法
    results = {}
    print("\n应用去噪方法:")
    for name, method in methods.items():
        print(f"  - {name}...", end=" ")
        try:
            result = method(original_image)
            results[name] = result
            # 保存单独的结果
            result.save(os.path.join(output_dir, f"{name}.png"))
            print("✓")
        except Exception as e:
            print(f"✗ 错误: {e}")
            results[name] = original_image
    
    # 创建对比图
    print("\n生成对比图...")
    create_comparison_grid(results, output_dir)
    
    # 创建详细对比图（2x5网格）
    create_detailed_comparison(results, output_dir)
    
    print(f"\n完成! 结果保存在: {output_dir}")
    return results


def create_comparison_grid(results, output_dir):
    """创建网格对比图"""
    setup_chinese_font()
    
    n_methods = len(results)
    cols = 5
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle('去噪算法对比', fontsize=20, fontweight='bold')
    
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    for idx, (name, image) in enumerate(results.items()):
        axes[idx].imshow(image)
        axes[idx].set_title(name, fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存网格对比图: {output_path}")


def create_detailed_comparison(results, output_dir):
    """创建详细对比图，包含局部放大"""
    setup_chinese_font()
    
    # 选择主要方法进行详细对比
    main_methods = ["原图", "中值滤波", "双边滤波", "非局部均值", 
                    "中值+双边", "中值+非局部均值"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('主要去噪方法详细对比', fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, method_name in enumerate(main_methods):
        if method_name in results:
            image = results[method_name]
            axes[idx].imshow(image)
            axes[idx].set_title(method_name, fontsize=16, fontweight='bold')
            axes[idx].axis('off')
            
            # 添加图像质量信息
            img_array = np.array(image)
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)
            axes[idx].text(0.02, 0.98, f'均值: {mean_val:.1f}\n标准差: {std_val:.1f}',
                          transform=axes[idx].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "detailed_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存详细对比图: {output_path}")


def compare_with_crop(input_path, crop_box=None, output_dir="outputs/denoise_comparison"):
    """
    对比去噪方法，包含局部裁剪放大
    
    Args:
        input_path: 输入图像路径
        crop_box: 裁剪区域 (left, top, right, bottom)，None则自动选择中心区域
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    # 自动选择裁剪区域（中心区域）
    if crop_box is None:
        w, h = original_image.size
        crop_size = min(w, h) // 4
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        crop_box = (left, top, left + crop_size, top + crop_size)
    
    # 主要方法
    methods = {
        "原图": denoise_none,
        "中值滤波": denoise_median,
        "双边滤波": denoise_bilateral,
        "非局部均值": denoise_nlmeans,
        "中值+双边": denoise_median_bilateral,
        "中值+非局部均值": denoise_median_nlmeans
    }
    
    setup_chinese_font()
    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 4 * len(methods)))
    fig.suptitle('去噪算法对比（全图 vs 局部放大）', fontsize=18, fontweight='bold')
    
    print("\n处理图像:")
    for idx, (name, method) in enumerate(methods.items()):
        print(f"  - {name}...")
        result = method(original_image)
        cropped = result.crop(crop_box)
        
        # 全图
        axes[idx, 0].imshow(result)
        axes[idx, 0].set_title(f'{name} - 全图', fontsize=14)
        axes[idx, 0].axis('off')
        
        # 添加裁剪框标记
        from matplotlib.patches import Rectangle
        rect = Rectangle((crop_box[0], crop_box[1]), 
                         crop_box[2]-crop_box[0], crop_box[3]-crop_box[1],
                         linewidth=2, edgecolor='red', facecolor='none')
        axes[idx, 0].add_patch(rect)
        
        # 局部放大
        axes[idx, 1].imshow(cropped)
        axes[idx, 1].set_title(f'{name} - 局部放大', fontsize=14)
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_with_crop.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存对比图: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='去噪算法对比工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--output', '-o', default='outputs/denoise_comparison',
                       help='输出目录 (默认: outputs/denoise_comparison)')
    parser.add_argument('--crop', action='store_true',
                       help='生成包含局部放大的对比图')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("去噪算法对比工具")
    print("=" * 60)
    
    # 基础对比
    compare_denoise_methods(args.input, args.output)
    
    # 局部放大对比
    if args.crop:
        print("\n" + "=" * 60)
        print("生成局部放大对比图")
        print("=" * 60)
        compare_with_crop(args.input, output_dir=args.output)
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
