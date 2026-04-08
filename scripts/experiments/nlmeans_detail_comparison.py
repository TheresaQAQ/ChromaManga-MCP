"""
非局部均值去噪详细对比工具
展示原图 vs 去噪结果，包含局部细节放大对比
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入去噪函数
from utils.preprocess import denoise_nlmeans


def setup_chinese_font():
    """设置中文字体"""
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if os.path.exists(font_path):
            plt.rcParams['font.sans-serif'] = ['SimHei']
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体")


def calculate_metrics(original, denoised):
    """计算图像质量指标"""
    orig_array = np.array(original)
    den_array = np.array(denoised)
    
    # 均值和标准差
    orig_mean = np.mean(orig_array)
    den_mean = np.mean(den_array)
    orig_std = np.std(orig_array)
    den_std = np.std(den_array)
    
    # MSE (均方误差)
    mse = np.mean((orig_array.astype(float) - den_array.astype(float)) ** 2)
    
    # PSNR (峰值信噪比)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    return {
        'orig_mean': orig_mean,
        'den_mean': den_mean,
        'orig_std': orig_std,
        'den_std': den_std,
        'mse': mse,
        'psnr': psnr
    }


def auto_select_crop_regions(image, num_regions=3):
    """
    自动选择有代表性的裁剪区域
    
    Args:
        image: PIL Image
        num_regions: 要选择的区域数量
    
    Returns:
        list of (left, top, right, bottom) tuples
    """
    w, h = image.size
    crop_size = min(w, h) // 4  # 裁剪区域大小为图像较小边的1/4
    
    regions = []
    
    # 区域1: 左上角（通常包含细节）
    regions.append((
        w // 6,
        h // 6,
        w // 6 + crop_size,
        h // 6 + crop_size
    ))
    
    # 区域2: 中心区域
    regions.append((
        (w - crop_size) // 2,
        (h - crop_size) // 2,
        (w + crop_size) // 2,
        (h + crop_size) // 2
    ))
    
    # 区域3: 右下角
    if num_regions >= 3:
        regions.append((
            w - w // 6 - crop_size,
            h - h // 6 - crop_size,
            w - w // 6,
            h - h // 6
        ))
    
    return regions[:num_regions]


def create_nlmeans_comparison(input_path, output_dir=None, crop_regions=None, num_crops=3):
    """
    创建非局部均值去噪的详细对比图（生成两张独立图片）
    
    Args:
        input_path: 输入图像路径
        output_dir: 输出目录
        crop_regions: 自定义裁剪区域列表 [(left, top, right, bottom), ...]
        num_crops: 自动选择的裁剪区域数量（当crop_regions为None时）
    
    Returns:
        tuple: (denoised_image, metrics, comparison_path, detail_path)
    """
    setup_chinese_font()
    
    # 读取原始图像
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    # 应用非局部均值去噪
    print("应用非局部均值去噪...")
    denoised_image = denoise_nlmeans(original_image)
    
    # 计算质量指标
    metrics = calculate_metrics(original_image, denoised_image)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    
    # 自动选择裁剪区域
    if crop_regions is None:
        crop_regions = auto_select_crop_regions(original_image, num_crops)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = "outputs/nlmeans_detail_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    # ========== 图1: 原图 vs 去噪结果左右对比 ==========
    print("\n生成图1: 原图 vs 去噪结果对比...")
    comparison_path = create_side_by_side_figure(
        original_image, denoised_image, metrics, crop_regions,
        os.path.join(output_dir, f"{basename}_comparison.png")
    )
    
    # ========== 图2: 局部细节放大对比 ==========
    print("生成图2: 局部细节放大对比...")
    detail_path = create_detail_crops_figure(
        original_image, denoised_image, crop_regions,
        os.path.join(output_dir, f"{basename}_details.png")
    )
    
    # 保存单独的去噪结果
    denoised_output = os.path.join(output_dir, f"{basename}_denoised.png")
    denoised_image.save(denoised_output)
    print(f"去噪结果已保存: {denoised_output}")
    
    return denoised_image, metrics, comparison_path, detail_path


def create_side_by_side_figure(original_image, denoised_image, metrics, crop_regions, output_path):
    """
    创建原图 vs 去噪结果的左右对比图（图1）
    """
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('非局部均值去噪对比', fontsize=22, fontweight='bold', y=0.98)
    
    # 第一行: 原图
    # 全图
    ax1 = plt.subplot(2, num_cols, 1)
    ax1.imshow(original_image)
    ax1.set_title('原图（全图）', fontsize=16, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # 在原图上标记裁剪区域
    colors = ['red', 'blue', 'green', 'yellow', 'cyan']
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        color = colors[idx % len(colors)]
        rect = Rectangle((left, top), right - left, bottom - top,
                        linewidth=3, edgecolor=color, facecolor='none',
                        linestyle='--')
        ax1.add_patch(rect)
        # 添加区域编号
        ax1.text(left + 10, top + 30, f'区域{idx+1}',
                fontsize=14, color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加图像统计信息
    info_text = (
        f"图像尺寸: {original_image.size[0]} × {original_image.size[1]}\n"
        f"均值: {metrics['orig_mean']:.2f}\n"
        f"标准差: {metrics['orig_std']:.2f}"
    )
    ax1.text(0.02, 0.98, info_text,
            transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 原图的裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        ax = plt.subplot(2, num_cols, idx + 2)
        cropped = original_image.crop((left, top, right, bottom))
        ax.imshow(cropped)
        color = colors[idx % len(colors)]
        ax.set_title(f'原图 - 区域{idx+1}', fontsize=14, fontweight='bold', 
                    color=color, pad=10)
        ax.axis('off')
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    # 第二行: 去噪结果
    # 全图
    ax2 = plt.subplot(2, num_cols, num_cols + 1)
    ax2.imshow(denoised_image)
    ax2.set_title('非局部均值去噪（全图）', fontsize=16, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # 在去噪图上也标记裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        color = colors[idx % len(colors)]
        rect = Rectangle((left, top), right - left, bottom - top,
                        linewidth=3, edgecolor=color, facecolor='none',
                        linestyle='--')
        ax2.add_patch(rect)
        ax2.text(left + 10, top + 30, f'区域{idx+1}',
                fontsize=14, color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加去噪后统计信息和质量指标
    info_text = (
        f"均值: {metrics['den_mean']:.2f}\n"
        f"标准差: {metrics['den_std']:.2f}\n\n"
        f"质量指标:\n"
        f"MSE: {metrics['mse']:.2f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB"
    )
    ax2.text(0.02, 0.98, info_text,
            transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # 去噪结果的裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        ax = plt.subplot(2, num_cols, num_cols + idx + 2)
        cropped = denoised_image.crop((left, top, right, bottom))
        ax.imshow(cropped)
        color = colors[idx % len(colors)]
        ax.set_title(f'去噪后 - 区域{idx+1}', fontsize=14, fontweight='bold',
                    color=color, pad=10)
        ax.axis('off')
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
    
    plt.tight_layout()
    
    # 保存图像
    if output_path is None:
        output_dir = "outputs/nlmeans_detail_comparison"
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_nlmeans_detail.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {output_path}")
    
    # 同时保存单独的去噪结果
    denoised_output = output_path.replace('_detail.png', '_denoised.png')
    denoised_image.save(denoised_output)
    print(f"去噪结果已保存: {denoised_output}")
    
    return denoised_image, metrics


def create_simple_comparison(input_path, output_path=None):
    """
    创建简单的左右对比图（无局部放大）
    
    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
    """
    setup_chinese_font()
    
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    print("应用非局部均值去噪...")
    denoised_image = denoise_nlmeans(original_image)
    
    metrics = calculate_metrics(original_image, denoised_image)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    
    # 创建左右对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('非局部均值去噪对比', fontsize=20, fontweight='bold', y=0.98)
    
    # 原图
    axes[0].imshow(original_image)
    axes[0].set_title('原图', fontsize=16, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    info_text = (
        f"图像尺寸: {original_image.size[0]} × {original_image.size[1]}\n"
        f"均值: {metrics['orig_mean']:.2f}\n"
        f"标准差: {metrics['orig_std']:.2f}"
    )
    axes[0].text(0.02, 0.98, info_text,
                transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 去噪结果
    axes[1].imshow(denoised_image)
    axes[1].set_title('非局部均值去噪结果', fontsize=16, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    info_text = (
        f"均值: {metrics['den_mean']:.2f}\n"
        f"标准差: {metrics['den_std']:.2f}\n\n"
        f"质量指标:\n"
        f"MSE: {metrics['mse']:.2f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB"
    )
    axes[1].text(0.02, 0.98, info_text,
                transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout()
    
    if output_path is None:
        output_dir = "outputs/nlmeans_detail_comparison"
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_nlmeans_simple.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {output_path}")
    
    return denoised_image, metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='非局部均值去噪详细对比工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成详细对比图（含3个局部放大区域）
  python nlmeans_detail_comparison.py inputs/test.png
  
  # 生成简单左右对比图（无局部放大）
  python nlmeans_detail_comparison.py inputs/test.png --simple
  
  # 指定局部放大区域数量
  python nlmeans_detail_comparison.py inputs/test.png --num-crops 2
  
  # 指定输出路径
  python nlmeans_detail_comparison.py inputs/test.png -o my_output.png
        """
    )
    
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出图像路径')
    parser.add_argument('--simple', action='store_true',
                       help='生成简单左右对比图（无局部放大）')
    parser.add_argument('--num-crops', type=int, default=3,
                       help='局部放大区域数量 (默认: 3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("非局部均值去噪详细对比工具")
    print("=" * 70)
    print()
    
    if args.simple:
        create_simple_comparison(args.input, args.output)
    else:
        create_nlmeans_comparison(args.input, args.output, num_crops=args.num_crops)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认测试图像
    if len(sys.argv) == 1:
        test_images = [
            "inputs/1f2ebb3f0192a22376e8981774375813.png",
            "inputs/553d7fe3f77db5f73264cad52de45117.png",
            "inputs/699428c8c8e94ee4751d33f3bb9c1234.png"
        ]
        
        input_path = None
        for img in test_images:
            if os.path.exists(img):
                input_path = img
                break
        
        if input_path:
            sys.argv = ['nlmeans_detail_comparison.py', input_path]
        else:
            print("错误: 找不到测试图像")
            print("用法: python nlmeans_detail_comparison.py <图像路径>")
            sys.exit(1)
    
    main()
