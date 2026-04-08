"""
非局部均值去噪对比工具（分离版）
生成两张独立的图片：
1. 原图 vs 去噪结果的左右对比图
2. 局部细节放大对比图
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
        # 尝试多个中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                from matplotlib import font_manager
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return
        
        # 如果都找不到，使用系统默认
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字体设置警告: {e}")
        plt.rcParams['axes.unicode_minus'] = False


def calculate_metrics(original, denoised):
    """计算图像质量指标"""
    orig_array = np.array(original)
    den_array = np.array(denoised)
    
    orig_mean = np.mean(orig_array)
    den_mean = np.mean(den_array)
    orig_std = np.std(orig_array)
    den_std = np.std(den_array)
    
    mse = np.mean((orig_array.astype(float) - den_array.astype(float)) ** 2)
    
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
    """自动选择有代表性的裁剪区域"""
    w, h = image.size
    crop_size = min(w, h) // 4
    
    regions = []
    
    # 区域1: 左上角
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


def create_comparison_figure(original_image, denoised_image, metrics, crop_regions, output_path):
    """
    创建图1: 原图 vs 去噪结果的左右对比图（标注裁剪区域）
    """
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Non-Local Means Denoising Comparison', fontsize=22, fontweight='bold', y=0.98)
    
    colors = ['red', 'blue', 'green', 'yellow', 'cyan']
    
    # 左侧: 原图
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=18, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # 在原图上标记裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        color = colors[idx % len(colors)]
        rect = Rectangle((left, top), right - left, bottom - top,
                        linewidth=4, edgecolor=color, facecolor='none',
                        linestyle='--')
        axes[0].add_patch(rect)
        axes[0].text(left + 15, top + 40, f'Region {idx+1}',
                    fontsize=16, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                             edgecolor=color, linewidth=2))
    
    # 添加原图统计信息（左上角）
    info_text = (
        f"Image Size: {original_image.size[0]} x {original_image.size[1]}\n"
        f"Mean: {metrics['orig_mean']:.2f}\n"
        f"Std Dev: {metrics['orig_std']:.2f}\n\n"
        f"Comparison Metrics:\n"
        f"MSE: {metrics['mse']:.2f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB"
    )
    axes[0].text(0.02, 0.98, info_text,
                transform=axes[0].transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
                family='monospace')
    
    # 右侧: 去噪结果
    axes[1].imshow(denoised_image)
    axes[1].set_title('NLMeans Denoised Result', fontsize=18, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # 在去噪图上也标记裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        color = colors[idx % len(colors)]
        rect = Rectangle((left, top), right - left, bottom - top,
                        linewidth=4, edgecolor=color, facecolor='none',
                        linestyle='--')
        axes[1].add_patch(rect)
        axes[1].text(left + 15, top + 40, f'Region {idx+1}',
                    fontsize=16, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                             edgecolor=color, linewidth=2))
    
    # 添加去噪后统计信息和质量指标（左上角）
    info_text = (
        f"Image Size: {original_image.size[0]} x {original_image.size[1]}\n"
        f"Mean: {metrics['den_mean']:.2f}\n"
        f"Std Dev: {metrics['den_std']:.2f}\n\n"
        f"Comparison Metrics:\n"
        f"MSE: {metrics['mse']:.2f}\n"
        f"PSNR: {metrics['psnr']:.2f} dB"
    )
    axes[1].text(0.02, 0.98, info_text,
                transform=axes[1].transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.95),
                family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_detail_figure(original_image, denoised_image, crop_regions, output_path):
    """
    创建图2: 局部细节放大对比图
    """
    setup_chinese_font()
    
    num_regions = len(crop_regions)
    colors = ['red', 'blue', 'green', 'yellow', 'cyan']
    
    # 布局: 2行 x num_regions列
    fig, axes = plt.subplots(2, num_regions, figsize=(6 * num_regions, 12))
    fig.suptitle('Detail Comparison (Zoomed Regions)', fontsize=22, fontweight='bold', y=0.98)
    
    # 如果只有一个区域，axes需要reshape
    if num_regions == 1:
        axes = axes.reshape(2, 1)
    
    # 第一行: 原图的裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        cropped = original_image.crop((left, top, right, bottom))
        color = colors[idx % len(colors)]
        
        axes[0, idx].imshow(cropped)
        axes[0, idx].set_title(f'Original - Region {idx+1}', 
                              fontsize=16, fontweight='bold', color=color, pad=15)
        axes[0, idx].axis('off')
        
        # 添加彩色边框
        for spine in axes[0, idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)
    
    # 第二行: 去噪后的裁剪区域
    for idx, (left, top, right, bottom) in enumerate(crop_regions):
        cropped = denoised_image.crop((left, top, right, bottom))
        color = colors[idx % len(colors)]
        
        axes[1, idx].imshow(cropped)
        axes[1, idx].set_title(f'Denoised - Region {idx+1}', 
                              fontsize=16, fontweight='bold', color=color, pad=15)
        axes[1, idx].axis('off')
        
        # 添加彩色边框
        for spine in axes[1, idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def process_image(input_path, output_dir=None, num_crops=3):
    """
    处理图像，生成两张独立的对比图
    
    Args:
        input_path: 输入图像路径
        output_dir: 输出目录
        num_crops: 裁剪区域数量
    
    Returns:
        dict: 包含所有输出路径和指标的字典
    """
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
    crop_regions = auto_select_crop_regions(original_image, num_crops)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = "outputs/nlmeans_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 生成图1: 左右对比图
    print("\n生成图1: 原图 vs 去噪结果对比...")
    comparison_path = os.path.join(output_dir, f"{basename}_comparison.png")
    create_comparison_figure(original_image, denoised_image, metrics, 
                            crop_regions, comparison_path)
    print(f"  已保存: {comparison_path}")
    
    # 生成图2: 局部细节放大图
    print("生成图2: 局部细节放大对比...")
    detail_path = os.path.join(output_dir, f"{basename}_details.png")
    create_detail_figure(original_image, denoised_image, crop_regions, detail_path)
    print(f"  已保存: {detail_path}")
    
    # 保存单独的去噪结果
    denoised_path = os.path.join(output_dir, f"{basename}_denoised.png")
    denoised_image.save(denoised_path)
    print(f"去噪结果已保存: {denoised_path}")
    
    return {
        'comparison': comparison_path,
        'details': detail_path,
        'denoised': denoised_path,
        'metrics': metrics
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='非局部均值去噪对比工具（分离版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成两张独立的对比图（默认3个裁剪区域）
  python nlmeans_comparison_split.py inputs/test.png
  
  # 指定裁剪区域数量
  python nlmeans_comparison_split.py inputs/test.png --num-crops 2
  
  # 指定输出目录
  python nlmeans_comparison_split.py inputs/test.png -o my_output

输出文件:
  - {basename}_comparison.png  : 原图 vs 去噪结果左右对比
  - {basename}_details.png     : 局部细节放大对比
  - {basename}_denoised.png    : 去噪结果图像
        """
    )
    
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--num-crops', type=int, default=3,
                       help='局部放大区域数量 (默认: 3)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("非局部均值去噪对比工具（分离版）")
    print("=" * 70)
    print()
    
    result = process_image(args.input, args.output, args.num_crops)
    
    print("\n" + "=" * 70)
    print("完成! 生成的文件:")
    print(f"  1. 对比图: {result['comparison']}")
    print(f"  2. 细节图: {result['details']}")
    print(f"  3. 去噪图: {result['denoised']}")
    print(f"\nPSNR: {result['metrics']['psnr']:.2f} dB")
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
            sys.argv = ['nlmeans_comparison_split.py', input_path]
        else:
            print("错误: 找不到测试图像")
            print("用法: python nlmeans_comparison_split.py <图像路径>")
            sys.exit(1)
    
    main()
