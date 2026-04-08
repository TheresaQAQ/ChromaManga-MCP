"""
去噪算法对比工具 - 左右对比版
为每个去噪算法生成单独的对比图（原图 vs 去噪结果）
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


def create_side_by_side_comparison(original_image, denoised_image, method_name, 
                                   output_path, metrics=None):
    """
    创建左右对比图
    
    Args:
        original_image: 原始图像
        denoised_image: 去噪后图像
        method_name: 方法名称
        output_path: 输出路径
        metrics: 图像质量指标
    """
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'去噪对比: {method_name}', fontsize=20, fontweight='bold', y=0.98)
    
    # 原图
    axes[0].imshow(original_image)
    axes[0].set_title('原图', fontsize=16, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # 去噪结果
    axes[1].imshow(denoised_image)
    axes[1].set_title(f'{method_name} 去噪结果', fontsize=16, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    # 添加图像质量信息
    if metrics:
        info_text = (
            f"原图统计:\n"
            f"  均值: {metrics['orig_mean']:.2f}\n"
            f"  标准差: {metrics['orig_std']:.2f}\n\n"
            f"去噪后统计:\n"
            f"  均值: {metrics['den_mean']:.2f}\n"
            f"  标准差: {metrics['den_std']:.2f}\n\n"
            f"差异指标:\n"
            f"  MSE: {metrics['mse']:.2f}\n"
            f"  PSNR: {metrics['psnr']:.2f} dB"
        )
        
        axes[0].text(0.02, 0.02, info_text,
                    transform=axes[0].transAxes,
                    fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_comparisons(input_path, output_dir="outputs/side_by_side_comparison"):
    """
    为所有去噪方法生成左右对比图
    
    Args:
        input_path: 输入图像路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始图像
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    # 定义所有去噪方法
    methods = {
        "01_median": ("中值滤波", denoise_median),
        "02_bilateral": ("双边滤波", denoise_bilateral),
        "03_nlmeans": ("非局部均值", denoise_nlmeans),
        "04_median_bilateral": ("中值+双边", denoise_median_bilateral),
        "05_median_nlmeans": ("中值+非局部均值", denoise_median_nlmeans),
        "06_bilateral_strong": ("双边滤波(强)", denoise_bilateral_strong),
        "07_median_large": ("中值滤波(大)", denoise_median_large),
        "08_gaussian": ("高斯模糊", denoise_gaussian),
        "09_morphology": ("形态学", denoise_morphology),
    }
    
    print("\n生成对比图:")
    print("=" * 70)
    
    for filename_prefix, (method_name, method_func) in methods.items():
        print(f"{method_name:20s} ", end="", flush=True)
        
        try:
            # 应用去噪
            denoised_image = method_func(original_image)
            
            # 计算指标
            metrics = calculate_metrics(original_image, denoised_image)
            
            # 生成对比图
            output_path = os.path.join(output_dir, f"{filename_prefix}_{method_name}.png")
            create_side_by_side_comparison(
                original_image, 
                denoised_image, 
                method_name, 
                output_path,
                metrics
            )
            
            print(f"✓ PSNR: {metrics['psnr']:6.2f} dB | {output_path}")
            
        except Exception as e:
            print(f"✗ 错误: {e}")
    
    print("=" * 70)
    print(f"\n完成! 所有对比图保存在: {output_dir}")
    print(f"共生成 {len(methods)} 张对比图")


def generate_single_comparison(input_path, method_name, output_dir="outputs/side_by_side_comparison"):
    """
    为单个去噪方法生成对比图
    
    Args:
        input_path: 输入图像路径
        method_name: 方法名称
        output_dir: 输出目录
    """
    methods_map = {
        "median": ("中值滤波", denoise_median),
        "bilateral": ("双边滤波", denoise_bilateral),
        "nlmeans": ("非局部均值", denoise_nlmeans),
        "median_bilateral": ("中值+双边", denoise_median_bilateral),
        "median_nlmeans": ("中值+非局部均值", denoise_median_nlmeans),
        "bilateral_strong": ("双边滤波(强)", denoise_bilateral_strong),
        "median_large": ("中值滤波(大)", denoise_median_large),
        "gaussian": ("高斯模糊", denoise_gaussian),
        "morphology": ("形态学", denoise_morphology),
    }
    
    if method_name not in methods_map:
        print(f"错误: 未知的方法 '{method_name}'")
        print(f"可用方法: {', '.join(methods_map.keys())}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"读取图像: {input_path}")
    original_image = Image.open(input_path).convert("RGB")
    
    display_name, method_func = methods_map[method_name]
    print(f"应用去噪方法: {display_name}")
    
    denoised_image = method_func(original_image)
    metrics = calculate_metrics(original_image, denoised_image)
    
    output_path = os.path.join(output_dir, f"{method_name}_comparison.png")
    create_side_by_side_comparison(
        original_image, 
        denoised_image, 
        display_name, 
        output_path,
        metrics
    )
    
    print(f"\n完成! 对比图保存在: {output_path}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")


def create_index_html(output_dir):
    """创建HTML索引页面，方便查看所有对比图"""
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>去噪算法对比</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .comparison {
            background: white;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison h2 {
            margin-top: 0;
            color: #2c3e50;
        }
        .comparison img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>去噪算法对比结果</h1>
"""
    
    # 获取所有PNG文件
    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    for png_file in png_files:
        method_name = png_file.replace('.png', '').split('_', 1)[1] if '_' in png_file else png_file
        html_content += f"""
        <div class="comparison">
            <h2>{method_name}</h2>
            <img src="{png_file}" alt="{method_name}">
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n已生成HTML索引: {html_path}")
    print("在浏览器中打开此文件可以方便地查看所有对比图")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='去噪算法左右对比工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有方法的对比图
  python denoise_side_by_side.py inputs/test.png
  
  # 只生成特定方法的对比图
  python denoise_side_by_side.py inputs/test.png --method nlmeans
  
  # 指定输出目录
  python denoise_side_by_side.py inputs/test.png -o my_output
  
  # 生成HTML索引页面
  python denoise_side_by_side.py inputs/test.png --html

可用方法:
  median, bilateral, nlmeans, median_bilateral, median_nlmeans,
  bilateral_strong, median_large, gaussian, morphology
        """
    )
    
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('--method', '-m', help='指定单个去噪方法')
    parser.add_argument('--output', '-o', default='outputs/side_by_side_comparison',
                       help='输出目录 (默认: outputs/side_by_side_comparison)')
    parser.add_argument('--html', action='store_true',
                       help='生成HTML索引页面')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        sys.exit(1)
    
    print("=" * 70)
    print("去噪算法左右对比工具")
    print("=" * 70)
    print()
    
    if args.method:
        # 生成单个方法的对比图
        generate_single_comparison(args.input, args.method, args.output)
    else:
        # 生成所有方法的对比图
        generate_all_comparisons(args.input, args.output)
    
    if args.html:
        create_index_html(args.output)
    
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
            sys.argv = ['denoise_side_by_side.py', input_path, '--html']
        else:
            print("错误: 找不到测试图像")
            print("用法: python denoise_side_by_side.py <图像路径>")
            sys.exit(1)
    
    main()
