"""
快速去噪测试脚本
简单对比几种主要去噪方法
"""
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.preprocess import (
    denoise_none,
    denoise_median,
    denoise_bilateral,
    denoise_nlmeans,
    denoise_median_bilateral
)


def quick_test(input_path):
    """快速测试主要去噪方法"""
    
    # 创建输出目录
    output_dir = "outputs/quick_denoise_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"读取图像: {input_path}")
    original = Image.open(input_path).convert("RGB")
    
    # 测试方法
    methods = {
        "1_original": ("原图", denoise_none),
        "2_median": ("中值滤波", denoise_median),
        "3_bilateral": ("双边滤波", denoise_bilateral),
        "4_nlmeans": ("非局部均值", denoise_nlmeans),
        "5_median_bilateral": ("中值+双边", denoise_median_bilateral)
    }
    
    print("\n处理中...")
    print("-" * 50)
    
    for filename, (name, method) in methods.items():
        print(f"{name:15s} ", end="", flush=True)
        try:
            result = method(original)
            output_path = os.path.join(output_dir, f"{filename}.png")
            result.save(output_path)
            print(f"✓ 已保存: {output_path}")
        except Exception as e:
            print(f"✗ 错误: {e}")
    
    print("-" * 50)
    print(f"\n完成! 结果保存在: {output_dir}")
    print("\n提示: 使用图像查看器打开输出目录，对比查看效果")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # 使用默认测试图像
        test_images = [
            "inputs/1f2ebb3f0192a22376e8981774375813.png",
            "inputs/553d7fe3f77db5f73264cad52de45117.png",
            "inputs/699428c8c8e94ee4751d33f3bb9c1234.png"
        ]
        
        # 找到第一个存在的图像
        input_path = None
        for img in test_images:
            if os.path.exists(img):
                input_path = img
                break
        
        if input_path is None:
            print("错误: 找不到测试图像")
            print("用法: python quick_denoise_test.py <图像路径>")
            sys.exit(1)
    else:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"错误: 找不到文件 {input_path}")
            sys.exit(1)
    
    quick_test(input_path)
