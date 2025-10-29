"""
DCT图像压缩实验 - 实验一
功能：对512x512灰度图像进行8x8分块DCT变换，只保留DC分量进行重建
作者：滕彦翕-2023302616
"""

import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_dct_matrix(n):
    """
    创建n×n的DCT变换矩阵
    
    DCT变换公式:
    C[i,j] = α(i) * cos[(2j+1)iπ/(2n)]
    其中 α(i) = 1/√n (当i=0), √(2/n) (当i≠0)
    
    参数:
        n: 矩阵大小
    返回:
        dct_matrix: n×n的DCT变换矩阵
    """
    dct_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            # 归一化系数
            if i == 0:
                alpha_i = np.sqrt(1.0 / n)
            else:
                alpha_i = np.sqrt(2.0 / n)
            
            # DCT基函数
            dct_matrix[i, j] = alpha_i * np.cos((2 * j + 1) * i * np.pi / (2 * n))
    
    return dct_matrix

def dct_2d(block):
    """
    实现2D DCT变换
    
    公式: F = C * f * C^T
    其中 f 是原始块，F 是DCT系数，C 是DCT变换矩阵
    
    参数:
        block: 输入图像块 (n×n)
    返回:
        dct_block: DCT系数块 (n×n)
    """
    n = block.shape[0]
    dct_matrix = create_dct_matrix(n)
    
    # 应用DCT变换: F = C * f * C^T
    dct_block = dct_matrix @ block @ dct_matrix.T
    return dct_block

def idct_2d(dct_block):
    """
    实现2D逆DCT变换
    
    公式: f = C^T * F * C
    其中 F 是DCT系数，f 是重建块，C 是DCT变换矩阵
    
    参数:
        dct_block: DCT系数块 (n×n)
    返回:
        block: 重建图像块 (n×n)
    """
    n = dct_block.shape[0]
    dct_matrix = create_dct_matrix(n)
    
    # 应用逆DCT变换: f = C^T * F * C
    block = dct_matrix.T @ dct_block @ dct_matrix
    return block

def calculate_quality_metrics(original, reconstructed):
    """
    计算所有质量评价指标: MSE, RMSE, SNR, PSNR
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
    返回:
        metrics: 包含所有指标的字典
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    # MSE - Mean Square Error (均方误差)
    mse = np.mean((original - reconstructed) ** 2)
    
    # RMSE - Root Mean Square Error (均方根误差)
    rmse = np.sqrt(mse) if mse > 0 else 0
    
    # SNR - Signal-to-Noise Ratio (信噪比)
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # PSNR - Peak Signal-to-Noise Ratio (峰值信噪比)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse) if mse > 0 else float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'SNR': snr,
        'PSNR': psnr
    }

def calculate_compression_ratio(original_size, compressed_size):
    """
    计算压缩比
    
    参数:
        original_size: 原始数据大小
        compressed_size: 压缩后数据大小
    返回:
        ratio: 压缩比
    """
    return original_size / compressed_size if compressed_size > 0 else 0

def process_image(image_path="test_image.jpg", block_size=8):
    """
    对图像进行DCT变换处理，只保留DC分量
    
    参数:
        image_path: 图像文件路径
        block_size: 分块大小 (默认8x8)
    返回:
        gray: 原始灰度图像
        result: 处理后的图像
        stats: 统计信息字典
    """
    print("="*60)
    print("DCT图像压缩实验 - 只保留DC分量")
    print("="*60)
    
    # 1. 读取图像
    if not os.path.exists(image_path):
        print(f"[错误] 找不到图像文件 '{image_path}'")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"   请确保图像文件存在")
        return None, None, None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取图像文件 '{image_path}'")
        return None, None, None
    
    print(f"[成功] 读取图像: {image_path}")
    
    # 2. 转换为灰度图像
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"   图像转换: 彩色 → 灰度")
    else:
        gray = img.copy()
        print(f"   图像类型: 灰度")
    
    # 3. 调整图像大小
    original_shape = gray.shape
    if gray.shape != (512, 512):
        print(f"   原始大小: {gray.shape[0]}x{gray.shape[1]}")
        gray = cv2.resize(gray, (512, 512))
        print(f"   调整为: 512x512")
    else:
        print(f"   图像大小: 512x512")
    
    # 4. 分块DCT处理
    result = gray.copy().astype(np.float64)
    print(f"\n[处理] 开始DCT变换处理...")
    print(f"   分块大小: {block_size}x{block_size}")
    
    num_blocks_h = gray.shape[0] // block_size
    num_blocks_w = gray.shape[1] // block_size
    total_blocks = num_blocks_h * num_blocks_w
    print(f"   总块数: {total_blocks} ({num_blocks_h}x{num_blocks_w})")
    
    # 统计信息
    total_coefficients = total_blocks * block_size * block_size
    kept_coefficients = total_blocks  # 每块只保留1个DC分量
    
    processed_blocks = 0
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            # 提取块
            block = gray[i:i+block_size, j:j+block_size].astype(np.float64)
            
            # DCT变换
            dct_block = dct_2d(block)
            
            # 只保留DC分量
            dct_block_processed = np.zeros_like(dct_block)
            dct_block_processed[0, 0] = dct_block[0, 0]
            
            # 逆DCT变换
            processed_block = idct_2d(dct_block_processed)
            
            # 放回结果图像
            result[i:i+block_size, j:j+block_size] = processed_block
            
            processed_blocks += 1
            # 显示进度
            if processed_blocks % 512 == 0:
                progress = (processed_blocks / total_blocks) * 100
                print(f"   处理进度: {progress:.1f}% ({processed_blocks}/{total_blocks})")
    
    # 5. 像素值裁剪
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    print(f"\n[完成] 图像处理完成！")
    
    # 6. 统计信息
    stats = {
        'total_coefficients': total_coefficients,
        'kept_coefficients': kept_coefficients,
        'compression_ratio': total_coefficients / kept_coefficients,
        'kept_percentage': (kept_coefficients / total_coefficients) * 100,
        'num_blocks': total_blocks
    }
    
    return gray, result, stats

def visualize_results(gray, result, stats, psnr):
    """
    可视化实验结果
    
    参数:
        gray: 原始图像
        result: 处理后图像
        stats: 统计信息
        psnr: PSNR值
    """
    # 创建图形 - 改为2x2布局
    fig = plt.figure(figsize=(14, 10))
    
    # 第一行：图像对比
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image (512x512 Grayscale)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.title('Reconstructed Image (DC Only)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 第二行：统计信息和直方图
    ax3 = plt.subplot(2, 2, 3)
    # 统计信息文本 - 使用英文避免字体问题
    # 添加所有质量指标
    metrics = calculate_quality_metrics(gray, result)
    
    info_text = f"""
    [Compression Statistics]
    Total Coefficients: {stats['total_coefficients']:,}
    Kept Coefficients: {stats['kept_coefficients']:,}
    Compression Ratio: {stats['compression_ratio']:.2f}:1
    Kept Percentage: {stats['kept_percentage']:.4f}%
    
    [Quality Assessment]
    MSE:  {metrics['MSE']:.4f}
    RMSE: {metrics['RMSE']:.4f}
    SNR:  {metrics['SNR']:.2f} dB
    PSNR: {metrics['PSNR']:.2f} dB
    
    [Processing Parameters]
    Image Size: 512x512
    Block Size: 8x8
    Total Blocks: {stats['num_blocks']}
    """
    ax3.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    ax3.axis('off')
    
    # 直方图对比
    ax4 = plt.subplot(2, 2, 4)
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color='blue', 
             alpha=0.5, label='Original')
    plt.hist(result.ravel(), bins=256, range=(0, 256), color='red', 
             alpha=0.5, label='Reconstructed')
    plt.title('Pixel Value Distribution Comparison', fontsize=11)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('DCT Image Compression Experiment - DC Component Only', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """主函数"""
    # 处理图像
    original, processed, stats = process_image("test_image.jpg")
    
    if original is None:
        print("\n[失败] 程序执行失败，请检查图像文件路径")
        return
    
    # 计算所有质量指标
    metrics = calculate_quality_metrics(original, processed)
    
    # 主观质量评级
    psnr = metrics['PSNR']
    if psnr == float('inf'):
        quality_rating = "Perfect"
    elif psnr > 40:
        quality_rating = "Excellent"
    elif psnr > 30:
        quality_rating = "Good"
    elif psnr > 25:
        quality_rating = "Acceptable"
    elif psnr > 20:
        quality_rating = "Fair"
    else:
        quality_rating = "Poor"
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Experiment Results - Statistics")
    print("="*60)
    print(f"[Compression Statistics]")
    print(f"  Total DCT Coefficients: {stats['total_coefficients']:,}")
    print(f"  Kept Coefficients: {stats['kept_coefficients']:,} (DC only)")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}:1")
    print(f"  Kept Percentage: {stats['kept_percentage']:.4f}%")
    print(f"\n[Quality Assessment - Objective Metrics]")
    print(f"  MSE  (Mean Square Error):        {metrics['MSE']:.4f}")
    print(f"  RMSE (Root Mean Square Error):   {metrics['RMSE']:.4f}")
    print(f"  SNR  (Signal-to-Noise Ratio):    {metrics['SNR']:.2f} dB")
    print(f"  PSNR (Peak Signal-to-Noise):     {metrics['PSNR']:.2f} dB")
    print(f"\n[Quality Assessment - Subjective Rating]")
    print(f"  Quality Grade: {quality_rating}")
    print(f"\n[Processing Parameters]")
    print(f"  Total Blocks Processed: {stats['num_blocks']}")
    print("="*60)
    
    # 保存处理后的图像
    output_path = "processed_image.jpg"
    cv2.imwrite(output_path, processed)
    print(f"\n[保存] 处理后的图像已保存: {output_path}")
    
    # 可视化结果
    print(f"[可视化] 正在生成可视化结果...")
    fig = visualize_results(original, processed, stats, psnr)
    
    # 保存可视化结果
    viz_path = "dct_result_visualization.png"
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"[保存] 可视化结果已保存: {viz_path}")
    
    # 显示图像
    try:
        plt.show()
        print("\n[完成] 实验完成!")
    except Exception as e:
        print(f"\n[警告] 无法显示图形窗口: {e}")
        print(f"   请查看保存的图像文件")

# 运行实验
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断程序执行")
    except Exception as e:
        print(f"\n\n[错误] 程序执行出错: {e}")
        import traceback
        traceback.print_exc()