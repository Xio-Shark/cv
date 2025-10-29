"""
DCT图像压缩实验 - 实验二
功能：对整个512x512灰度图像进行DCT变换，只保留DC分量进行重建
特点：与实验一不同，这里对整幅图像做一次DCT，而非分块处理
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
from PIL import Image

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

def dct_2d(image):
    """
    实现2D DCT变换
    
    公式: F = C * f * C^T
    其中 f 是原始图像，F 是DCT系数，C 是DCT变换矩阵
    
    参数:
        image: 输入图像 (MxN)
    返回:
        dct_coeffs: DCT系数 (MxN)
    """
    M, N = image.shape
    C_m = create_dct_matrix(M)
    C_n = create_dct_matrix(N)
    
    # 应用DCT变换: F = C_m * image * C_n^T
    dct_coeffs = C_m @ image @ C_n.T
    return dct_coeffs

def idct_2d(dct_coeffs):
    """
    实现2D逆DCT变换
    
    公式: f = C^T * F * C
    其中 F 是DCT系数，f 是重建图像，C 是DCT变换矩阵
    
    参数:
        dct_coeffs: DCT系数 (MxN)
    返回:
        image: 重建图像 (MxN)
    """
    M, N = dct_coeffs.shape
    C_m = create_dct_matrix(M)
    C_n = create_dct_matrix(N)
    
    # 应用逆DCT变换: f = C_m^T * F * C_n
    image = C_m.T @ dct_coeffs @ C_n
    return image

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

def load_image(image_path):
    """
    加载图像并转换为灰度
    
    参数:
        image_path: 图像路径
    返回:
        gray: 灰度图像数组
    """
    if not os.path.exists(image_path):
        print(f"[错误] 找不到图像文件: {image_path}")
        return None
    
    # 使用PIL读取图像
    img = Image.open(image_path)
    
    # 转换为灰度
    gray = img.convert('L')
    
    # 转换为numpy数组
    gray_array = np.array(gray, dtype=np.float64)
    
    return gray_array

def process_whole_image(image_path="test_image.jpg"):
    """
    对整幅图像进行DCT变换，只保留DC分量
    
    参数:
        image_path: 图像文件路径
    返回:
        original: 原始图像
        result: 处理后的图像
        stats: 统计信息字典
    """
    print("="*60)
    print("DCT图像压缩实验 - 整图DCT只保留DC分量")
    print("="*60)
    
    # 1. 加载图像
    gray = load_image(image_path)
    if gray is None:
        return None, None, None
    
    print(f"[成功] 读取图像: {image_path}")
    print(f"   原始大小: {gray.shape[0]}x{gray.shape[1]}")
    
    # 2. 调整为512x512
    if gray.shape != (512, 512):
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(gray.astype(np.uint8))
        img_pil = img_pil.resize((512, 512))
        gray = np.array(img_pil, dtype=np.float64)
        print(f"   调整为: 512x512")
    else:
        print(f"   图像大小: 512x512")
    
    # 3. 对整幅图像进行DCT变换
    print(f"\n[处理] 开始整图DCT变换...")
    print(f"   注意: 这是对整个512x512图像做一次DCT变换")
    print(f"   与实验一的8x8分块DCT不同")
    
    dct_coeffs = dct_2d(gray)
    print(f"   DCT变换完成")
    
    # 4. 只保留DC分量
    dc_value = dct_coeffs[0, 0]
    print(f"   DC分量值: {dc_value:.2f}")
    
    # 创建只有DC分量的系数矩阵
    dct_dc_only = np.zeros_like(dct_coeffs)
    dct_dc_only[0, 0] = dc_value
    
    # 5. 逆DCT重建
    print(f"[处理] 开始逆DCT变换重建...")
    result = idct_2d(dct_dc_only)
    
    # 6. 裁剪到有效范围
    result = np.clip(result, 0, 255)
    
    print(f"\n[完成] 图像处理完成!")
    
    # 7. 统计信息
    total_coefficients = 512 * 512
    kept_coefficients = 1  # 只保留DC分量
    
    stats = {
        'total_coefficients': total_coefficients,
        'kept_coefficients': kept_coefficients,
        'compression_ratio': total_coefficients / kept_coefficients,
        'kept_percentage': (kept_coefficients / total_coefficients) * 100,
        'dc_value': dc_value
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
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    
    # 原始图像
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image (512x512 Grayscale)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 重建图像
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Reconstructed Image (DC Only)\nDC Value: {stats["dc_value"]:.2f}', 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 统计信息
    ax3 = plt.subplot(2, 2, 3)
    # 添加所有质量指标
    metrics = calculate_quality_metrics(gray, result)
    
    info_text = f"""
    [Compression Statistics]
    Total Coefficients: {stats['total_coefficients']:,}
    Kept Coefficients: {stats['kept_coefficients']:,}
    Compression Ratio: {stats['compression_ratio']:.2f}:1
    Kept Percentage: {stats['kept_percentage']:.6f}%
    
    [Quality Assessment]
    MSE:  {metrics['MSE']:.4f}
    RMSE: {metrics['RMSE']:.4f}
    SNR:  {metrics['SNR']:.2f} dB
    PSNR: {metrics['PSNR']:.2f} dB
    
    [Processing Parameters]
    Image Size: 512x512
    DCT Type: Whole Image DCT
    (Different from 8x8 block DCT)
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
    
    plt.suptitle('DCT Experiment 2 - Whole Image DCT (DC Component Only)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """主函数"""
    # 处理图像
    original, processed, stats = process_whole_image("test_image.jpg")
    
    if original is None:
        print("\n[失败] 程序执行失败，请检查图像文件路径")
        return
    
    # 计算PSNR
    psnr = calculate_psnr(original, processed)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Experiment Results - Statistics")
    print("="*60)
    
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
    
    print(f"[Compression Statistics]")
    print(f"  Total DCT Coefficients: {stats['total_coefficients']:,}")
    print(f"  Kept Coefficients: {stats['kept_coefficients']:,} (DC only)")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}:1")
    print(f"  Kept Percentage: {stats['kept_percentage']:.6f}%")
    print(f"\n[Quality Assessment - Objective Metrics]")
    print(f"  MSE  (Mean Square Error):        {metrics['MSE']:.4f}")
    print(f"  RMSE (Root Mean Square Error):   {metrics['RMSE']:.4f}")
    print(f"  SNR  (Signal-to-Noise Ratio):    {metrics['SNR']:.2f} dB")
    print(f"  PSNR (Peak Signal-to-Noise):     {metrics['PSNR']:.2f} dB")
    print(f"\n[Quality Assessment - Subjective Rating]")
    print(f"  Quality Grade: {quality_rating}")
    print(f"\n[Additional Info]")
    print(f"  DC Component Value: {stats['dc_value']:.2f}")
    print("="*60)
    
    # 保存处理后的图像
    from PIL import Image
    output_path = "whole_image_dc_only.jpg"
    Image.fromarray(processed.astype(np.uint8)).save(output_path)
    print(f"\n[保存] 处理后的图像已保存: {output_path}")
    
    # 可视化结果
    print(f"[可视化] 正在生成可视化结果...")
    fig = visualize_results(original, processed, stats, psnr)
    
    # 保存可视化结果
    viz_path = "dct2_result_visualization.png"
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
