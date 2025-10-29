"""
图像质量评价工具
实现实验要求的所有质量评价指标: MSE、RMSE、SNR、PSNR
"""

import numpy as np

def calculate_mse(original, reconstructed):
    """
    计算均方误差 (MSE - Mean Square Error)
    
    公式: MSE = (1/MN) Σ Σ [f(i,j) - f̂(i,j)]²
    
    参数:
        original: 原始图像 (MxN)
        reconstructed: 重建图像 (MxN)
    返回:
        mse: 均方误差值
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def calculate_rmse(original, reconstructed):
    """
    计算均方根误差 (RMSE - Root Mean Square Error)
    
    公式: RMSE = √MSE
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
    返回:
        rmse: 均方根误差值
    """
    mse = calculate_mse(original, reconstructed)
    rmse = np.sqrt(mse)
    return rmse

def calculate_snr(original, reconstructed):
    """
    计算均方信噪比 (SNR - Signal-to-Noise Ratio)
    
    公式: SNR = 10 * log₁₀(Σf² / Σ(f-f̂)²)
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
    返回:
        snr: 信噪比值(dB)
    """
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    # 信号能量
    signal_power = np.sum(original ** 2)
    
    # 噪声能量
    noise_power = np.sum((original - reconstructed) ** 2)
    
    # 避免除零
    if noise_power == 0:
        return float('inf')
    
    # SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(original, reconstructed, max_pixel=255.0):
    """
    计算峰值信噪比 (PSNR - Peak Signal-to-Noise Ratio)
    
    公式: PSNR = 10 * log₁₀(MAX² / MSE)
         或: PSNR = 20 * log₁₀(MAX / √MSE)
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
        max_pixel: 最大像素值(默认255)
    返回:
        psnr: 峰值信噪比值(dB)
    """
    mse = calculate_mse(original, reconstructed)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    # 或等价地: psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def evaluate_image_quality(original, reconstructed, max_pixel=255.0):
    """
    综合评价图像质量(实验要求的所有指标)
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
        max_pixel: 最大像素值
    返回:
        metrics: 包含所有质量指标的字典
    """
    metrics = {
        'MSE': calculate_mse(original, reconstructed),
        'RMSE': calculate_rmse(original, reconstructed),
        'SNR': calculate_snr(original, reconstructed),
        'PSNR': calculate_psnr(original, reconstructed, max_pixel)
    }
    
    return metrics

def subjective_quality_rating(metrics):
    """
    基于客观指标的主观质量评级
    
    参考标准:
    - PSNR > 40 dB: 极好
    - 30-40 dB: 良好  
    - 20-30 dB: 通过/勉强
    - < 20 dB: 低劣
    
    参数:
        metrics: 质量指标字典
    返回:
        rating: 主观质量等级
    """
    psnr = metrics['PSNR']
    
    if psnr == float('inf'):
        return "完美 (Perfect) - 无失真"
    elif psnr > 40:
        return "极好 (Excellent)"
    elif psnr > 30:
        return "良好 (Good)"
    elif psnr > 25:
        return "通过 (Acceptable)"
    elif psnr > 20:
        return "勉强 (Fair)"
    else:
        return "低劣 (Poor)"

def print_quality_report(original, reconstructed, experiment_name=""):
    """
    打印完整的质量评价报告
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
        experiment_name: 实验名称
    """
    print("="*60)
    if experiment_name:
        print(f"图像质量评价报告 - {experiment_name}")
    else:
        print("图像质量评价报告")
    print("="*60)
    
    # 计算所有指标
    metrics = evaluate_image_quality(original, reconstructed)
    
    # 主观评级
    rating = subjective_quality_rating(metrics)
    
    print(f"\n[客观保真度准则]")
    print(f"  MSE  (均方误差):        {metrics['MSE']:.4f}")
    print(f"  RMSE (均方根误差):      {metrics['RMSE']:.4f}")
    print(f"  SNR  (信噪比):          {metrics['SNR']:.2f} dB")
    print(f"  PSNR (峰值信噪比):      {metrics['PSNR']:.2f} dB")
    
    print(f"\n[主观保真度准则]")
    print(f"  质量等级: {rating}")
    
    print("="*60)
    
    return metrics

# 示例使用
if __name__ == "__main__":
    # 创建测试图像
    original = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    # 模拟不同质量的重建
    print("\n测试1: 完美重建")
    reconstructed1 = original.copy()
    print_quality_report(original, reconstructed1, "完美重建")
    
    print("\n测试2: 添加少量噪声")
    noise = np.random.randn(512, 512) * 5
    reconstructed2 = np.clip(original + noise, 0, 255).astype(np.uint8)
    print_quality_report(original, reconstructed2, "轻度失真")
    
    print("\n测试3: 添加较多噪声")
    noise = np.random.randn(512, 512) * 20
    reconstructed3 = np.clip(original + noise, 0, 255).astype(np.uint8)
    print_quality_report(original, reconstructed3, "中度失真")
    
    print("\n测试4: 严重失真")
    reconstructed4 = np.full_like(original, 128)
    print_quality_report(original, reconstructed4, "严重失真")

