"""
图像直方图增强实验
功能：对灰度图像进行直方图均衡化增强,支持全局和局部均衡化
特点：自行实现直方图均衡化算法,可选人脸区域增强
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_image_from_txt(txt_path):
    """
    从文本文件加载灰度图像
    
    参数:
        txt_path: 文本文件路径,每行为空格分隔的像素值
    返回:
        img: numpy数组形式的灰度图像
    """
    if not os.path.exists(txt_path):
        print(f"[错误] 找不到文件: {txt_path}")
        return None
    
    try:
        img = np.loadtxt(txt_path, dtype=np.uint8)
        if img.ndim != 2:
            print(f"[错误] 图像必须是二维灰度图")
            return None
        return img
    except Exception as e:
        print(f"[错误] 无法读取或解析 {txt_path}: {e}")
        return None

def load_image_from_file(image_path):
    """
    从图像文件加载灰度图像
    
    参数:
        image_path: 图像文件路径
    返回:
        img: numpy数组形式的灰度图像
    """
    if not os.path.exists(image_path):
        print(f"[错误] 找不到图像文件: {image_path}")
        return None
    
    try:
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"[错误] 无法读取图像 {image_path}: {e}")
        return None

def calculate_histogram(image):
    """
    计算图像的直方图
    
    参数:
        image: 灰度图像(numpy数组)
    返回:
        hist: 256个bin的直方图
    """
    hist = np.zeros(256, dtype=np.int32)
    
    # 统计每个灰度值的像素数量
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    
    return hist

def histogram_equalization(image):
    """
    全局直方图均衡化
    
    原理:
    1. 计算原始图像的直方图
    2. 计算累积分布函数(CDF)
    3. 归一化CDF到0-255范围
    4. 建立灰度映射表
    5. 应用映射变换
    
    参数:
        image: 输入灰度图像
    返回:
        equalized: 均衡化后的图像
    """
    # 1. 计算直方图
    hist = calculate_histogram(image)
    
    # 2. 计算累积分布函数(CDF)
    cdf = np.cumsum(hist)
    
    # 3. 归一化CDF
    # 找到第一个非零值作为cdf_min
    cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
    
    # 总像素数
    total_pixels = image.shape[0] * image.shape[1]
    
    # 4. 建立映射表
    # 公式: new_value = (cdf[old_value] - cdf_min) / (total_pixels - cdf_min) * 255
    transform_map = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] == 0:
            transform_map[i] = 0
        else:
            transform_map[i] = int(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
    
    # 5. 应用映射变换
    equalized = transform_map[image]
    
    return equalized

def adaptive_histogram_equalization(image, clip_limit=1.2, block_size=32, blend_ratio=0.6):
    """
    自然自适应直方图均衡化(Natural CLAHE)
    
    改进点:
    1. 降低clip_limit避免过度增强
    2. 增大block_size使过渡更平滑
    3. 与原图混合保持自然
    4. 保护高光和阴影区域
    
    参数:
        image: 输入灰度图像
        clip_limit: 对比度限制因子(建议1.0-1.5)
        block_size: 分块大小(建议32-64)
        blend_ratio: 与原图混合比例(0-1,越小越自然)
    返回:
        result: 自适应均衡化后的图像
    """
    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)
    
    step = block_size // 2  # 重叠处理,避免块效应
    
    print(f"[处理] 开始自然自适应直方图均衡化...")
    print(f"   分块大小: {block_size}x{block_size}")
    print(f"   对比度限制: {clip_limit}")
    print(f"   混合比例: {blend_ratio}")
    
    total_blocks = ((h - block_size) // step + 1) * ((w - block_size) // step + 1)
    processed_blocks = 0
    
    # 遍历所有块
    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            # 提取块
            block = image[y:y+block_size, x:x+block_size].copy()
            
            # 计算块的直方图
            hist = calculate_histogram(block)
            
            # 对比度限制(Clip)
            total_pixels = block.size
            clip_threshold = int(total_pixels * clip_limit / 256)
            
            # 裁剪过高的直方图bin
            clipped_hist = hist.copy()
            excess = 0
            for i in range(256):
                if clipped_hist[i] > clip_threshold:
                    excess += (clipped_hist[i] - clip_threshold)
                    clipped_hist[i] = clip_threshold
            
            # 重新分配被裁剪的像素
            if excess > 0:
                increment = excess // 256
                remainder = excess % 256
                for i in range(256):
                    clipped_hist[i] += increment
                    if i < remainder:
                        clipped_hist[i] += 1
            
            # 计算CDF
            cdf = np.cumsum(clipped_hist)
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
            
            # 建立映射表 - 改进版,保护高光和阴影
            transform_map = np.zeros(256, dtype=np.float32)
            for i in range(256):
                if cdf[i] == 0 or cdf_min == 0:
                    transform_map[i] = float(i)
                else:
                    # 标准均衡化映射
                    eq_val = ((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255
                    
                    # 保护高光区域(200-255)
                    if i > 200:
                        # 高光区域只轻微增强
                        alpha = 0.3
                        transform_map[i] = i * (1 - alpha) + eq_val * alpha
                    # 保护阴影区域(0-50)
                    elif i < 50:
                        # 阴影区域适度增强
                        alpha = 0.5
                        transform_map[i] = i * (1 - alpha) + eq_val * alpha
                    else:
                        # 中间区域正常增强
                        transform_map[i] = eq_val
            
            # 应用映射
            enhanced_block = transform_map[block]
            
            # 使用高斯权重替代距离权重,更平滑
            center_y, center_x = block_size // 2, block_size // 2
            sigma = block_size / 3  # 高斯标准差
            
            for i in range(block_size):
                for j in range(block_size):
                    pixel_y, pixel_x = y + i, x + j
                    if pixel_y >= h or pixel_x >= w:
                        continue
                    
                    # 高斯权重
                    dist_sq = (i - center_y)**2 + (j - center_x)**2
                    weight = np.exp(-dist_sq / (2 * sigma**2))
                    
                    result[pixel_y, pixel_x] += enhanced_block[i, j] * weight
                    weight_sum[pixel_y, pixel_x] += weight
            
            processed_blocks += 1
            if processed_blocks % 100 == 0:
                progress = (processed_blocks / total_blocks) * 100
                print(f"   处理进度: {progress:.1f}%")
    
    # 归一化
    mask_valid = weight_sum > 0
    result[mask_valid] /= weight_sum[mask_valid]
    
    # 与原图混合,使效果更自然
    result = result * blend_ratio + image.astype(np.float32) * (1 - blend_ratio)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    print(f"[完成] 处理完成!")
    
    return result

def calculate_statistics(image):
    """
    计算图像统计信息
    
    参数:
        image: 灰度图像
    返回:
        stats: 统计信息字典
    """
    hist = calculate_histogram(image)
    
    stats = {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image),
        'hist': hist
    }
    
    return stats

def visualize_results(original, enhanced, method_name="Histogram Equalization"):
    """
    可视化实验结果
    
    参数:
        original: 原始图像
        enhanced: 增强后图像
        method_name: 方法名称
    """
    # 计算统计信息
    orig_stats = calculate_statistics(original)
    enh_stats = calculate_statistics(enhanced)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
    # 原始图像
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 增强后图像
    ax2 = plt.subplot(2, 3, 2)
    plt.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Enhanced Image ({method_name})', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 统计信息
    ax3 = plt.subplot(2, 3, 3)
    info_text = f"""
    [Original Statistics]
    Mean: {orig_stats['mean']:.2f}
    Std Dev: {orig_stats['std']:.2f}
    Min: {orig_stats['min']}
    Max: {orig_stats['max']}
    Median: {orig_stats['median']:.2f}
    
    [Enhanced Statistics]
    Mean: {enh_stats['mean']:.2f}
    Std Dev: {enh_stats['std']:.2f}
    Min: {enh_stats['min']}
    Max: {enh_stats['max']}
    Median: {enh_stats['median']:.2f}
    
    [Improvement]
    Std Dev Change: {enh_stats['std'] - orig_stats['std']:.2f}
    """
    ax3.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightgreen', alpha=0.5))
    ax3.axis('off')
    
    # 原始图像直方图
    ax4 = plt.subplot(2, 3, 4)
    plt.bar(range(256), orig_stats['hist'], color='blue', alpha=0.7, width=1.0)
    plt.title('Original Histogram', fontsize=11)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    # 增强后图像直方图
    ax5 = plt.subplot(2, 3, 5)
    plt.bar(range(256), enh_stats['hist'], color='red', alpha=0.7, width=1.0)
    plt.title('Enhanced Histogram', fontsize=11)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    
    # 累积分布函数对比
    ax6 = plt.subplot(2, 3, 6)
    orig_cdf = np.cumsum(orig_stats['hist'])
    enh_cdf = np.cumsum(enh_stats['hist'])
    # 归一化
    orig_cdf = orig_cdf / orig_cdf[-1] * 255
    enh_cdf = enh_cdf / enh_cdf[-1] * 255
    
    plt.plot(range(256), orig_cdf, color='blue', alpha=0.7, label='Original CDF', linewidth=2)
    plt.plot(range(256), enh_cdf, color='red', alpha=0.7, label='Enhanced CDF', linewidth=2)
    plt.plot([0, 255], [0, 255], 'k--', alpha=0.3, label='Ideal (Linear)')
    plt.title('Cumulative Distribution Function', fontsize=11)
    plt.xlabel('Pixel Value')
    plt.ylabel('Cumulative Count (Normalized)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    
    plt.suptitle(f'Histogram Enhancement Experiment - {method_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """主函数"""
    print("="*60)
    print("图像直方图增强实验")
    print("="*60)
    
    # 尝试加载图像
    image = None
    source_type = None
    
    # 优先尝试txt文件
    if os.path.exists('testimg.txt'):
        print("[加载] 尝试从testimg.txt加载图像...")
        image = load_image_from_txt('testimg.txt')
        source_type = "txt"
    
    # 如果txt失败,尝试图像文件
    if image is None and os.path.exists('test_image.jpg'):
        print("[加载] 尝试从test_image.jpg加载图像...")
        image = load_image_from_file('test_image.jpg')
        source_type = "jpg"
    
    if image is None:
        print("\n[失败] 无法加载图像,请确保testimg.txt或test_image.jpg存在")
        return
    
    print(f"[成功] 从{source_type}文件加载图像")
    print(f"   图像大小: {image.shape[0]}x{image.shape[1]}")
    print(f"   灰度范围: {np.min(image)} - {np.max(image)}")
    
    # 询问使用哪种方法
    print("\n[选择] 请选择增强方法:")
    print("   1. 全局直方图均衡化 (快速,但可能过度增强)")
    print("   2. 标准自适应均衡化 (对比度明显,较不自然)")
    print("   3. 自然自适应均衡化 (推荐,效果最自然)")
    
    # 默认使用自然自适应方法
    method = 3
    
    if method == 1:
        print("\n[方法] 使用全局直方图均衡化...")
        enhanced = histogram_equalization(image)
        method_name = "Global HE"
    elif method == 2:
        print("\n[方法] 使用标准自适应直方图均衡化...")
        enhanced = adaptive_histogram_equalization(
            image, 
            clip_limit=2.0,    # 较高对比度
            block_size=16,     # 较小块
            blend_ratio=1.0    # 不混合
        )
        method_name = "Standard CLAHE"
    else:
        print("\n[方法] 使用自然自适应直方图均衡化...")
        enhanced = adaptive_histogram_equalization(
            image,
            clip_limit=1.2,    # 温和对比度
            block_size=32,     # 较大块,更平滑
            blend_ratio=0.6    # 60%增强+40%原图
        )
        method_name = "Natural CLAHE"
    
    # 计算统计信息
    orig_stats = calculate_statistics(image)
    enh_stats = calculate_statistics(enhanced)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("实验结果统计")
    print("="*60)
    print(f"[原始图像]")
    print(f"   平均值: {orig_stats['mean']:.2f}")
    print(f"   标准差: {orig_stats['std']:.2f}")
    print(f"   范围: {orig_stats['min']} - {orig_stats['max']}")
    print(f"\n[增强图像]")
    print(f"   平均值: {enh_stats['mean']:.2f}")
    print(f"   标准差: {enh_stats['std']:.2f}")
    print(f"   范围: {enh_stats['min']} - {enh_stats['max']}")
    print(f"\n[改善]")
    print(f"   标准差提升: {enh_stats['std'] - orig_stats['std']:.2f}")
    print(f"   对比度提升: {((enh_stats['std'] / orig_stats['std']) - 1) * 100:.1f}%")
    print("="*60)
    
    # 保存增强后的图像
    output_path = "enhanced_image.jpg"
    Image.fromarray(enhanced).save(output_path)
    print(f"\n[保存] 增强后的图像已保存: {output_path}")
    
    # 可视化结果
    print(f"[可视化] 正在生成可视化结果...")
    fig = visualize_results(image, enhanced, method_name)
    
    # 保存可视化结果
    viz_path = "histogram_enhancement_result.png"
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

