"""
DCT变换实验
功能：对灰度图像进行8×8分块DCT变换,分析DCT系数分布
特点：完全手工实现DCT变换,不使用scipy或opencv的DCT函数
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

def load_image_from_txt(txt_path):
    """
    从文本文件加载灰度图像
    
    参数:
        txt_path: 文本文件路径
    返回:
        img: numpy数组形式的灰度图像
    """
    if not os.path.exists(txt_path):
        print(f"[错误] 找不到文件: {txt_path}")
        return None
    
    try:
        img = np.loadtxt(txt_path, dtype=np.float64)
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
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"[错误] 无法读取图像 {image_path}: {e}")
        return None

def process_image_dct(image, block_size=8):
    """
    对图像进行分块DCT变换
    
    参数:
        image: 输入灰度图像
        block_size: 分块大小(默认8×8)
    返回:
        dct_image: DCT系数图像
        stats: 统计信息
    """
    h, w = image.shape
    
    # 确保图像大小是block_size的倍数
    h_blocks = h // block_size
    w_blocks = w // block_size
    h_pad = h_blocks * block_size
    w_pad = w_blocks * block_size
    
    # 裁剪或填充
    if h != h_pad or w != w_pad:
        print(f"[警告] 图像大小{h}x{w}不是{block_size}的倍数")
        print(f"   将使用{h_pad}x{w_pad}区域")
        image = image[:h_pad, :w_pad]
        h, w = h_pad, w_pad
    
    print(f"\n[处理] 开始DCT变换...")
    print(f"   分块大小: {block_size}x{block_size}")
    print(f"   总块数: {h_blocks * w_blocks} ({h_blocks}x{w_blocks})")
    
    # 存储DCT系数
    dct_image = np.zeros_like(image, dtype=np.float64)
    
    # 统计DC和AC系数
    dc_coeffs = []
    ac_energy = []
    
    total_blocks = h_blocks * w_blocks
    processed = 0
    
    # 分块处理
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 提取块
            block = image[i:i+block_size, j:j+block_size]
            
            # DCT变换
            dct_block = dct_2d(block)
            
            # 存储
            dct_image[i:i+block_size, j:j+block_size] = dct_block
            
            # 统计
            dc_coeffs.append(dct_block[0, 0])  # DC分量
            ac_energy.append(np.sum(np.abs(dct_block[1:, :])) + np.sum(np.abs(dct_block[0, 1:])))  # AC能量
            
            processed += 1
            if processed % 1000 == 0:
                progress = (processed / total_blocks) * 100
                print(f"   处理进度: {progress:.1f}%")
    
    print(f"[完成] DCT变换完成!")
    
    # 统计信息
    stats = {
        'dc_mean': np.mean(dc_coeffs),
        'dc_std': np.std(dc_coeffs),
        'dc_min': np.min(dc_coeffs),
        'dc_max': np.max(dc_coeffs),
        'ac_energy_mean': np.mean(ac_energy),
        'ac_energy_std': np.std(ac_energy),
        'total_blocks': total_blocks,
        'block_size': block_size
    }
    
    return dct_image, stats

def analyze_dct_coefficients(dct_image, block_size=8):
    """
    分析DCT系数分布
    
    参数:
        dct_image: DCT系数图像
        block_size: 分块大小
    返回:
        analysis: 分析结果字典
    """
    h, w = dct_image.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    # 统计每个位置的系数
    position_stats = np.zeros((block_size, block_size), dtype=np.float64)
    position_counts = np.zeros((block_size, block_size), dtype=np.int32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i:i+block_size, j:j+block_size]
            position_stats += np.abs(block)
            position_counts += 1
    
    # 平均每个位置的系数幅值
    avg_coeffs = position_stats / position_counts
    
    # 能量集中分析
    total_energy = np.sum(position_stats)
    cumulative_energy = []
    positions = []
    
    # 按Zig-Zag顺序统计能量
    for u in range(block_size):
        for v in range(block_size):
            positions.append((u, v))
    
    # 按能量排序
    sorted_positions = sorted(positions, key=lambda p: avg_coeffs[p[0], p[1]], reverse=True)
    
    cumsum = 0
    for pos in sorted_positions:
        cumsum += avg_coeffs[pos[0], pos[1]]
        cumulative_energy.append((pos, cumsum / total_energy * 100))
    
    analysis = {
        'avg_coeffs': avg_coeffs,
        'total_energy': total_energy,
        'cumulative_energy': cumulative_energy,
        'sorted_positions': sorted_positions
    }
    
    return analysis

def visualize_dct_results(original, dct_image, stats, analysis, block_size=8):
    """
    可视化DCT变换结果
    
    参数:
        original: 原始图像
        dct_image: DCT系数图像
        stats: 统计信息
        analysis: 分析结果
        block_size: 分块大小
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 原始图像
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # DCT系数图像(对数尺度显示)
    ax2 = plt.subplot(2, 3, 2)
    dct_display = np.log(np.abs(dct_image) + 1)  # 对数显示
    plt.imshow(dct_display, cmap='jet')
    plt.title('DCT Coefficients (Log Scale)', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 统计信息
    ax3 = plt.subplot(2, 3, 3)
    info_text = f"""
    [DCT Statistics]
    Total Blocks: {stats['total_blocks']}
    Block Size: {stats['block_size']}x{stats['block_size']}
    
    [DC Component]
    Mean: {stats['dc_mean']:.2f}
    Std Dev: {stats['dc_std']:.2f}
    Range: [{stats['dc_min']:.2f}, {stats['dc_max']:.2f}]
    
    [AC Energy]
    Mean: {stats['ac_energy_mean']:.2f}
    Std Dev: {stats['ac_energy_std']:.2f}
    
    [Energy Concentration]
    Top 10 coeffs: {analysis['cumulative_energy'][9][1]:.1f}%
    Top 20 coeffs: {analysis['cumulative_energy'][19][1]:.1f}%
    """
    ax3.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.5))
    ax3.axis('off')
    
    # 平均DCT系数分布(8x8热图)
    ax4 = plt.subplot(2, 3, 4)
    avg_coeffs_log = np.log(analysis['avg_coeffs'] + 1)
    im = plt.imshow(avg_coeffs_log, cmap='hot', interpolation='nearest')
    plt.title('Average DCT Coefficient Distribution\n(8x8 block, log scale)', fontsize=11)
    
    # 标注DC位置
    plt.text(0, 0, 'DC', ha='center', va='center', color='white', fontweight='bold')
    
    # 添加网格
    for i in range(block_size + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5)
        plt.axvline(i - 0.5, color='gray', linewidth=0.5)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax4.set_xticks(range(block_size))
    ax4.set_yticks(range(block_size))
    
    # 能量累积曲线
    ax5 = plt.subplot(2, 3, 5)
    energy_values = [e[1] for e in analysis['cumulative_energy']]
    plt.plot(range(1, len(energy_values)+1), energy_values, 'b-', linewidth=2)
    plt.axhline(90, color='r', linestyle='--', alpha=0.5, label='90% Energy')
    plt.axhline(95, color='g', linestyle='--', alpha=0.5, label='95% Energy')
    plt.axhline(99, color='orange', linestyle='--', alpha=0.5, label='99% Energy')
    
    # 找到90%能量需要的系数数
    for i, (pos, energy) in enumerate(analysis['cumulative_energy']):
        if energy >= 90:
            plt.axvline(i+1, color='r', linestyle=':', alpha=0.3)
            plt.text(i+1, 50, f'{i+1} coeffs', rotation=90, va='center')
            break
    
    plt.title('Energy Concentration Curve', fontsize=11)
    plt.xlabel('Number of Coefficients (sorted by magnitude)')
    plt.ylabel('Cumulative Energy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 64])
    plt.ylim([0, 100])
    
    # 一个8x8块的示例
    ax6 = plt.subplot(2, 3, 6)
    # 取中间的一个8x8块
    h, w = original.shape
    mid_i = (h // block_size // 2) * block_size
    mid_j = (w // block_size // 2) * block_size
    sample_block = original[mid_i:mid_i+block_size, mid_j:mid_j+block_size]
    sample_dct = dct_image[mid_i:mid_i+block_size, mid_j:mid_j+block_size]
    
    # 显示样例块的DCT系数
    im = plt.imshow(sample_dct, cmap='RdBu_r', interpolation='nearest')
    plt.title(f'Sample 8x8 Block DCT Coefficients\n(DC={sample_dct[0,0]:.1f})', fontsize=11)
    
    # 添加数值标注(只标注前几个重要系数)
    for i in range(min(3, block_size)):
        for j in range(min(3, block_size)):
            plt.text(j, i, f'{sample_dct[i,j]:.0f}', 
                    ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax6.set_xticks(range(block_size))
    ax6.set_yticks(range(block_size))
    
    plt.suptitle(f'DCT Transform Analysis - {block_size}x{block_size} Block DCT', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def main():
    """主函数"""
    print("="*60)
    print("DCT变换实验")
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
    print(f"   数值范围: {np.min(image):.2f} - {np.max(image):.2f}")
    
    # 进行DCT变换
    block_size = 8
    dct_image, stats = process_image_dct(image, block_size=block_size)
    
    # 分析DCT系数
    print(f"\n[分析] 分析DCT系数分布...")
    analysis = analyze_dct_coefficients(dct_image, block_size=block_size)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("DCT变换结果统计")
    print("="*60)
    print(f"[处理信息]")
    print(f"   总块数: {stats['total_blocks']}")
    print(f"   分块大小: {stats['block_size']}x{stats['block_size']}")
    print(f"\n[DC分量统计]")
    print(f"   平均值: {stats['dc_mean']:.2f}")
    print(f"   标准差: {stats['dc_std']:.2f}")
    print(f"   范围: [{stats['dc_min']:.2f}, {stats['dc_max']:.2f}]")
    print(f"\n[AC能量统计]")
    print(f"   平均AC能量: {stats['ac_energy_mean']:.2f}")
    print(f"   AC能量标准差: {stats['ac_energy_std']:.2f}")
    print(f"\n[能量集中特性]")
    print(f"   前10个系数占总能量: {analysis['cumulative_energy'][9][1]:.2f}%")
    print(f"   前20个系数占总能量: {analysis['cumulative_energy'][19][1]:.2f}%")
    print(f"   前30个系数占总能量: {analysis['cumulative_energy'][29][1]:.2f}%")
    print("="*60)
    
    # 保存DCT系数到txt
    output_txt = "dct_result.txt"
    np.savetxt(output_txt, dct_image, fmt='%.6f')
    print(f"\n[保存] DCT系数已保存: {output_txt}")
    
    # 可视化结果
    print(f"[可视化] 正在生成可视化结果...")
    fig = visualize_dct_results(image, dct_image, stats, analysis, block_size)
    
    # 保存可视化结果
    viz_path = "dct_transform_result.png"
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

