"""
DCT图像压缩实验 - 综合质量评价
对实验一、实验二、实验三的重建图像进行完整质量评价
包括: MSE、RMSE、SNR、PSNR和主观评价
"""

import numpy as np
import sys
import os

# 导入质量评价模块
from quality_evaluation import (
    calculate_mse, calculate_rmse, calculate_snr, calculate_psnr,
    evaluate_image_quality, subjective_quality_rating, print_quality_report
)

# 导入各实验的处理函数
sys.path.append('1')
sys.path.append('2')
sys.path.append('3')

from dct_1 import process_image as process_exp1
from dct_2 import process_whole_image as process_exp2
from dct_3 import process_with_top_k_coefficients as process_exp3

def run_comprehensive_evaluation():
    """
    运行综合质量评价
    """
    print("="*80)
    print("DCT图像压缩实验 - 综合质量评价")
    print("="*80)
    print()
    
    # 检查测试图像
    test_image_path = "1/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"[错误] 找不到测试图像: {test_image_path}")
        return
    
    print(f"[信息] 使用测试图像: {test_image_path}\n")
    
    # 存储所有实验结果
    all_results = []
    
    # ============ 实验一: 8×8分块DCT + 保留DC分量 ============
    print("="*80)
    print("实验一: 8×8分块DCT + 保留DC分量")
    print("="*80)
    print("[运行] 执行实验一...")
    
    try:
        original1, processed1, stats1 = process_exp1(test_image_path)
        if original1 is not None:
            metrics1 = print_quality_report(original1, processed1, "实验一")
            all_results.append({
                'name': '实验一 (8×8分块DC)',
                'metrics': metrics1,
                'stats': stats1
            })
        else:
            print("[错误] 实验一执行失败")
    except Exception as e:
        print(f"[错误] 实验一执行出错: {e}")
    
    print("\n")
    
    # ============ 实验二: 整图DCT + 只保留DC分量 ============
    print("="*80)
    print("实验二: 整图DCT + 只保留DC分量")
    print("="*80)
    print("[运行] 执行实验二...")
    
    try:
        original2, processed2, stats2 = process_exp2(test_image_path)
        if original2 is not None:
            metrics2 = print_quality_report(original2, processed2, "实验二")
            all_results.append({
                'name': '实验二 (整图DC)',
                'metrics': metrics2,
                'stats': stats2
            })
        else:
            print("[错误] 实验二执行失败")
    except Exception as e:
        print(f"[错误] 实验二执行出错: {e}")
    
    print("\n")
    
    # ============ 实验三: 整图DCT + 保留前K个最大系数 ============
    print("="*80)
    print("实验三: 整图DCT + 保留前1000个最大系数")
    print("="*80)
    print("[运行] 执行实验三...")
    
    try:
        original3, processed3, stats3 = process_exp3(test_image_path, k=1000)
        if original3 is not None:
            metrics3 = print_quality_report(original3, processed3, "实验三")
            all_results.append({
                'name': '实验三 (Top-1000)',
                'metrics': metrics3,
                'stats': stats3
            })
        else:
            print("[错误] 实验三执行失败")
    except Exception as e:
        print(f"[错误] 实验三执行出错: {e}")
    
    print("\n")
    
    # ============ 综合对比 ============
    if len(all_results) > 0:
        print("="*80)
        print("三个实验的综合对比")
        print("="*80)
        print()
        
        # 打印对比表格
        print(f"{'实验':<25} {'MSE':<12} {'RMSE':<12} {'SNR (dB)':<12} {'PSNR (dB)':<12} {'主观评级':<20}")
        print("-" * 95)
        
        for result in all_results:
            name = result['name']
            m = result['metrics']
            rating = subjective_quality_rating(m)
            
            print(f"{name:<25} {m['MSE']:<12.4f} {m['RMSE']:<12.4f} "
                  f"{m['SNR']:<12.2f} {m['PSNR']:<12.2f} {rating:<20}")
        
        print("="*95)
        print()
        
        # 详细分析
        print("="*80)
        print("详细分析")
        print("="*80)
        print()
        
        for result in all_results:
            name = result['name']
            m = result['metrics']
            s = result['stats']
            
            print(f"[{name}]")
            print(f"  压缩统计:")
            print(f"    - 保留系数: {s['kept_coefficients']:,}")
            print(f"    - 压缩比: {s['compression_ratio']:.2f}:1")
            print(f"    - 保留比例: {s['kept_percentage']:.4f}%")
            print(f"  质量指标:")
            print(f"    - MSE:  {m['MSE']:.4f}")
            print(f"    - RMSE: {m['RMSE']:.4f}")
            print(f"    - SNR:  {m['SNR']:.2f} dB")
            print(f"    - PSNR: {m['PSNR']:.2f} dB")
            print(f"  主观评级: {subjective_quality_rating(m)}")
            print()
        
        # 推荐结论
        print("="*80)
        print("实验结论与建议")
        print("="*80)
        print()
        
        # 找出PSNR最高的
        best_psnr = max(all_results, key=lambda x: x['metrics']['PSNR'])
        # 找出压缩比最高的
        best_compression = max(all_results, key=lambda x: x['stats']['compression_ratio'])
        # 找出最平衡的
        
        print(f"[最佳质量] {best_psnr['name']}")
        print(f"  PSNR: {best_psnr['metrics']['PSNR']:.2f} dB")
        print(f"  压缩比: {best_psnr['stats']['compression_ratio']:.2f}:1")
        print()
        
        print(f"[最高压缩比] {best_compression['name']}")
        print(f"  压缩比: {best_compression['stats']['compression_ratio']:.2f}:1")
        print(f"  PSNR: {best_compression['metrics']['PSNR']:.2f} dB")
        print()
        
        print("[综合评价]")
        print("  实验一 (8×8分块DC):")
        print("    - 优点: 压缩比和质量平衡良好,JPEG标准基础")
        print("    - 缺点: 有明显块效应")
        print("    - 推荐: 实际应用,进一步实现JPEG编码")
        print()
        print("  实验二 (整图DC):")
        print("    - 优点: 压缩比极高")
        print("    - 缺点: 图像完全不可识别,无实用价值")
        print("    - 推荐: 理论研究,理解DC分量含义")
        print()
        print("  实验三 (Top-K系数):")
        print("    - 优点: 质量最高,无块效应,验证能量集中特性")
        print("    - 缺点: 计算量大,系数选择需排序")
        print("    - 推荐: 研究DCT能量分布,理解压缩原理")
        print()
        
        print("="*80)
        print("质量评价完成!")
        print("="*80)
    
    else:
        print("[错误] 没有实验成功执行,无法生成综合评价")

if __name__ == "__main__":
    try:
        run_comprehensive_evaluation()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断程序执行")
    except Exception as e:
        print(f"\n\n[错误] 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

