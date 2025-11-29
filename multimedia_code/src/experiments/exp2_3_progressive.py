"""
Experiment 2-3 Progressive: Progressive DCT Compression Analysis.

This module performs a progressive analysis of DCT compression by varying 
the number of retained coefficients (K) and measuring PSNR.
"""

import os
import numpy as np
from ..core.dct import dct_2d_manual, idct_2d_manual
from ..core.metrics import calculate_psnr
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results, plot_line_chart

def run(image_path, output_dir):
    """
    Execute the progressive experiment.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Path to the output directory.
    """
    print("Running Experiment 2-3: Progressive Compression...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # Full Image DCT
    print("Calculating full image DCT (this may take a while)...")
    dct_coeffs = dct_2d_manual(img)
    
    # Flatten and sort coefficients by magnitude
    flat_coeffs = dct_coeffs.flatten()
    abs_coeffs = np.abs(flat_coeffs)
    # Get indices of sorted coefficients (descending)
    sorted_indices = np.argsort(abs_coeffs)[::-1]
    
    # Updated K values with more breakpoints between 100 and 600
    k_values = [10, 50, 100, 200, 300, 400, 500, 600, 1000, 2000, 5000, 10000]
    results = []
    titles = []
    psnr_values = []
    
    # Print table header
    print(f"{'K Value':<10} | {'PSNR (dB)':<10} | {'Compression Ratio':<20}")
    print("-" * 45)

    for k in k_values:
        # Select top K
        # Create a mask for the top K coefficients
        indices = sorted_indices[:k]
        mask = np.zeros_like(flat_coeffs)
        mask[indices] = 1
        
        # Apply mask
        compressed_flat = flat_coeffs * mask
        compressed_coeffs = compressed_flat.reshape(dct_coeffs.shape)
        
        # Reconstruct
        reconstructed = idct_2d_manual(compressed_coeffs)
        
        # Metrics
        psnr = calculate_psnr(img, reconstructed)
        psnr_values.append(psnr)
        ratio = (img.shape[0] * img.shape[1]) / k
        
        # Print row
        print(f"{k:<10} | {psnr:<10.2f} | {ratio:<20.1f}")

        # Save individual image - DISABLED as requested
        # save_image(os.path.join(output_dir, f'progressive_k{k:05d}.jpg'), reconstructed)
        
        results.append(reconstructed)
        titles.append(f"K={k}\nPSNR={psnr:.2f}dB")
        
    # Plot comparison
    print("Generating comparison plot...")
    plot_results(results, titles, os.path.join(output_dir, 'progressive_comparison.png'))
    
    # Plot PSNR curve
    print("Generating PSNR curve...")
    plot_line_chart(k_values, psnr_values, 
                   "PSNR vs Number of Coefficients (K)", 
                   "Number of Coefficients (K)", 
                   "PSNR (dB)", 
                   os.path.join(output_dir, 'psnr_curve.png'),
                   log_x=True)
                   
    print("Experiment 2-3 Progressive Completed.")
