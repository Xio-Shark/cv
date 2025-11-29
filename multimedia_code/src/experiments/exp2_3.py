"""
Experiment 2-3: Full Image DCT with Top-K Coefficient Retention.

This module implements the compression of an image by retaining only the 
top K coefficients with the largest magnitude in the frequency domain.
"""

import os
import numpy as np
from ..core.dct import dct_2d_manual, idct_2d_manual
from ..core.metrics import calculate_psnr
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results

def run(image_path, output_dir):
    """
    Execute the experiment.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Path to the output directory.
    """
    print("Running Experiment 2-3: Top-K Coefficients Compression...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # Full Image DCT
    dct_coeffs = dct_2d_manual(img)
    
    # Flatten and sort coefficients by magnitude
    flat_coeffs = dct_coeffs.flatten()
    abs_coeffs = np.abs(flat_coeffs)
    
    # Get indices of top K coefficients
    K = 1000
    indices = np.argsort(abs_coeffs)[::-1][:K]
    
    # Create mask
    mask = np.zeros_like(flat_coeffs)
    mask[indices] = 1
    
    # Apply mask
    compressed_flat = flat_coeffs * mask
    compressed_coeffs = compressed_flat.reshape(dct_coeffs.shape)
            
    # Reconstruct
    reconstructed = idct_2d_manual(compressed_coeffs)
    
    # Calculate PSNR
    psnr = calculate_psnr(img, reconstructed)
    print(f"Exp 2-3 PSNR: {psnr:.2f} dB")
    
    # Save results
    save_image(os.path.join(output_dir, 'exp2_3_reconstructed.jpg'), reconstructed)
    
    # Plot
    plot_results(
        [img, reconstructed],
        ['Original', f'Top-{K} Coeffs (PSNR: {psnr:.2f} dB)'],
        os.path.join(output_dir, 'exp2_3_comparison.png')
    )
    print("Experiment 2-3 Completed.")
