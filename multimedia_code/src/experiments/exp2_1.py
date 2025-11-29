import os
import numpy as np
from ..core.dct import block_dct, block_idct
from ..core.metrics import calculate_psnr
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results

def run(image_path, output_dir):
    print("Running Experiment 2-1: Block DC Compression...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # 8x8 Block DCT
    dct_coeffs = block_dct(img, block_size=8)
    
    # Keep only DC coefficient (0,0) in each 8x8 block
    h, w = dct_coeffs.shape
    compressed_coeffs = np.zeros_like(dct_coeffs)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            compressed_coeffs[i, j] = dct_coeffs[i, j]
            
    # Reconstruct
    reconstructed = block_idct(compressed_coeffs, block_size=8)
    
    # Calculate PSNR
    psnr = calculate_psnr(img, reconstructed)
    print(f"Exp 2-1 PSNR: {psnr:.2f} dB")
    
    # Save results
    save_image(os.path.join(output_dir, 'exp2_1_reconstructed.jpg'), reconstructed)
    
    # Plot
    plot_results(
        [img, reconstructed],
        ['Original', f'Block DC Only (PSNR: {psnr:.2f} dB)'],
        os.path.join(output_dir, 'exp2_1_comparison.png')
    )
    print("Experiment 2-1 Completed.")
