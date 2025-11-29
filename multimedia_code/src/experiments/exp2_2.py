import os
import numpy as np
from ..core.dct import dct_2d_manual, idct_2d_manual
from ..core.metrics import calculate_psnr
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results

def run(image_path, output_dir):
    print("Running Experiment 2-2: Full Image DC Compression...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # Full Image DCT
    dct_coeffs = dct_2d_manual(img)
    
    # Keep only Global DC coefficient (0,0)
    compressed_coeffs = np.zeros_like(dct_coeffs)
    compressed_coeffs[0, 0] = dct_coeffs[0, 0]
            
    # Reconstruct
    reconstructed = idct_2d_manual(compressed_coeffs)
    
    # Calculate PSNR
    psnr = calculate_psnr(img, reconstructed)
    print(f"Exp 2-2 PSNR: {psnr:.2f} dB")
    
    # Save results
    save_image(os.path.join(output_dir, 'exp2_2_reconstructed.jpg'), reconstructed)
    
    # Plot
    plot_results(
        [img, reconstructed],
        ['Original', f'Global DC Only (PSNR: {psnr:.2f} dB)'],
        os.path.join(output_dir, 'exp2_2_comparison.png')
    )
    print("Experiment 2-2 Completed.")
