import os
import numpy as np
from ..core.dct import block_dct, block_idct
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results

def run(image_path, output_dir):
    print("Running Experiment 1-2: DCT Transform...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # 8x8 Block DCT
    dct_coeffs = block_dct(img, block_size=8)
    
    # Inverse DCT (should reconstruct image)
    reconstructed = block_idct(dct_coeffs, block_size=8)
    
    # Visualize DCT coefficients (log scale for visibility)
    dct_vis = np.log(np.abs(dct_coeffs) + 1)
    dct_vis = (dct_vis / dct_vis.max()) * 255
    
    # Save results
    save_image(os.path.join(output_dir, 'exp1_2_reconstructed.jpg'), reconstructed)
    save_image(os.path.join(output_dir, 'exp1_2_dct_vis.jpg'), dct_vis)
    
    # Plot
    plot_results(
        [img, dct_vis, reconstructed],
        ['Original', 'DCT Coefficients (Log)', 'Reconstructed'],
        os.path.join(output_dir, 'exp1_2_comparison.png')
    )
    print("Experiment 1-2 Completed.")
