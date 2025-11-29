import os
from ..core.histogram import equalize_hist_manual, clahe_manual
from ..utils.io import read_image_grayscale, save_image
from ..utils.visualization import plot_results

def run(image_path, output_dir):
    print("Running Experiment 1-1: Histogram Enhancement...")
    
    img = read_image_grayscale(image_path)
    if img is None:
        return
        
    # 1. Global HE
    he_img = equalize_hist_manual(img)
    
    # 2. CLAHE
    clahe_img = clahe_manual(img, clip_limit=2.0, grid_size=(8, 8))
    
    # Save results
    save_image(os.path.join(output_dir, 'exp1_1_he.jpg'), he_img)
    save_image(os.path.join(output_dir, 'exp1_1_clahe.jpg'), clahe_img)
    
    # Plot
    plot_results(
        [img, he_img, clahe_img],
        ['Original', 'Global HE', 'CLAHE'],
        os.path.join(output_dir, 'exp1_1_comparison.png')
    )
    print("Experiment 1-1 Completed.")
