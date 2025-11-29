import sys
import os

# Add multimedia_code to path
current_dir = os.path.dirname(os.path.abspath(__file__))
multimedia_code_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'multimedia_code'))
sys.path.append(multimedia_code_path)

from src.experiments import exp1_2

if __name__ == "__main__":
    print("Running Experiment 1-2 (Wrapper)...")
    # Redirect output to multimedia_code/output/exp1_2_dct_transform
    output_dir = os.path.join(multimedia_code_path, 'output', 'exp1_2_dct_transform')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_path = os.path.join(current_dir, 'test_image.jpg')
    
    if not os.path.exists(image_path):
        # Try to find test image in common locations
        common_paths = [
            os.path.join(current_dir, '..', '..', 'multimedia_code', 'test_image.jpg'),
            os.path.join(current_dir, '..', '..', 'test_image.jpg')
        ]
        for p in common_paths:
            if os.path.exists(p):
                image_path = p
                break
    
    exp1_2.run(image_path, output_dir)
