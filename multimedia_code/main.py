import os
import argparse
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments import exp1_1, exp1_2, exp2_1, exp2_2, exp2_3, exp2_3_progressive

EXPERIMENTS = {
    '1-1': exp1_1,
    '1-2': exp1_2,
    '2-1': exp2_1,
    '2-2': exp2_2,
    '2-3': exp2_3,
    '2-3-progressive': exp2_3_progressive
}

def main():
    parser = argparse.ArgumentParser(description="Multimedia Experiments Runner")
    parser.add_argument('experiment', choices=EXPERIMENTS.keys(), help="Experiment ID to run")
    parser.add_argument('--image', default='test_image.jpg', help="Input image path")
    parser.add_argument('--output', default='output', help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        # Try looking in parent directories or common locations if not found
        common_paths = [
            os.path.join('..', '实验二', '1', 'test_image.jpg'),
            os.path.join('..', 'test_image.jpg'),
            r"c:\Users\XioSh\Documents\多媒体\实验二\1\test_image.jpg"
        ]
        for p in common_paths:
            if os.path.exists(p):
                print(f"Image not found at '{args.image}', using '{p}' instead.")
                args.image = p
                break
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print(f"Running experiment: {args.experiment}")
    print(f"Input image: {args.image}")
    print(f"Output directory: {args.output}")
    
    # Run the experiment
    module = EXPERIMENTS[args.experiment]
    if hasattr(module, 'run'):
        try:
            module.run(args.image, args.output)
        except Exception as e:
            print(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: Module {args.experiment} does not have a run() function.")

if __name__ == "__main__":
    main()
