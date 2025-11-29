import cv2
import numpy as np
from PIL import Image

def read_image_grayscale(path):
    """
    Read an image and convert to grayscale.
    
    Args:
        path: Path to the image file.
        
    Returns:
        2D numpy array representing the grayscale image.
    """
    # Using PIL for reading to be safe, then convert to numpy
    try:
        img = Image.open(path).convert('L')
        return np.array(img)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def save_image(path, image):
    """
    Save a numpy array as an image.
    
    Args:
        path: Output path.
        image: 2D numpy array.
    """
    try:
        img = Image.fromarray(image.astype(np.uint8))
        img.save(path)
    except Exception as e:
        print(f"Error saving image {path}: {e}")
