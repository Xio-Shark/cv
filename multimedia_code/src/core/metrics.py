import numpy as np

def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) manually.
    
    Args:
        img1: 2D numpy array.
        img2: 2D numpy array.
        
    Returns:
        float: MSE value.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    diff = img1.astype(float) - img2.astype(float)
    mse = np.mean(diff ** 2)
    return mse

def calculate_psnr(img1, img2, max_val=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) manually.
    
    Args:
        img1: 2D numpy array (original).
        img2: 2D numpy array (compressed/reconstructed).
        max_val: Maximum possible pixel value.
        
    Returns:
        float: PSNR value in dB.
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr
