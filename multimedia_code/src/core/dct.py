import numpy as np

def create_dct_matrix(n):
    """
    Create the DCT-II matrix of size n x n.
    C[i, j] = alpha(i) * cos((2*j + 1) * i * pi / (2*n))
    
    Args:
        n: Size of the matrix.
        
    Returns:
        n x n numpy array.
    """
    dct_matrix = np.zeros((n, n))
    for i in range(n):
        alpha = np.sqrt(1/n) if i == 0 else np.sqrt(2/n)
        for j in range(n):
            dct_matrix[i, j] = alpha * np.cos((2 * j + 1) * i * np.pi / (2 * n))
    return dct_matrix

def dct_1d_manual(vector):
    """
    Perform 1D Discrete Cosine Transform (DCT-II) manually.
    Using matrix multiplication for efficiency.
    """
    n = len(vector)
    C = create_dct_matrix(n)
    return C @ vector

def idct_1d_manual(vector):
    """
    Perform Inverse 1D Discrete Cosine Transform (DCT-III) manually.
    Using matrix multiplication for efficiency.
    """
    n = len(vector)
    C = create_dct_matrix(n)
    return C.T @ vector

def dct_2d_manual(matrix):
    """
    Perform 2D DCT manually using matrix multiplication.
    F = C * f * C^T
    
    Args:
        matrix: 2D numpy array.
        
    Returns:
        2D numpy array representing the DCT coefficients.
    """
    h, w = matrix.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    
    return C_h @ matrix @ C_w.T

def idct_2d_manual(matrix):
    """
    Perform Inverse 2D DCT manually using matrix multiplication.
    f = C^T * F * C
    
    Args:
        matrix: 2D numpy array (DCT coefficients).
        
    Returns:
        2D numpy array representing the spatial data.
    """
    h, w = matrix.shape
    C_h = create_dct_matrix(h)
    C_w = create_dct_matrix(w)
    
    return C_h.T @ matrix @ C_w

def block_dct(image, block_size=8):
    """
    Apply DCT to the image in blocks.
    """
    h, w = image.shape
    dct_image = np.zeros((h, w), dtype=float)
    
    # Pre-compute DCT matrix for block size
    C = create_dct_matrix(block_size)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                # F = C * f * C^T
                dct_image[i:i+block_size, j:j+block_size] = C @ block @ C.T
            else:
                # Handle edge cases (fallback to slower method or pad)
                # For now, just use the generic 2D function which handles any size
                dct_image[i:i+block_size, j:j+block_size] = dct_2d_manual(block)
                
    return dct_image

def block_idct(dct_image, block_size=8):
    """
    Apply Inverse DCT to the image in blocks.
    """
    h, w = dct_image.shape
    img_reconstructed = np.zeros((h, w), dtype=float)
    
    # Pre-compute DCT matrix for block size
    C = create_dct_matrix(block_size)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                # f = C^T * F * C
                img_reconstructed[i:i+block_size, j:j+block_size] = C.T @ block @ C
            else:
                img_reconstructed[i:i+block_size, j:j+block_size] = idct_2d_manual(block)
                
    return img_reconstructed
