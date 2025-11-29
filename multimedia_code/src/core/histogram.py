import numpy as np

def calculate_histogram(image):
    """
    Calculate the histogram of a grayscale image manually.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        
    Returns:
        1D numpy array of size 256 representing the histogram.
    """
    hist = np.zeros(256, dtype=int)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            hist[image[i, j]] += 1
    return hist

def calculate_cdf(hist):
    """
    Calculate the Cumulative Distribution Function (CDF) manually.
    
    Args:
        hist: 1D numpy array representing the histogram.
        
    Returns:
        1D numpy array representing the CDF.
    """
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

def equalize_hist_manual(image):
    """
    Perform global histogram equalization manually.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        
    Returns:
        2D numpy array representing the equalized image.
    """
    hist = calculate_histogram(image)
    cdf = calculate_cdf(hist)
    
    # Normalize CDF
    total_pixels = image.size
    cdf_normalized = cdf / total_pixels
    
    # Calculate mapping
    mapping = np.floor(cdf_normalized * 255).astype(np.uint8)
    
    # Apply mapping
    height, width = image.shape
    equalized_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = mapping[image[i, j]]
            
    return equalized_image

def clahe_manual(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) manually.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        clip_limit: Threshold for contrast limiting.
        grid_size: Tuple (rows, cols) defining the grid for tiling.
        
    Returns:
        2D numpy array representing the CLAHE processed image.
    """
    height, width = image.shape
    tile_h = height // grid_size[0]
    tile_w = width // grid_size[1]
    
    # Pad image to be divisible by tile size if necessary (simplified: assuming divisible for now or cropping)
    # For robustness, we'll work on the valid area
    
    clahe_image = np.zeros_like(image)
    
    # Calculate histograms for each tile
    tiles_hist = np.zeros((grid_size[0], grid_size[1], 256))
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            r_start, r_end = i * tile_h, (i + 1) * tile_h
            c_start, c_end = j * tile_w, (j + 1) * tile_w
            tile = image[r_start:r_end, c_start:c_end]
            
            # Calculate histogram
            hist = calculate_histogram(tile)
            
            # Clip histogram
            limit = clip_limit * (tile_h * tile_w) / 256
            excess = 0
            for k in range(256):
                if hist[k] > limit:
                    excess += hist[k] - limit
                    hist[k] = limit
            
            # Redistribute excess
            increment = excess / 256
            hist = hist + increment # Float addition
            
            # Calculate CDF (mapping)
            cdf = calculate_cdf(hist)
            cdf_normalized = cdf / (tile_h * tile_w)
            tiles_hist[i, j] = np.floor(cdf_normalized * 255)

    # Bilinear interpolation
    # This part is complex to implement fully manually without edge artifacts.
    # We will implement a simplified version or a standard interpolation logic.
    # For strict manual implementation, we interpolate mappings, not pixels directly usually, 
    # but applying the mapping of the nearest tiles.
    
    # Standard CLAHE interpolation logic:
    # For each pixel, find the four surrounding tile centers and interpolate their mappings.
    
    # To keep it concise and functional as a "manual implementation" demonstration without excessive complexity,
    # we can use the tile mappings directly with interpolation.
    
    # Re-implementing full bilinear interpolation for CLAHE mappings:
    for y in range(height):
        for x in range(width):
            # Find relative position
            # Center of top-left tile is at (tile_h/2, tile_w/2)
            
            # Map pixel to grid coordinates (0 to grid_size-1)
            # We treat the grid centers as the reference points
            
            gy = (y + 0.5) / tile_h - 0.5
            gx = (x + 0.5) / tile_w - 0.5
            
            r1 = int(np.floor(gy))
            c1 = int(np.floor(gx))
            r2 = r1 + 1
            c2 = c1 + 1
            
            # Clip to valid grid range
            r1 = max(0, min(r1, grid_size[0] - 1))
            r2 = max(0, min(r2, grid_size[0] - 1))
            c1 = max(0, min(c1, grid_size[1] - 1))
            c2 = max(0, min(c2, grid_size[1] - 1))
            
            # Weights
            wy2 = gy - r1
            wy1 = 1.0 - wy2
            wx2 = gx - c1
            wx1 = 1.0 - wx2
            
            # Handle boundary conditions where we might be outside the center-to-center range
            # If we are in the corner/edge regions, the weights will naturally favor the nearest tile
            # But strictly, gy/gx can be negative or > grid_size-1. 
            # The clipping above handles the index, but we need to adjust weights if we are purely in one tile's domain?
            # Actually, standard bilinear works if we clamp the indices.
            # But if we clamp r1=r2, we need to be careful.
            
            if r1 == r2: wy1, wy2 = 1, 0
            if c1 == c2: wx1, wx2 = 1, 0

            val = image[y, x]
            
            map11 = tiles_hist[r1, c1, val]
            map12 = tiles_hist[r1, c2, val]
            map21 = tiles_hist[r2, c1, val]
            map22 = tiles_hist[r2, c2, val]
            
            interpolated_val = (map11 * wy1 * wx1 +
                                map12 * wy1 * wx2 +
                                map21 * wy2 * wx1 +
                                map22 * wy2 * wx2)
                                
            clahe_image[y, x] = int(interpolated_val)
            
    return clahe_image
