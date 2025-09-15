import numpy as np

def estimate_gain_org(array_this, array_correction):
    eps = 1e-12
    denom = np.sum((array_this * array_correction) ** 2) + eps
    numer = np.sum(array_this * array_correction) * 50000.0
    return numer / denom

def estimate_gain(array_this, array_correction):
    eps = 1e-12
    
    # Compute element-wise products
    products = array_this * array_correction
    
    # Flatten to 1D for processing
    flat_products = products.flatten()
    
    if len(flat_products) == 0:
        return 0.0  # Return 0 for empty arrays
    
    # Get absolute values and sort them in descending order
    abs_products = np.abs(flat_products)
    sorted_indices = np.argsort(abs_products)[::-1]  # Sort in descending order
    
    # Take the first half (largest elements)
    n_elements = len(flat_products)
    n_full_intensity = max(1, n_elements // 2)  # Ensure at least 1 element
    n_too_big_outliers = n_elements // 10
    
    # Get indices of the largest half elements
    largest_indices = sorted_indices[n_too_big_outliers:n_full_intensity]
    
    # Extract the largest half elements
    filtered_products = flat_products[largest_indices]
    
    # Compute gain
    denom = np.sum(filtered_products ** 2) + eps
    numer = np.sum(filtered_products) * 50000.0
    
    return numer / denom

def correct_image(array_this, array_correction):
    gain = estimate_gain(array_this, array_correction)
    array_corrected = array_this * array_correction * gain
    return array_corrected