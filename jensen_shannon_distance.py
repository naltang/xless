import numpy as np
from scipy.spatial.distance import jensenshannon
import sys
from typing import List, Tuple

def jensen_shannon_distance(array1, array2):
    """
    Calculate Jensen-Shannon divergence between histograms of two uint16 grayscale images.
    
    Parameters:
    array1, array2 (np.uint16): uint16 grayscale images
    
    Returns:
    float: Jensen-Shannon divergence value
    """
    try:
        # Load and process first image
        pixels1 = array1.flatten().astype(float)
        hist1, _ = np.histogram(pixels1, bins=65536, range=(0, 65536))
        
        # Normalize first histogram
        hist1 = hist1.astype(float)
        if np.sum(hist1) > 0:
            hist1 = hist1 / np.sum(hist1)
        else:
            raise ValueError("No pixels found in first image")
        
        # Load and process second image
        
        pixels2 = array2.flatten().astype(float)
        hist2, _ = np.histogram(pixels2, bins=65536, range=(0, 65536))
        
        # Normalize second histogram
        hist2 = hist2.astype(float)
        if np.sum(hist2) > 0:
            hist2 = hist2 / np.sum(hist2)
        else:
            raise ValueError("No pixels found in second image")
        
        # Calculate Jensen-Shannon divergence
        jsd = jensenshannon(hist1, hist2)
        
        return jsd
    
    except Exception as e:
        print(f"Error processing images: {e}")
        raise

def float_to_uint16(arr, min_val=0.5, max_val=1.5):
    """
    Map a float array in [min_val, max_val] to uint16 in [0, 65535].

    Parameters
    ----------
    arr : np.ndarray
        Input array of floats.
    min_val : float, optional
        Minimum value expected in the input (default 0.5).
    max_val : float, optional
        Maximum value expected in the input (default 1.5).

    Returns
    -------
    uint16_arr : np.ndarray
        Output array of type np.uint16.
    """
    # Ensure we work with a numpy array
    arr = np.asarray(arr)

    # Clip to [min_val, max_val] – remove out‑of‑range values
    arr_clipped = np.clip(arr, min_val, max_val)

    # Normalise to 0…1
    norm = (arr_clipped - min_val) / (max_val - min_val)

    # Scale to the full uint16 range and round
    scaled = np.round(norm * np.iinfo(np.uint16).max)

    # Convert to uint16
    return scaled.astype(np.uint16)

def jensen_shannon_distance_of_two_images(param : Tuple[str, str]):
    array1 = np.fromfile(param[0], dtype=np.uint16)
    array2 = np.fromfile(param[1], dtype=np.uint16)
    return jensen_shannon_distance(array1, array2)

def jensen_shannon_distance_of_two_images_float(param : Tuple[str, str]):
    array1_float = np.fromfile(param[0], dtype=np.float32)
    array2_float = np.fromfile(param[1], dtype=np.float32)
    array1 = float_to_uint16(array1_float)
    array2 = float_to_uint16(array2_float)

    return jensen_shannon_distance(array1, array2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide two RAW format image filenames as arguments.")
        print("Example: jensen_shannon_distance.py ex1.raw ex2.raw")
        sys.exit(1)
    jsd = jensen_shannon_distance_of_two_images((sys.argv[1], sys.argv[2]))
    print(f"{jsd:.6f}")
