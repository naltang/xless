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
    return array_corrected.astype(np.uint16)


import numpy as np
import os
import sys
from scipy import ndimage
import argparse
import multiprocessing
import time

# Define image dimensions
IMAGE_HEIGHT = 2560
IMAGE_WIDTH = 2048

def process_single_image(args):
    """
    Process a single image file
    args: tuple of (input_folder, output_folder, filename)
    """
    input_folder, output_folder, filename, array_correction_low, array_correction_high = args

    try:
        filepath = os.path.join(input_folder, filename)
        filepath_low = filepath + ".low"
        filepath_high = filepath + ".high"
        filepath_out = os.path.join(output_folder, filename)
        filepath_low_cor = filepath_out + ".low.cor"
        filepath_high_cor = filepath_out + ".high.cor"
        filepath_ratio = filepath_out + ".ratio"

        # Read 16-bit grayscale raw data. We do not need 2d reshaping
        image_low = np.fromfile(filepath_low, dtype=np.uint16)
        image_high = np.fromfile(filepath_high, dtype=np.uint16)

        #print(f"IL={image_low.shape}, IH={image_high.shape}, CH={array_correction_high.shape}, CL={array_correction_low.shape}")
    
        image_low_corrected = correct_image(image_low, array_correction_low)
        image_high_corrected = correct_image(image_high, array_correction_high)
        image_ratio_corrected = image_low_corrected / image_high_corrected
        
        image_low_corrected.astype(np.uint16).tofile(filepath_low_cor)    # np.uint16
        image_high_corrected.astype(np.uint16).tofile(filepath_high_cor)  # np.uint16
        image_ratio_corrected.astype(np.float32).tofile(filepath_ratio)    # np.float32

        msg = f"✅ Processed: {input_folder} {filename}.low (and high) -> {output_folder}"
        print(msg)
        return msg
    except Exception as e:
        msg = f"❌ Error processing {filename}: {str(e)}"
        print(msg)
        return msg

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process raw images with median filter and cropping')
    parser.add_argument('--input_folder', required=True, help='Input folder containing .raw files')
    parser.add_argument('--output_folder', required=True, help='Output folder for processed images')
    parser.add_argument('--high', nargs="?", default="correction_high.csv", help='Correction matrix for HIGH, default=correction_high.csv')
    parser.add_argument('--low', nargs="?", default="correction_low.csv", help='Correction matrix for LOW, default=correction_low.csv')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Assign arguments to variables
    input_folder = args.input_folder
    output_folder = args.output_folder
    filename_correction_high = args.high
    filename_correction_low = args.low

    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"'{input_folder}' is not a directory.")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .low files in input folder
    list_filename = []
    for filename_in in sorted(os.listdir(input_folder)):
        if not filename_in.endswith(".raw.low"):
            continue
        
        filename = filename_in[:-len(".low")]
        list_filename.append(filename)

    if not list_filename:
        print(f"No .raw files found in {input_folder}")
        sys.exit(1)

    array_cor_high = np.loadtxt(filename_correction_high)
    array_cor_low = np.loadtxt(filename_correction_low)

    print(f"Processing {len(list_filename)} raw files...")
    
    # Prepare arguments for each process
    process_args = [(input_folder, output_folder, filename, array_cor_high, array_cor_low)
                     for filename in list_filename]
    
    # Process images in parallel
    start_time = time.time()
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_image, process_args)
    
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
