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
    args: tuple of (filepath_in, filepath_out, top_crop, bottom_crop, left_crop, right_crop)
    """
    filepath_in, filepath_out, top_crop, bottom_crop, left_crop, right_crop = args
    
    try:
        
        # Read 16-bit grayscale raw data
        image_data = np.fromfile(filepath_in, dtype=np.uint16)
        
        # Reshape to image dimensions (height x width)
        image = image_data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Step 1: Apply 3x3 median filter
        filtered_image = ndimage.median_filter(image, size=3)
        
        # Step 2: Crop the image
        cropped_image = filtered_image[top_crop:IMAGE_HEIGHT-bottom_crop, 
                                     left_crop:IMAGE_WIDTH-right_crop]
        
        # Write as little-endian raw data
        cropped_image.astype(np.uint16).tofile(filepath_out)
        
        msg = f"✅ Processed: {filepath_in} -> {filepath_out}"
        print(msg)
        return msg
    except Exception as e:
        msg = f"❌ Error processing {filepath_in}: {str(e)}"
        print(msg)
        return msg

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process raw images with median filter and cropping')
    parser.add_argument('input_folder', required=True, help='Input folder containing .raw files')
    parser.add_argument('output_folder', required=True, help='Output folder for processed images')
    parser.add_argument('--top', type=int, default=20, help='Top crop amount')
    parser.add_argument('--bottom', type=int, default=20, help='Bottom crop amount')
    parser.add_argument('--left', type=int, default=20, help='Left crop amount')
    parser.add_argument('--right', type=int, default=460, help='Right crop amount')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Assign arguments to variables
    input_folder = args.input_folder
    output_folder = args.output_folder
    top_crop = args.top
    bottom_crop = args.bottom
    left_crop = args.left
    right_crop = args.right

    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"'{input_folder}' is not a directory.")

        
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .raw files in input folder
    count_image = 0
    list_filepath_in = []
    list_filepath_out = []
    for filename_in in sorted(os.listdir(input_folder)):
        if not filename_in.endswith(".raw"):
            continue

        if count_image % 2 == 0:
            imagetype = "low"
        else:
            imagetype = "high"
        
        filename_out = f"{count_image // 2}.raw.{imagetype}"
        list_filepath_out.append(os.path.join(output_folder, filename_out))
        list_filepath_in.append(os.path.join(input_folder, filename_in))
        count_image += 1

    if not list_filepath_in:
        print(f"No .raw files found in {input_folder}")
        sys.exit(1)
    
    print(f"Processing {len(list_filepath_in)} raw files...")
    
    # Prepare arguments for each process
    process_args = [(path_in, path_out, top_crop, bottom_crop, left_crop, right_crop) 
                   for path_in, path_out in zip(list_filepath_in, list_filepath_out)]
    
    # Process images in parallel
    start_time = time.time()
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_image, process_args)
    
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    cropped_height = IMAGE_HEIGHT - top_crop - bottom_crop
    cropped_width = IMAGE_WIDTH - left_crop - right_crop

    print(f"Original image size(H,W)=({IMAGE_HEIGHT},{IMAGE_WIDTH})")
    print(f"Cropped  image size(H,W)=({cropped_height},{cropped_width})")

if __name__ == "__main__":
    main()
