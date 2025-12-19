#=============================GRAYSCALE==============================
import os
import cv2
import numpy as np

def convert_to_grayscale(input_path, output_path):
    """
    Converts a source image into a standard grayscale image.
    """
    try:
        print(f"Processing '{os.path.basename(input_path)}'...")
        
        # --- Step 1: Robustly read the image directly into grayscale ---
        with open(input_path, 'rb') as f:
            nparr = np.frombuffer(f.read(), np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            print(f"  - Could not read image file. Skipping.")
            return

        # --- Step 2: Save the result as a high-quality TIFF file ---
        cv2.imwrite(output_path, img_gray)
        print(f"  - Successfully saved grayscale image to '{os.path.basename(output_path)}'")

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(input_path)}: {e}")


def process_folder(input_folder, output_folder):
    """
    Finds all supported images in a folder and converts them to grayscale.
    """
    supported_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')
    print(f"\nScanning for {', '.join(supported_extensions)} files in '{input_folder}'...")
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_grayscale.tif"
            output_path = os.path.join(output_folder, output_filename)

            convert_to_grayscale(input_path, output_path)
            processed_count += 1
    
    if processed_count == 0:
        print("No supported image files were found in the specified folder.")
    else:
        print(f"\nProcessing complete. Processed {processed_count} image(s).")

# --- Main execution block ---
if __name__ == "__main__":
    input_folder = input("Enter the full path to the folder containing your images: ")

    if os.path.isdir(input_folder):
        output_folder = os.path.join(input_folder, "grayscale_converted")
        os.makedirs(output_folder, exist_ok=True)
        print(f"Converted images will be saved in: '{output_folder}'")
        process_folder(input_folder, output_folder)
    else:
        print("Operation cancelled: The path provided is not a valid folder.")




