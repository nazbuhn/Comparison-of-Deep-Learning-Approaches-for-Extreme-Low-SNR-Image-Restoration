from PIL import Image, ImageDraw
import os
import re
import numpy as np
import argparse

# Function to crop TIFF images with overlap
def crop_all_tiffs_in_directory(input_dir, output_dir, tile_size, overlap):
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through all TIFF files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.tif')):
            image_path = os.path.join(input_dir, filename)
            crop_tiff_with_overlap(image_path, output_dir, tile_size, overlap)

# Function to crop a single TIFF image with overlap
def crop_tiff_with_overlap(image_path, output_dir, tile_size, overlap):
    img = Image.open(image_path)
    _crop_single_image(img.copy(), output_dir, image_path, tile_size, overlap)
            

# Function to crop a single image with overlap and make a visual crop map
def _crop_single_image(img, output_dir, base_path, tile_size, overlap):
    arr = np.array(img)
    arr = np.pad(arr, pad_width=overlap, mode='reflect')
    img = Image.fromarray(arr)
    width, height = img.size
    step = tile_size - 2* overlap

    base_name = os.path.splitext(os.path.basename(base_path))[0]
    match = re.search(r"(\d+)", base_name)
    image_number = match.group(1) if match else base_name

    # Create a drawing canvas for crop visualization
    vis_img = img.convert("RGB")
    draw = ImageDraw.Draw(vis_img)

    for top in range(0, height -   overlap, step):
        for left in range(0, width - overlap, step):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)

            if bottom - top < tile_size:
                top = height - tile_size
            if right - left < tile_size:
                left = width - tile_size

            # Save cropped tile
            tile = img.crop((left, top, left + tile_size, top + tile_size))
            tile = tile.convert("I;16") if img.mode == "I;16" else tile.convert(img.mode)
            tile_name = f"{image_number}_{left}_{top}.tif"
            tile.save(os.path.join(output_dir, tile_name), format="TIFF")

            # Draw red rectangle on visualization image
            draw.rectangle([left, top, left + tile_size, top + tile_size], outline="red", width=2)

    # Save the crop map image
    vis_path = os.path.join(output_dir, f"{image_number}_crop_map.png")
    vis_img.save(vis_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_dir", type=str, help='path to tile dir')
    parser.add_argument("output_dir", type=str)
    parser.add_argument("tile_size", type=int)
    parser.add_argument("overlap", type=int)
    args = parser.parse_args()

    crop_all_tiffs_in_directory(
        input_dir=args.tile_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
