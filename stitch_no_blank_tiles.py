import os
import re
import numpy as np
from PIL import Image
from glob import glob
from collections import defaultdict
from scipy.optimize import least_squares
import argparse
from imageio.v3 import imread, imwrite
import matplotlib.pyplot as plt

#function to load image
def load_tiles(tile_dir):
    #initialize dictionary for images for each tile
    tiles_by_image = defaultdict(list)


    for filepath in sorted(glob(os.path.join(tile_dir, "*.tif"))):

        filename = os.path.basename(filepath) # name of tile
        img = Image.open(filepath).convert("F")
        base = filename.replace(".tif", "")
        # split name on _
        parts = base.split("_")
        # obtain parts of image name
        image_number, left, top = map(int, parts)
        # add tiles to dict by image
        tiles_by_image[int(image_number)].append({
            "left": left,
            "top": top,
            "image": np.array(img),
            "path": filepath
        })  


    return tiles_by_image

#stitch tiles 
def stitch_simple(tiles, tile_size=640):
    """Stitch tiles without blending. Last tile added overwrites any overlapping tiles."""
    #get all left coordinates
    all_lefts = [t["left"] for t in tiles]
    #get all top corrdinates
    all_tops = [t["top"] for t in tiles]
    #find furthest right tile edge
    max_w = max(all_lefts) + tile_size
    #find bottom tile edge
    max_h = max(all_tops) + tile_size
    #create blank canvas from dimensions
    canvas = np.zeros((max_h, max_w), dtype=np.float32)

    for tile in tiles:
        img = tile["image"]
        top, left = tile["top"], tile["left"]
        bottom, right = top + tile_size, left + tile_size
        canvas[top:bottom, left:right] = img  # last tile overwrites if overlap
    return canvas

# find neighboring tiles
def find_all_neighbors(tiles):
    step = 512 
    coord_to_index = {(tile["left"], tile["top"]): idx for idx, tile in enumerate(tiles)} #dictionary of tiles position to index in tile list
    # directions = [(step, 0), (-step, 0), (0, step), (0, -step)] # right, left, down, up neighbors
    directions = [(step, 0), (0, step)] # right, left, down, up neighbors

    neighbors = []
    for idx, tile in enumerate(tiles):
        x, y = tile["left"], tile["top"]
        # for each direction from tile
        for dx, dy in directions:
            neighbor_coord = (x + dx, y + dy)
            # if the direction is valid ( not an edge tile or corner tile)
            if neighbor_coord in coord_to_index:
                print('neighbor found', neighbor_coord)
                #get index
                neighbor_idx = coord_to_index[neighbor_coord]
                #append index to list of neighbors
                neighbors.append((idx, neighbor_idx))
    return neighbors #return neighboring tuplezs (current tile index, neighbor tile index)

#get the overlapping region between two tiles
def get_overlap_regions(tile_i, tile_j, tile_size=640, overlap=128):
    img_i = tile_i["image"]
    img_j = tile_j["image"]
    xi, yi = tile_i["left"], tile_i["top"] # obtain top left corner of tile i
    xj, yj = tile_j["left"], tile_j["top"] # obtain top left corder of tile j
    step = 512

    if xi == xj and yj == yi + step:      # tile J under tile i 
        region_i = img_i[-overlap:, :]
        region_j = img_j[:overlap, :]
    elif yi == yj and xj == xi + step:    # tile j is to the right
        region_i = img_i[:, -overlap:]
        region_j = img_j[:, :overlap]
    else:
        return None, None
    # return the overlapping region of i and j 
    return region_i.flatten(), region_j.flatten()

def build_residual_function(tiles, neighbors, const_idx=None):
    def residuals(params): # passed to optimizer
        # list of residuals
        res = []
        # for all neighbor pairs
        for i, j in neighbors:
            a_i, b_i = params[2 * i], params[2 * i + 1] # get scale and shift for i
            if i == const_idx:
                a_i, b_i = 1, 0
            a_j, b_j = params[2 * j], params[2 * j + 1] # get scale and shift for j
            if j == const_idx:
                a_j, b_j = 1, 0
            I_i, I_j = get_overlap_regions(tiles[i], tiles[j])  # extract overlapping regions for tiles
            if I_i is not None:
                res.extend((a_i * I_i + b_i - a_j * I_j - b_j).tolist()) #apply transformation to overlapping region and subtract to get pixel-wise difference
        return np.array(res, dtype=np.float32) 
    return residuals

#function to find tile with the greatest intensity from noisy tiles
def find_highest_intensity_tile(noisy_tiles):
    """
    Helper function to find tile with the greatest intensity from noisy tiles
    """
    max_intensity = -np.inf
    max_tile = None

    for tile in noisy_tiles:
        img_array = tile["image"]  
        intensity = img_array.mean()  

        if intensity > max_intensity:
            max_intensity = intensity
            max_tile = tile

    return max_tile, max_intensity

def solve_scale_shift(tiles, neighbors, noisy_tiles):
    n = len(tiles) #number of tiles
    max_tile, max_intensity = find_highest_intensity_tile(noisy_tiles)
    print("max_tile", max_tile['path'], "max_intensity", max_intensity)
    initial_params = np.ones(2 * n, dtype=np.float32) 
    initial_params[1::2] = 0

    const_idx = next(
        idx for idx, t in enumerate(tiles)
        if os.path.basename(t["path"]) == os.path.basename(max_tile["path"])
    )
    residual_fn = build_residual_function(tiles, neighbors, const_idx) 
    result = least_squares(residual_fn, initial_params, method='trf', verbose=2)  # perform least squares
    result.x[const_idx*2] = 1
    result.x[const_idx*2+1] = 0
    return result.x.reshape(n, 2)


#create weight mask for each tile 
def create_weight_mask(h, w, edge=128):
    # array of edge values for 0-1
    y = np.linspace(0, 1, edge)
    # middle of image
    ones = np.ones(w - 2 * edge)
    # fade left right top bottom
    x_weight = np.concatenate([y, ones, y[::-1]])
    y_weight = np.concatenate([y, np.ones(h - 2 * edge), y[::-1]])
    return np.outer(y_weight, x_weight)

def stitch_with_feathering(tiles, ab_params, tile_size=640, overlap=128):
    step = 512
    # left and top position of tile
    all_lefts = [t["left"] for t in tiles]
    all_tops = [t["top"] for t in tiles]
    # total size of full image
    max_w = max(all_lefts) + tile_size
    max_h = max(all_tops) + tile_size
    # create empty canvas and weight map
    # canvas to hold stitched image
    canvas = np.zeros((max_h, max_w), dtype=np.float32)
    # weight map to hold weights for each pixel
    weight = np.zeros_like(canvas)

    for idx, tile in enumerate(tiles): # for each tile
        a, b = ab_params[idx] # obtain scale and shift
        corrected = a * tile["image"] + b # apply affine correction to title
        top, left = tile["top"], tile["left"]
        # calculate bottom and right position
        bottom, right = top + tile_size, left + tile_size
        
        # print(idx,top,left)
        weight_mask = create_weight_mask(tile_size, tile_size, edge=overlap) # calculate feathering mask
        # imwrite('weight_mask.tif', weight_mask.astype('float32')) # save weight mask for debugging
        canvas[top:bottom, left:right] += corrected * weight_mask # add weighted tile to full canvas
        weight[top:bottom, left:right] += weight_mask  # add weight to map
    # avoid division by zero
    weight[weight == 0] = 1
    # return stitched image divided by weight map to average overlapping regions
    return canvas / weight, canvas, weight


def get_black_tiles(image, tile_size=512, border=64, thresh=5):
    """
    Given the simple restitched version of a tile, evaluate 512x512 crops to find entirely black/blank tiles. 
    Input: reconstructed noisy image, tile size 512x512, border 64, and black threshold 
    Return coordinates of blank tiles as x coordinate, y coordinate or bottom right corner
    """
    #get height and width of reconstructed image
    h, w = image.shape[:2]
    black_tiles = []

    # Loop inside the padding, by tile size to create tiles
    for y in range(border, h - border, tile_size):
        for x in range(border, w - border, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]

            # Handle case where tile hits the image edge
            if tile.shape[0] == 0 or tile.shape[1] == 0:
                continue

            # Check if the tile is fully black / blank 
            if np.max(tile) <= thresh:
                br_x = min(x + tile_size, w)
                br_y = min(y + tile_size, h)
                #if the tile is fully black append tile bottom left coordinate to list
                black_tiles.append((f'tile_{br_x}_{br_y}'))
    #return list of black 512x512 tiles based on their bottom right corrdiates
    return black_tiles



def extract_tile_info(tile_list, black_tiles, black_tile_size=512, border=64):
    """
    640x640 denoised tiles by top left corner, black tiles by bottom left corner, black tile size 512x512 (unless touching corner or edge), border 64
    overlapping_tiles: list of tile names whose top-left corner (adjusted for border) is inside a black tile
    non_overlapping_tiles: list of tile filenames to keep
    """
    # convert black tile names to bounding boxes
    black_tiles_info = [] #blank/black tile coordinates top, left, bottom, right
    for name in black_tiles:
        br_x, br_y = map(int, name.replace("tile_", "").split("_"))
        tl_x = br_x - black_tile_size
        tl_y = br_y - black_tile_size
        black_tiles_info.append({'top': tl_y, 'left': tl_x, 'bottom': br_y, 'right': br_x})

    overlapping_tiles = []
    non_overlapping_tiles = []

    for tile in tile_list:
        path = tile['path']
        filename = os.path.basename(path)
        base = os.path.splitext(filename)[0]

        #get left and top coordinates of tiles
        parts = base.split('_')
        if len(parts) >= 3:
            _, left, top = parts[-3:]
            left, top = int(left), int(top)


        # Adjust top-left for padding if at edge
        adjusted_left = left + border if left == 0 else left
        adjusted_top = top + border if top == 0 else top

        # Check if the top left corner of the tiles is within a black tile
        overlaps = False
        for black in black_tiles_info:
            if black['left'] <= adjusted_left < black['right'] and black['top'] <= adjusted_top < black['bottom']:
                overlaps = True
                break

        if overlaps:
            overlapping_tiles.append(filename)
        else:
            non_overlapping_tiles.append(filename)
    #return list of overlapping_tiles (tiles that contain black/blank regions), list of non_overlapping_tiles (tiles that do not contain blank regions)
    return overlapping_tiles, non_overlapping_tiles

#main function
def reconstruct_all(tile_dir, output_dir, noisy_dir, mask_dir):
    #create output and mask directory if they dont already exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    #load denoised tiles
    tiles_by_image = load_tiles(tile_dir)
    #load noisy tiles
    noisy_tiles_by_image = load_tiles(noisy_dir)
    
    #for each images' tiles 
    for image_number, tiles in tiles_by_image.items():
        #obtain noisy tiles
        noisy_tiles = noisy_tiles_by_image[image_number]

        # simple stitch noisy tiles (no least squares)
        stitched_temp = stitch_simple(noisy_tiles)
        #get list of blank/black 512x512 tile coordinates (bottom right)
        black_tiles = get_black_tiles(stitched_temp)
        # check denoised tiles for containing blank regions. Filter out denoised tiles containing blank regions
        overlap, nonoverlap = extract_tile_info(tiles, black_tiles)
        
        if not nonoverlap:
            continue
        else:
            #filter denoised and noisy tiles to only use tiles not containing blank regions
            tiles_to_use = [t for t in tiles if os.path.basename(t["path"]) in nonoverlap]
            noisy_tiles_to_use = [t for t in noisy_tiles if os.path.basename(t["path"]) in nonoverlap]

        # create mask of valid regions to use for psnr and ssim.
        all_lefts = [t["left"] for t in tiles_to_use]
        all_tops = [t["top"] for t in tiles_to_use]
        tile_size = 640
        max_w = max(all_lefts) + tile_size
        max_h = max(all_tops) + tile_size
        mask = np.zeros((max_h, max_w), dtype=np.uint8)
        for t in tiles_to_use:
            top, left = t["top"], t["left"]
            mask[top:top+tile_size, left:left+tile_size] = 1

        mask_path = os.path.join(mask_dir, f"mask_{image_number}.tif")
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

        # for each denoised tile find its neighboring tiles
        neighbors = find_all_neighbors(tiles_to_use)
        ab_params = solve_scale_shift(tiles_to_use, neighbors, noisy_tiles_to_use)
        #stitch tiles with feathering
        stitched, canvas, weight = stitch_with_feathering(tiles_to_use, ab_params)

        # Save results
        out_path = os.path.join(output_dir, f"stitched_{image_number}.tif")
        Image.fromarray(stitched.astype(np.float32)).save(out_path)
        out_path = os.path.join(output_dir, f"stitched_{image_number}.png")
        Image.fromarray(((stitched-stitched.min())/stitched.max()*255).astype(np.uint8)).save(out_path)
        out_path = os.path.join(output_dir, f"canvas_{image_number}.tif")
        Image.fromarray(canvas.astype(np.float32)).save(out_path)
        out_path = os.path.join(output_dir, f"weight_{image_number}.tif")
        Image.fromarray(weight.astype(np.float32)).save(out_path)


#argument parser set up
parser = argparse.ArgumentParser(description='')


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("tile_dir", type=str, help = 'path to denoised tile dir')
parser.add_argument("output_dir", type=str, help = 'path to output directory for stitched denoised tiles')
parser.add_argument("noisy_dir", type=str, help = 'path to noisy tile directory')
parser.add_argument("mask_dir", type=str, help = 'path to directory to save noisy tiles to')
args = parser.parse_args()
reconstruct_all(args.tile_dir, args.output_dir, args.noisy_dir, args.mask_dir)