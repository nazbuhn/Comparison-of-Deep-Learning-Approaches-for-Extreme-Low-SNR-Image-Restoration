import argparse
import time
import os
import glob
import imageio
## to keep track of the images the images will be put into gt and raw folders with the title #_#_#.tif the first # respresents the image number
# the second # represents the images X position according to the fullscale image and the third # represents the y position. 

#take images and set output folder 
parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to directory with patches')
parser.add_argument('--dataset',required=True,help='dataset (e.g. actin-20x-noise1)')
parser.add_argument('--out',required=True,help='dataset (e.g. actin-20x-noise1)')
args = parser.parse_args()

os.makedirs(f'{args.out}/{args.dataset}/raw', exist_ok=True)
os.makedirs(f'{args.out}/{args.dataset}/gt', exist_ok=True)

img_paths = []
original_images_path = os.path.join(args.path,"%s*.tif"%args.dataset)
original_images_path = os.path.join(args.path, args.dataset, 'raw/*.tif')
paths = sorted(glob.glob(original_images_path))
print("Found image files:", paths)

for path in paths:
    img_paths.append({'raw': path, 'gt': path.replace('raw', 'gt')})

print("Image paths:", img_paths)

def crop_img(img_path, img_num, condition):
    img = imageio.imread(img_path)
    print(img.shape)
    X_len, Y_len = img.shape
    print(X_len)
    print(Y_len)
    if X_len != Y_len:
        #raise ('dimensions not the same', img.shape)
        raise ValueError(f'dimensions not the same: {img.shape}')

    if X_len <512:
        raise ValueError(f'image dimensions are less than 512x512: {img.shape}')
    if Y_len <512:
        raise ValueError(f'image dimensions are less than 512x512: {img.shape}')

    X_dim = X_len // 512
    print('Number of X crops', X_dim)
    Y_dim = Y_len //512
    print('Number of Y crops', Y_dim)
    num_squares = X_dim * Y_dim
    print('total number of squares', num_squares)

    crop_Y_num = 0
    crop_X_num = 0
    X_dims = []
    Y_dims = []

    for i in range(X_dim+1):
        X_dims.append(i *512)
    for i in range(Y_dim+1):
        Y_dims.append(i *512)

    for i in range(len(X_dims)):
        cropped_img = None
        if i +1 < len(X_dims):
            x_start = X_dims[i]
            x_stop = X_dims[i+1]
            print('xx', x_start, x_stop)
            for n in range(len(Y_dims)):
                if n +1 < len(Y_dims):
                    y_start = Y_dims[n]
                    y_stop = Y_dims[n+1]
                    print('yy', y_start, y_stop)
                    print('img.shape', img.shape)
                    cropped_img = img[y_start:y_stop, x_start:x_stop]
                    output_dir = '{}{}/{}/{}_{}_{}.tif'.format(args.out, args.dataset, condition, img_num, i, n)
                    print(output_dir)
                    imageio.imwrite(output_dir, cropped_img)


def crop(img_paths):
    raw_paths = []
    gt_paths = []
    for img_path in img_paths:
        raw_paths.append(img_path['raw'])
        gt_paths.append(img_path['gt'])
    num_iters = 0
    for raw_path in raw_paths:
        crop_img(raw_path, num_iters, 'raw')
        num_iters+=1
    count = 0
    for gt_path in gt_paths:
        crop_img(gt_path, count, 'gt')
        count+=1

crop(img_paths)

# save_file = os.path.join(args.out,'%s.npz'%args.dataset)