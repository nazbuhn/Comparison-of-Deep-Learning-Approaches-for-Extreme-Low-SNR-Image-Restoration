import numpy as np
import skimage
import skimage.metrics
import os

import imageio
import glob

import csbdeep 
from csbdeep.models import *
from csbdeep.utils import *

from .utils import norm_minmse

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset')
parser.add_argument('--dataset',required=True,help='dataset (e.g. actin-20x)')
parser.add_argument('--ntiles',type=int,default=1,help='number of tiles (increase if getting out-of-memory errors)')

args = parser.parse_args()

model = CARE(config=None,name="weights/care-%s"%args.dataset)

model.load_weights("weights_best.h5")

test_images = []
gt_images = []

def load_dataset(condition):
    name_list = []
    image_list = []
    path = os.path.join(args.path,args.dataset,condition,'*.tif')
    paths = sorted(glob.glob(path))
    paths = paths[len(paths)*9//10:]
    for im_path in paths:
        name_list.append(os.path.basename(im_path))
        im = imageio.imread(im_path).astype('float32')
        image_list.append(im)
    return name_list, image_list

test_names, test_images = load_dataset('raw')
gt_names, gt_images = load_dataset('gt')
test_indices = range(len(test_images))

print('%d test images'%len(test_images))
print('%d gt images'%len(gt_images))

print('test image shape:',test_images[0].shape)

os.makedirs('results/care.%s'%args.dataset,exist_ok=True)

def write_image(path,im):
    imout32 = im.astype('float32')
    imageio.imwrite(path,imout32)

results_path = 'results/care.%s.tab'%(args.dataset)
times = []
with open(results_path,'w') as f:
    f.write('Noisy PSNR\tDenoised PSNR\tNoisy SSIM\tDenoised SSIM\n')
    for name,im,gt,index in zip(test_names,test_images,gt_images,test_indices):
        import time
        start_time = time.time()
        pred = model.predict(im,axes="YX",n_tiles=(args.ntiles,args.ntiles))
        times.append(time.time()-start_time)
        
        noisy = im

        norm_high, norm_low = norm_minmse(gt, noisy)
        psnr_noisy = skimage.metrics.peak_signal_noise_ratio(norm_high, norm_low, data_range = 1)
        ssim_noisy = skimage.metrics.structural_similarity(norm_high, norm_low, data_range = 1)
        norm_high, norm_restored = norm_minmse(gt, pred)
        psnr_restored = skimage.metrics.peak_signal_noise_ratio(norm_high, norm_restored, data_range = 1)
        ssim_restored = skimage.metrics.structural_similarity(norm_high, norm_restored, data_range = 1)
        
        print(psnr_noisy,psnr_restored,ssim_noisy,ssim_restored)
        f.write('%.15f\t%.15f\t%.15f\t%.15f\n'%(psnr_noisy,psnr_restored,ssim_noisy,ssim_restored))
         
        write_image('results/care.%s/%s'%(args.dataset,name),pred)

results = np.loadtxt(results_path,delimiter='\t',skiprows=1)
print('averages:')
print(np.mean(results,axis=0))

print('average time elapsed: %0.2f s'%np.mean(times))
