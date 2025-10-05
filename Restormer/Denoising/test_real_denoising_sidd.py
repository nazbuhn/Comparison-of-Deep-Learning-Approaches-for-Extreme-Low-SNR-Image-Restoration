## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import glob
import imageio
from utils import norm_minmse
import skimage
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--path',required=True,help='path to dataset')
parser.add_argument('--dataset',required=True,help='dataset (e.g. actin-20x)')
parser.add_argument('--weights', default='./pretrained_models/real_denoising.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

def load_dataset(condition):
    image_list = []
    path = os.path.join(args.path,args.dataset,condition,'*.tif')
    paths = sorted(glob.glob(path))
    paths = paths[len(paths)*9//10:]
    for im_path in paths:
        im = imageio.imread(im_path).astype('float32')/65535
        image_list.append(im)
    return image_list

test_images = load_dataset('raw')
gt_images = load_dataset('gt')
test_indices = range(len(test_images))

print('%d test images'%len(test_images))
print('%d gt images'%len(gt_images))

print('test image shape:',test_images[0].shape)

os.makedirs('results/restormer.%s'%args.dataset,exist_ok=True)
os.makedirs('results/restormer.gt',exist_ok=True)
os.makedirs('results/restormer.noisy',exist_ok=True)

def write_image(path,im):
    imout32 = im.astype('float32')
    imageio.imwrite(path,imout32)

results_path = 'results/restormer.%s.tab'%(args.dataset)
times = []
with open(results_path,'w') as f:
    f.write('Noisy PSNR\tDenoised PSNR\tNoisy SSIM\tDenoised SSIM\n')
    for im,gt,index in zip(test_images,gt_images,test_indices):
        import time
        start_time = time.time()

        with torch.no_grad():
            pred = model_restoration(torch.tensor(im[None,None,:,:]))
            pred = torch.clamp(pred,0,1).cpu().detach().squeeze().numpy()
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
         
        write_image('results/restormer.gt/%06d.tif'%(index),gt)
        write_image('results/restormer.noisy/%06d.tif'%(index),noisy)
        write_image('results/restormer.%s/%06d.tif'%(args.dataset,index),pred)

results = np.loadtxt(results_path,delimiter='\t',skiprows=1)
print('averages:')
print(np.mean(results,axis=0))

print('average time elapsed: %0.2f s'%np.mean(times))
