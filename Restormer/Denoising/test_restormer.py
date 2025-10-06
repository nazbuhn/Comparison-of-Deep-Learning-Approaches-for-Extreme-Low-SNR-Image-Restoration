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
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
from csbdeep.data.generate import sample_patches_from_multiple_stacks, no_background_patches, norm_percentiles

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
def generate_patches2(X, Y, patch_size):
    patch_filter = no_background_patches()
    normalization = norm_percentiles(percentiles=(1, 99.9))
    while True:
        X_batch = []
        Y_batch = []
        # # image_idx = np.random.randint(len(X),size=batch_size)
        # for i in image_idx:
        _Y,_X = sample_patches_from_multiple_stacks((Y,X), patch_size, 1, None, None)
        _X, _Y = normalization(_X,_Y, X,Y,None,None)
        X_batch.append(_X)
        Y_batch.append(_Y)
        yield np.concatenate(X_batch)[...,None],np.concatenate(Y_batch)[...,None]

def load_dataset():
    raw_image_list = []
    gt_image_list = []
    path_gt = os.path.join(args.path,args.dataset,'gt','*.tif')
    path_lq = os.path.join(args.path,args.dataset,'raw','*.tif')
    paths_gt = sorted(glob.glob(path_gt))
    paths_lq = sorted(glob.glob(path_lq))
    pathgt = paths_gt[len(paths_gt)*9//10:]
    print('pathgt', pathgt)
    pathlq = paths_lq[len(paths_lq)*9//10:]
    print('pathlq', pathlq)
    for im_path in range(len(pathgt)):
        # print('im_path', pathgt[im_path])
        img_gt = imageio.imread(pathgt[im_path]).astype('float32')
        # print('img_gt.shape1', img_gt.shape)
        img_lq = imageio.imread(pathlq[im_path]).astype('float32')
        gen = generate_patches2(img_lq, img_gt, (512,512))
        img_lq, img_gt = next(gen)
        # print('img_gt.shape2', img_gt.shape)
        img_lq = img_lq.squeeze(axis=0)
        img_gt = img_gt.squeeze(axis=0)
        img_lq = img_lq.squeeze(axis=-1)
        img_gt = img_gt.squeeze(axis=-1)
        # print('img_gt.shape3', img_gt.shape)
        raw_image_list.append(img_lq)
        gt_image_list.append(img_gt)
    return raw_image_list, gt_image_list


test_images, gt_images = load_dataset()
test_indices = range(len(test_images))

print('%d test images'%len(test_images))
print('%d gt images'%len(gt_images))


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
            pred = pred.cpu().detach().squeeze().numpy()
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
