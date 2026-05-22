import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from nets import *
from scipy.optimize import minimize

import os
from os import listdir
from os.path import join
from imageio import imread, imwrite
import glob
from tqdm import trange

import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--dataset',required=True,help='dataset name e.g. 01')
parser.add_argument('--mode',default='uncalib',help='noise model: mse, uncalib, gaussian, poisson, poissongaussian')
parser.add_argument('--reg',type=float,default=0.1,help='regularization weight on prior std. dev.')

args = parser.parse_args()

""" Re-create the model and load the weights """

model = gaussian_blindspot_network((512, 512, 1),'uncalib')

if args.mode == 'uncalib' or args.mode == 'mse':
    weights_path = 'weights/weights.%s.%s.latest.hdf5'%(args.dataset,args.mode)
else:
    weights_path = 'weights/weights.%s.%s.%0.3f.latest.hdf5'%(args.dataset,args.mode,args.reg)

model.load_weights(weights_path)

""" Load test images """

test_images = []

def load_images(noise):
    basepath = os.path.join(args.path,args.dataset,noise)
    names = sorted(os.listdir(basepath))
    N = len(names)
    names = names[N*9//10:]
    images = []
    for name in names:
        image = imread(os.path.join(basepath,name))
        image = image.astype('float32')/255
        images.append(image)
    return names, np.stack(images,axis=0)[:,:,:,None]

raw_names, X = load_images('raw')
gt_names, Y = load_images('gt')
gt = np.squeeze(Y)*255

""" Denoise test images """
def poisson_gaussian_loss(x,y,a,b):
    var = np.maximum(1e-4,a*x+b)
    loss = (y-x)**2 / var + np.log(var)
    return np.mean(loss)
optfun = lambda p, x, y : poisson_gaussian_loss(x,y,p[0],p[1])

def denoise_uncalib(y,loc,std,a,b):
    total_var = std**2
    noise_var = np.maximum(1e-3,a*loc+b)
    noise_std = noise_var**0.5
    prior_var = np.maximum(1e-4,total_var-noise_var)
    prior_std = prior_var**0.5
    return np.squeeze(gaussian_posterior_mean(y,loc,prior_std,noise_std))

if args.mode == 'mse' or args.mode == 'uncalib':
    experiment_name = '%s.%s'%(args.dataset,args.mode)
else:
    experiment_name = '%s.%s.%0.3f'%(args.dataset,args.mode,args.reg)
os.makedirs("results/%s"%experiment_name,exist_ok=True)
results_path = 'results/%s.tab'%experiment_name
with open(results_path,'w') as f:
    f.write('inputPSNR\tdenoisedPSNR\ttime\n')
    for index,im in enumerate(X):
        start_time = time.time()

        pred = model.predict(im.reshape(1,512,512,1))
        
        if args.mode == 'uncalib':
            # select only pixels above bottom 2% and below top 3% of noisy image
            good = np.logical_and(im >= np.quantile(im,0.02), im <= np.quantile(im,0.97))[None,:,:,:]
            pseudo_clean = pred[0][good]
            noisy = im[np.squeeze(good, axis=0)]

            # estimate noise level
            res = minimize(optfun, (0.01,0), (np.squeeze(pseudo_clean),np.squeeze(noisy)), method='Nelder-Mead')
            print('bootstrap poisson-gaussian fit: a = %f, b=%f, loss=%f'%(res.x[0],res.x[1],res.fun))
            a,b = res.x
            
            # run denoising
            denoised = denoise_uncalib(im[None,:,:,:],pred[0],pred[1],a,b)
        else:
            denoised = pred[0]
         
        # scale and clip to 8-bit
        denoised = np.clip(np.squeeze(denoised*255),0,65535)

        end_time = time.time()
        
        # write out image
        imwrite(f'results/{experiment_name}/{raw_names[index]}',denoised.astype('uint16'))

        noisy = np.squeeze(im)#*255
        psnr_noisy = peak_signal_noise_ratio(gt[index], noisy, data_range = 65535)
        psnr_denoised = peak_signal_noise_ratio(gt[index], denoised, data_range = 65535)

        print(psnr_noisy,psnr_denoised,end_time-start_time)
        f.write('%.15f\t%.15f\t%.15f\n'%(psnr_noisy,psnr_denoised,end_time-start_time))

""" Print averages """
results = np.loadtxt(results_path,delimiter='\t',skiprows=1)
print('averages:')
print(np.mean(results,axis=0))

