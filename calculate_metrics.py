import numpy as np
import skimage
import skimage.metrics
import os

from imageio.v3 import imread

import pandas as pd

from matplotlib import pyplot as plt

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import argparse

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def norm_minmse(y, x):
    x = np.squeeze(x)
    y = np.squeeze(y)
    y = normalize(y,0.1,99.9)
    x = x- x.mean()
    y = y-y.mean()
    scale=np.cov(x.flatten(), y.flatten())[0,1]/np.var(x.flatten())
    x = scale*x
    return y,x

class MetricCalculator:
    def __init__(self, images_path, results_path, methods):
        self.images_path = images_path
        self.results_path = results_path
        self.methods = methods

    def process_name(self, dataset, name):
        gt_path = os.path.join(self.images_path, dataset, 'gt', name)
        gt = imread(gt_path).astype('float32')

        results = []

        for method in self.methods:
            if method == 'raw':
                pred_path = os.path.join(self.images_path,dataset,method,name)    
            else:
                pred_path = os.path.join(self.results_path,dataset,method,name)
        
            pred = imread(pred_path).astype('float32')

            norm_gt, norm_pred = norm_minmse(gt, pred)
            psnr = skimage.metrics.peak_signal_noise_ratio(norm_gt, norm_pred, data_range = 1)
            ssim = skimage.metrics.structural_similarity(norm_gt, norm_pred, data_range = 1)

            results.append({
                'dataset':dataset,
                'name':name,
                'method':method,
                'psnr':psnr,
                'ssim':ssim
            })
        
        return results

    def process_dataset(self, dataset):
        print(dataset)

        names = [os.path.basename(n) for n in sorted(os.listdir(os.path.join(self.images_path,dataset,'gt')))]
        names = names[len(names)*9//10:]

        results = thread_map(
            lambda name:self.process_name(dataset,name),
            names,
            desc=dataset
        )
        results = sum(results,[])
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='crops', help='Path to image crops directory')
    parser.add_argument('--results', type=str, default='results', help='Path to results directory')
    parser.add_argument('--methods', type=str, nargs='+', default=['raw','sspg','sn2n','noise2fast','care','restormer'], help='List of methods to evaluate')
    args = parser.parse_args()

    datasets = sorted(list(map(os.path.basename,os.listdir(args.images))))

    calculator = MetricCalculator(args.images, args.results, args.methods)
    results = sum((calculator.process_dataset(dataset) for dataset in datasets),[])

    df = pd.DataFrame(results)
    df.to_csv('results.csv',index=False)

    psnr = df.groupby(['dataset','method'])['psnr'].mean().unstack().round(2)[args.methods]
    ssim = df.groupby(['dataset','method'])['ssim'].mean().unstack().round(2)[args.methods]

    psnr.to_csv('psnr.csv')
    ssim.to_csv('ssim.csv')

    print('PSNR:')
    print(psnr)
    print('SSIM:')
    print(ssim)


