import os
from tqdm import tqdm

import tifffile
import numpy as np
from SN2N.utils import *

import argparse

def normalize(imgpath,save_path,pmin=0,pmax=99.999):
    image_data = tifffile.imread(imgpath)
    image_data = normalize_percentage(x = image_data, pmin = pmin, pmax = pmax, axis=None, clip=True, eps=1e-20, dtype=np.float32)
    image_data = 255*image_data
    tifffile.imwrite(save_path, image_data.astype('uint8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',required=True,help='path to dataset root')
    parser.add_argument('--dataset',required=True,help='dataset name e.g. 01')
    parser.add_argument('--pathout',required=True,help='output path')
    args = parser.parse_args()

    path = args.path
    dataset = args.dataset
    pathout = args.pathout
    rawpath = os.path.join(path,dataset,'raw')

    names = sorted(os.listdir(rawpath))
    N = len(names)
    train_names = names[:N*9//10]
    test_names = names[N*9//10:]

    os.makedirs(args.pathout,exist_ok=True)

    def run(names,split):
        outpath = os.path.join(pathout,dataset,split)

        os.makedirs(outpath,exist_ok=True)
        for name in tqdm(names,desc=split):
            path_in = os.path.join(path,dataset,'raw',name)
            path_out = os.path.join(outpath,name)
            normalize(path_in,path_out)

    run(train_names,'train')
    run(test_names,'test')