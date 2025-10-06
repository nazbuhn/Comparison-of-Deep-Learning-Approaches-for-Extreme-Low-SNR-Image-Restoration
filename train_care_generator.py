import numpy as np

import argparse

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import csbdeep
from csbdeep.data import *
from csbdeep.models import Config, CARE
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import time

from .data_loader import load_data, augment_images
from .patch_generator import remove_empty_images, generate_patches


tf.random.set_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--dataset',required=True,help='dataset (e.g. actin-20x-noise1)')
parser.add_argument('--epoch',type=int,default=200,help='num epochs')
parser.add_argument('--patch_size',type=int,default=128,help='path size')
parser.add_argument('--batch_size',type=int,default=16,help='path size')
parser.add_argument('--steps_per_epoch',type=int,default=400,help='steps per epoch')
parser.add_argument('--n_patches_per_image',type=int,default=20,help='number of patches per image for validation data')
parser.add_argument('--workers',type=int,default=16,help='number of workers for parallel batch generation')

args = parser.parse_args()
patch_size=(args.patch_size,args.patch_size)

# Load images and apply augmentation
images_lowsnr, images_highsnr = load_data(args.path,args.dataset)
images_lowsnr, images_highsnr = remove_empty_images(images_lowsnr,images_highsnr,patch_size)
images_lowsnr_train,images_lowsnr_val,images_highsnr_train,images_highsnr_val = train_test_split(images_lowsnr,images_highsnr, test_size=0.1, random_state=1234, shuffle=True)
images_lowsnr_train = augment_images(images_lowsnr_train)
images_highsnr_train = augment_images(images_highsnr_train)

# Create patch generator for training data
gen = generate_patches(images_lowsnr_train,images_highsnr_train,patch_size,args.batch_size)

# Extract patches for validation data
raw_data = RawData.from_arrays(images_lowsnr_val,images_highsnr_val, axes = 'YX')
X_val, Y_val, XY_axes = create_patches(raw_data, patch_size=patch_size, n_patches_per_image=args.n_patches_per_image)
X_val = np.transpose(X_val,(0,2,3,1))
Y_val = np.transpose(Y_val,(0,2,3,1))

# Make CARE model
model = CARE(Config(), 'weights/care-%s'%args.dataset)

# Create callback
callbacks = []
callbacks.append(ModelCheckpoint(f'weights/care-{args.dataset}/weights_best.h5',save_best_only=True,  save_weights_only=True))

# Train model
start_time = time.time()
model.prepare_for_training()
history = model.keras_model.fit_generator(gen,
                                validation_data=(X_val,Y_val),
                                steps_per_epoch=args.steps_per_epoch,
                                epochs=args.epoch,
                                callbacks=callbacks,
                                use_multiprocessing=True,
                                workers=args.workers)
model._training_finished()
end_time = time.time()
print('time elapsed: %0.2f s'%(end_time-start_time))

# Make figure of train/val loss 
# plt.figure(figsize=(16,5))
# plot_history(history,['loss','val_loss']);
# plt.savefig('care_%s_history.png'%args.dataset)

