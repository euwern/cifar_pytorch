import torchvision
import pickle
from PIL import Image
import torch
import argparse
import os
from tqdm import tqdm
import h5py
import numpy as np

#please save cifar10 dataset in images before creating hdf5 file

parser = argparse.ArgumentParser()

args = parser.parse_args()

def save_images(dataset, save_name):
    for ix in tqdm(range(len(dataset))):
        img = np.asarray(Image.open('data/images/%s/img_%05d.png' % (save_name,ix)), dtype='uint8')        
        hf.create_dataset(name='%s_img_%05d' % (save_name, ix), data=img, dtype='uint8')

hf = h5py.File('data/images.hdf5', 'w')

train_labels = torch.load('data/train_labels.pt')
test_labels  = torch.load('data/test_labels.pt')
save_images(train_labels, 'train')
save_images(test_labels, 'test')
hf.close()


