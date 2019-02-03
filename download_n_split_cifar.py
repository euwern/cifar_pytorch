import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import scipy.misc
import torch.utils.data as data
from torch.autograd import Variable
import pickle
from tqdm import tqdm
import argparse
import random
from random import shuffle
import subprocess as sp

#=============================================================
# This script allows you to split cifar dataset into subsets
#  So that it can be trained on Semi-Supervised environment. 
#=============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--subset', type=int, default=400, help='subset num of images used per class, note: -1 means use everything') 

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if __name__ == '__main__':
    orig_dataset = None
    if args.dataset == 'cifar10':
        test_orig_datasets = datasets.CIFAR10(root='data/cifar10', train=False, download=True)
        train_orig_datasets = datasets.CIFAR10(root='data/cifar10', train=True, download=True)
        num_class = 10
    elif args.dataset == 'cifar100':
        test_orig_datasets = datasets.CIFAR100(root='data/cifar100', train=False, download=True)
        train_orig_datasets = datasets.CIFAR100(root='data/cifar100', train=True, download=True)
        num_class = 100

    #===========================================
    # Evenly sample x number of images per class
    #===========================================

    test_data = []
    for ix in range(test_orig_datasets.test_data.shape[0]):
        test_data.append(test_orig_datasets.test_data[ix])
    test_labels = test_orig_datasets.test_labels
    unsupervised_data = None

    if args.subset == -1:
        train_data = []
        for ix in range(train_orig_datasets.train_data.shape[0]):
            train_data.append(train_orig_datasets.train_data[ix])
        train_labels = train_orig_datasets.train_labels
    else:
        train_data = []
        train_labels = []
        unsupervised_data = []
        unsupervised_labels = []
        shuffle_ixs = list(range(train_orig_datasets.train_data.shape[0]))
        shuffle(shuffle_ixs)
        supervised_index = []
        for class_ix in range(num_class):
            num_class_max = args.subset
            num_class_ix = 0
            for ix in range(train_orig_datasets.train_data.shape[0]):
                curr_ix = shuffle_ixs[ix]
                if train_orig_datasets.train_labels[curr_ix] == class_ix:
                    train_data.append(train_orig_datasets.train_data[curr_ix])
                    train_labels.append(class_ix)
                    num_class_ix = num_class_ix + 1
                    supervised_index.append(curr_ix)
                if num_class_ix >= num_class_max:
                    break
        for ix in range(train_orig_datasets.train_data.shape[0]):
            if ix not in supervised_index:
                unsupervised_data.append(train_orig_datasets.train_data[ix])
                unsupervised_labels.append(train_orig_datasets.train_labels[ix])


    dataset = {}
    dataset['test_data'] = test_data
    dataset['test_labels'] = test_labels
    dataset['train_data'] = train_data
    dataset['train_labels'] = train_labels

    if unsupervised_data is not None:
        dataset['unsupervised_data'] = unsupervised_data
        dataset['unsupervised_labels'] = unsupervised_labels

    orig_name = ''
    if args.subset == -1:
        orig_name = 'data/{}_{}'.format(args.dataset, args.seed)
    else:
        orig_name = 'data/{}_{}_{}'.format(args.dataset, args.subset * num_class, args.seed)

    print('writing data')
    print(orig_name + '.p')
    pickle.dump(dataset, open(orig_name + '.p', 'wb'))
    exit()


