import torchvision
import pickle
from PIL import Image
import torch
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data/cifar10_4000_1111.p')

args = parser.parse_args()


pil2tensor = torchvision.transforms.ToTensor()

dataset = pickle.load(open(args.dataset, 'rb'))


img_dir = 'data/images'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(img_dir + '/train'):
    os.makedirs(img_dir + '/train')

if not os.path.exists(img_dir + '/test'):
    os.makedirs(img_dir + '/test')

def save_images(dataset, save_dir):
    for ix in tqdm(range(len(dataset))):
        pt_img = pil2tensor(Image.fromarray(dataset[ix]))
        torchvision.utils.save_image(pt_img, '%s/img_%05d.png' % (save_dir, ix))

save_images(dataset['train_data'], img_dir + '/train')
save_images(dataset['test_data'], img_dir + '/test')

torch.save(dataset['train_labels'], 'data/train_labels.pt')
torch.save(dataset['test_labels'], 'data/test_labels.pt')


