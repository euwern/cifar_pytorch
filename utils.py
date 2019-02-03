import time
import torch
import pickle
import torchvision.transforms as t
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import h5py

class TextLogger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
        self.log(title)
    
    def log(self, strdata):
        outstr = strdata + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()

train_transform = t.Compose([
    t.Resize((256, 256)),
    t.RandomCrop(224),
    t.RandomHorizontalFlip(),
    t.ToTensor(),
    ])
test_transform = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    ])

class ImageDataset_hdf5(data.Dataset):
    def __init__(self, dataset, train, use_pretrain):

        self.train = train
        self.data = h5py.File('data/images.hdf5', 'r')
        if self.train == True:
            self.label = torch.load('data/train_labels.pt')
            self.transform = train_transform
            self.dataset_path = 'train_img_%05d'
        else:
            self.label = torch.load('data/test_labels.pt')
            self.transform = test_transform
            self.dataset_path = 'test_img_%05d'

        pil2tensor = t.ToTensor()

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]
            labels = torch.load('data/train_labels.pt')

            print('Calculating mean and std')
            for ix in tqdm(range(len(labels))):
                np_dat = self.data['train_img_%05d' % ix]
                img = pil2tensor(Image.fromarray(np_dat.value))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(labels)
                mean_std['std'][cix] /= len(labels)

            torch.save(mean_std, 'data/mean_std.pt')

        else:
            mean_std = torch.load('data/mean_std.pt')

        if use_pretrain:
            mean_std['mean']=[0.485, 0.456, 0.406]
            mean_std['std']=[0.229, 0.224, 0.225]

        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))
        self.data.close()
        self.data = None

    def __getitem__(self, index):
        
        if self.data == None:
            self.data = h5py.File('data/images.hdf5', 'r')
        img = Image.fromarray(self.data[self.dataset_path % index].value)
        target = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.label)

class ImageDataset_image(data.Dataset):
    def __init__(self, dataset, train, use_pretrain):

        self.train = train
        if self.train == True:
            self.img_path = 'data/images/train/img_%05d.png'
            self.label = torch.load('data/train_labels.pt')
            self.transform = train_transform
            dataset_path = 'data/train_labels.pt'
        else:
            self.img_path = 'data/images/test/img_%05d.png'
            self.label = torch.load('data/test_labels.pt')
            self.transform = test_transform
            dataset_path = 'data/test_labels.pt'

        pil2tensor = t.ToTensor()

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]
            labels = torch.load('data/train_labels.pt')

            print('Calculating mean and std')
            for ix in tqdm(range(len(labels))):
                img = pil2tensor((Image.open('data/images/train/img_%05d.png' % ix)))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(labels)
                mean_std['std'][cix] /= len(labels)

            torch.save(mean_std, 'data/mean_std.pt')

        else:
            mean_std = torch.load('data/mean_std.pt')

        if use_pretrain:
            mean_std['mean']=[0.485, 0.456, 0.406]
            mean_std['std']=[0.229, 0.224, 0.225]

        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))


    def __getitem__(self, index):
        img = Image.open(self.img_path % index)
        target = self.label[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.label)

class ImageDataset_pickle(data.Dataset):
    def __init__(self, dataset, train, use_pretrain):
        
        dataset = pickle.load(open('data/' + dataset + '.p', 'rb')) 
        self.train = train

        train_data = dataset['train_data']
        train_labels = dataset['train_labels']

        pil2tensor = t.ToTensor()
        if self.train == True:
            self.data = train_data
            self.label = train_labels
            self.transform = train_transform
        else:
            self.data = dataset['test_data']
            self.label = dataset['test_labels']
            self.transform = test_transform

        if not os.path.exists('data/mean_std.pt'):
            mean_std = {}
            mean_std['mean'] = [0,0,0]
            mean_std['std'] = [0,0,0]

            print('Calculating mean and std')
            for ix in tqdm(range(len(train_labels))):
                img = pil2tensor((Image.fromarray(train_data[ix])))
                for cix in range(3):
                    mean_std['mean'][cix] += img[cix,:,:].mean()
                    mean_std['std'][cix] += img[cix,:,:].std()

            for cix in range(3):
                mean_std['mean'][cix] /= len(train_labels)
                mean_std['std'][cix] /= len(train_labels)

            torch.save(mean_std, 'data/mean_std.pt')
        else:
            mean_std = torch.load('data/mean_std.pt')


        if use_pretrain:
            mean_std['mean']=[0.485, 0.456, 0.406]
            mean_std['std']=[0.229, 0.224, 0.225]

        self.transform.transforms.append(t.Normalize(mean=mean_std['mean'], std=mean_std['std']))


    def __getitem__(self, index):
        img = self.data[index]
        target = self.label[index]
        img  = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_logger = TextLogger('Train loss', 'train_loss.log')
    for ix in range(30):
        print(ix)
        train_logger.log('%s, %s' % (str(torch.rand(1)[0]), str(torch.rand(1)[0])))
        time.sleep(1)


