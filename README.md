# Pytorch Image classification with Resnet on cifar dataset in pickle, image and hdf5 format

In this repo, you can find pytorch image classification script on cifar dataset in pickle, image and hdf5 format. 
This allows us to test pytorch training on a smaller scale dataset on different enviroment such as pickle, raw images and hdf5 format. 

## Steps to run the code:
```
#download cifar10 and only use 400 per class
python download_n_split_cifar.py --dataset cifar10 --subset 400 --seed 1111 

#training on cifar10 dataset stored as pickle file
python train_cifar.py --dataset cifar10_4000 --seed 1111 --dtype=pickle

#saving cifar subset as images
python saving_cifar_images.py --dataset data/cifar10_4000_1111.p 

#training on cifar10 dataset stored as images
python train_cifar.py --dataset cifar10_4000 --seed 1111 --dtype=image

#saving cifar subset images in hdf5
python saving_cifar10_hdf5.py

#training on cifar10 dataset stored as hdf5 file
python train_cifar.py --dataset cifar10_4000 --seed 1111 --dtype=hdf5

```

