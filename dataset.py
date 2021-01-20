import albumentations
import torch
import torchvision
#from torchvision.datasets.folder import accimage_loader
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as F
import os
import random
import numpy as np
from PIL import Image

# training transforms:
# random crop (to test size)
# random horiz flip
# random vertical flip
# resize to random resize <= 1/4
# resize to 1/4 scale
# add jpeg corruption
# add color distortion?
def build_training_transform(resolution=(256, 256)):
    randomcrop = torchvision.transforms.RandomCrop(resolution)
    randomverticalflip = torchvision.transforms.RandomVerticalFlip()
    randomhorizontalflip = torchvision.transforms.RandomHorizontalFlip()
    jpegcompression = albumentations.augmentations.transforms.JpegCompression(0, 100)
    colorjitter = albumentations.augmentations.transforms.ColorJitter(hue=0.1)
    totensor = torchvision.transforms.ToTensor()
    
    def _transform(image):
        hr = randomcrop(image)
        hr = randomverticalflip(hr)
        hr = randomhorizontalflip(hr)
        lrresize = random.randint(16, min(resolution))
        lr = F.resize(hr, lrresize)
        lr = F.resize(lr, (resolution[0]//4, resolution[1]//4))
        # albumentations refusing to work on PIL images is lame
        lr_np = np.array(lr)
        lr_np = jpegcompression(image=lr_np)['image']
        lr_np = colorjitter(image=lr_np)['image']
        lr = Image.fromarray(lr_np)
        return (totensor(hr), totensor(lr))

    return _transform

# validation transform
# split image into test size pieces
# resize each patch to 1/4
# lr, hr (get more than 1 patch per image)
# might need a custom collate function after this?
def build_validation_transform(resolution=(256, 256)):
    totensor = torchvision.transforms.ToTensor()

    def _transform(image):
        hchunks = int(np.ceil(image.height/resolution[0]))
        vchunks = int(np.ceil(image.width/resolution[1]))
        padded_height = hchunks*resolution[0]
        padded_width = vchunks*resolution[1]
        padding = (0,
                   0,
                   int(padded_width - image.width),
                   int(padded_height - image.height))
        image = F.pad(image, padding=padding)
        hrs = list()
        lrs = list() 
        for i in range(hchunks):
            for j in range(vchunks):
                hr = F.crop(image, 
                            i*resolution[0],
                            j*resolution[1],
                            resolution[0],
                            resolution[1])
                lr = F.resize(hr, (resolution[0]//4, resolution[1]//4))
                hrs.append(totensor(hr))
                lrs.append(totensor(lr))
        return (torch.stack(hrs), torch.stack(lrs))

    return _transform


class BWDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.images = list()
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    self.images.append(os.path.join(dirpath, filename))

    def __getitem__(self, index):
        image_path = self.images[index]
        image = pil_loader(image_path)
        # treat everything as 1920x1080 for now
        if (image.height, image.width) != (1080, 1920):
            image = F.resize(image, (1080, 1920))
        return self.transform(image) 

    def __len__(self):
        return len(self.images)


def testgeneration(path):
    dataset = BWDataset(path, build_training_transform())
    length = len(dataset)
    if not os.path.exists('test'):
        os.makedirs('test')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    for inp in trainloader:
        for idx in range(5):
            hr = inp[0][idx,:,:,:]
            lr = inp[1][idx,:,:,:]  
            hr_pil = F.to_pil_image(hr)
            lr_pil = F.to_pil_image(lr)
            hr_pil.save(f'test/{idx}_hr.png')
            lr_pil.save(f'test/{idx}_lr.png')
            print(hr.shape)
            print(lr.shape)
        break
    valdataset = BWDataset(path, build_validation_transform((512,512)))
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=2, shuffle=True)
    length = len(valdataset)
    for inp in valloader:
        inp = (torch.reshape(inp[0], (-1, 3, 512, 512)), torch.reshape(inp[1], (-1, 3, 128, 128)))
        print(inp[0].shape)
        print(inp[1].shape)
        for chunk in range(0, inp[0].shape[0]):
            hr_pil = F.to_pil_image(inp[0][chunk,:,:,:])
            lr_pil = F.to_pil_image(inp[1][chunk,:,:,:])
            hr_pil.save(f'test/val_{chunk}hr.png')
            lr_pil.save(f'test/val_{chunk}lr.png')
        print(hr.shape)
        print(lr.shape)
        break


if __name__ == '__main__':
    testgeneration('data5/train')
