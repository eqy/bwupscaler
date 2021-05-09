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
        lrresize = random.randint(8, min(resolution))
        lr = F.resize(hr, lrresize)
        lr = F.resize(lr, (resolution[0]//4, resolution[1]//4))
        # albumentations refusing to work on PIL images is lame
        lr_np = np.array(lr)
        lr_np = jpegcompression(image=lr_np, quality_lower=0, p=0.9)['image']
        #lr_np = colorjitter(image=lr_np)['image']
        lr = Image.fromarray(lr_np)
        return (totensor(hr), totensor(lr))

    return _transform

# validation transform
# split image into test size pieces
# resize each patch to 1/4
# lr, hr (get more than 1 patch per image)
# might need a custom collate function after this?
def build_validation_transform(resolution=(256, 256), scale=4):
    totensor = torchvision.transforms.ToTensor()

    def _transform(image):
        vchunks = int(np.ceil(image.height/resolution[0]))
        hchunks = int(np.ceil(image.width/resolution[1]))
        padded_height = vchunks*resolution[0]
        padded_width = hchunks*resolution[1]
        padding = (0,
                   0,
                   int(padded_width - image.width),
                   int(padded_height - image.height))
        image = F.pad(image, padding=padding)
        hrs = list()
        lrs = list() 
        for i in range(vchunks):
            for j in range(hchunks):
                hr = F.crop(image, 
                            i*resolution[0],
                            j*resolution[1],
                            resolution[0],
                            resolution[1])
                lr = F.resize(hr, (resolution[0]//scale, resolution[1]//scale))
                hrs.append(totensor(hr))
                lrs.append(totensor(lr))
        return (torch.stack(hrs), torch.stack(lrs))

    return _transform


def build_inference_transform(input_resolution=(64, 64), scale=4, overscan=8):
    assert (input_resolution[0] + overscan) % scale == 0
    assert (input_resolution[1] + overscan) % scale == 0
    totensor = torchvision.transforms.ToTensor()
    
    def _transform(image):
        orig_height = image.height
        orig_width = image.width
        vchunks = int(np.ceil(orig_height/input_resolution[0]))
        hchunks = int(np.ceil(orig_width/input_resolution[1]))
        # disgusting literal edge case
        image = F.pad(image, (0, 0, overscan, overscan))
        output_height = input_resolution[0] + 2*overscan
        output_width = input_resolution[1] + 2*overscan
        lrs = list()
        for i in range(vchunks):
            for j in range(hchunks):
                #vstart = min(0, i*input_resolution[0] - overscan)
                #hstart = min(0, j*input_resolution[1] - overscan)
                #vend = max((i+1)*input_resolution[0] + overscan, image.height)
                #hend = max((j+1)*input_resolution[1] + overscan, image.width)
                #patch_height = vend - vstart
                #patch_width = hend - hstart
                #assert patch_height <= output_height
                #assert patch_width <= output_width
                #if patch_height < output_height:
                #    if vstart > 0:
                #        vstart -= (patch_height - output_height)
                #        assert vstart >= 0
                #    else:
                #        vend += (patch_height - output_height)
                #        assert vend <= image.height
                #if patch_width < output_width:
                #    if hstart > 0:
                #        hstart -= (patch_width - output_width)
                #        assert hstart >= 0
                #    else:
                #        hend += (patch_width - output_width)
                #        assert hend <= image.width
                vstart = i*input_resolution[0] - overscan
                vend = (i+1)*input_resolution[0] + overscan
                hstart = j*input_resolution[1] - overscan
                hend = (j+1)*input_resolution[1] + overscan
                if i == 0:
                    vend += -vstart
                    vstart = 0
                elif i == vchunks - 1:
                    vstart -= (vend - orig_height)
                    vend = orig_height
                if j == 0:
                    hend += -hstart
                    hstart = 0
                elif j == hchunks - 1:
                    hstart -= (hend - orig_width)
                    hend = orig_width
                assert vstart >= 0 and hstart >= 0 and vend <= image.height and hend <= image.width
                lr = F.crop(image, vstart, hstart, output_height, output_width)
                lrs.append(totensor(lr))
        return torch.stack(lrs)
    return _transform 


def assemble_inference_image(total_input_resolution=(512, 512), input_resolution=(64, 64), scale=4, overscan=8):
    vchunks = int(np.ceil(total_input_resolution[0]/input_resolution[0]))
    hchunks = int(np.ceil(total_input_resolution[1]/input_resolution[1]))

    def assemble(batch):
        flatidx = 0
        assert batch.size(0) == vchunks*hchunks
        assembled_tensor = torch.empty((3,
                                        total_input_resolution[0]*scale,
                                        total_input_resolution[1]*scale))
        for i in range(vchunks):
            for j in range(hchunks): 
                cur_chunk = batch[flatidx,:,:,:]
                patchvstart = overscan
                patchvend = overscan + input_resolution[0]
                patchhstart = overscan
                patchhend = overscan + input_resolution[1]
                destvstart = i*input_resolution[0]*scale
                destvend = (i+1)*input_resolution[0]*scale
                desthstart = j*input_resolution[1]*scale
                desthend = (j+1)*input_resolution[1]*scale

                if i == 0:
                    patchvstart = 0
                    patchvend = input_resolution[0]
                elif i == vchunks - 1:
                    patchvend = input_resolution[0] + 2*overscan
                    patchvstart = 2*overscan
                    destvstart -= (destvend - assembled_tensor.size(1))
                    destvend = assembled_tensor.size(1)
                if j == 0:
                    patchhstart = 0
                    patchhend = input_resolution[1]
                elif j == hchunks - 1:
                    patchhend = input_resolution[1] + 2*overscan
                    patchhstart = 2*overscan
                    desthstart -= (desthend - assembled_tensor.size(2))
                    desthend = assembled_tensor.size(2)
                vstart = scale*patchvstart
                vend = scale*patchvend
                hstart = scale*patchhstart
                hend = scale*patchhend
                assembled_tensor[:,
                                 destvstart:destvend,
                                 desthstart:desthend] = cur_chunk[:,
                                                                  vstart:vend,
                                                                  hstart:hend]
                flatidx += 1
        return assembled_tensor
    return assemble

 
class BWDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, extension='.jpg', passfilename=False, resizetarget=(1080, 1920), filterfunc=None):
        self.path = path
        self.transform = transform
        self.images = list()
        self.passfilename = passfilename
        self.resizetarget = resizetarget
        self.filter = filterfunc
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                name, ext = os.path.splitext(filename)
                if ext == extension:
                    if self.filter is None or self.filter(filename):
                        self.images.append(os.path.join(dirpath, filename))

    def __getitem__(self, index):
        image_path = self.images[index]
        image = pil_loader(image_path)
        # treat everything as 1920x1080 for now
        if self.resizetarget is not None and (image.height, image.width) != self.resizetarget:
            image = F.resize(image, self.resizetarget)
        if self.passfilename:
            return (self.transform(image), image_path)
        return self.transform(image) 

    def __len__(self):
        return len(self.images)


class FusedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = sum([len(dataset) for dataset in datasets])
        
    def __getitem__(self, index):
        dataset = self.datasets[0]
        idx = index
        for dataset in self.datasets:
            if idx >= len(dataset):
                idx -= len(dataset)
            else:
                return dataset[idx] 
        raise ValueError

    def __len__(self):
        return self.len


def testgeneration(path):
    dataset = BWDataset(path, build_training_transform())
    length = len(dataset)
    if not os.path.exists('test'):
        os.makedirs('test')
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    print("testing generation...")
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
    print("testing assembly...")
    infdataset = BWDataset(path, build_inference_transform((256, 256), overscan=32))
    infdataloader = torch.utils.data.DataLoader(valdataset, batch_size=1, shuffle=False)
    assemble_transform = assemble_inference_image((270, 480))
    for idx, (inp_patches, _) in enumerate(infdataloader):
        dummy_lr_patches = torchvision.transforms.functional.resize(inp_patches, (80, 80))
        resized = torchvision.transforms.functional.resize(dummy_lr_patches, (320, 320))
        assembled = assemble_transform(resized)
        for subidx, dummy_lr_patch in enumerate(dummy_lr_patches):
            lr_dummy_pil = F.to_pil_image(dummy_lr_patch)
            lr_dummy_pil.save(f'test/inf_dummy_lr_{idx}_{subidx}.png')
        sr_dummy_pil = F.to_pil_image(assembled)
        sr_dummy_pil.save(f'test/inf_dummy_sr_{idx}.png') 
        if idx >= 4:
            break 

if __name__ == '__main__':
    testgeneration('data5/train')
