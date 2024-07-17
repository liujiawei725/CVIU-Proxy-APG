import torch
from torchvision import transforms
import PIL
import numpy as np
import os

def noise_patch_transform(patch,patch_mask):
    noise_transform =transforms.Compose([transforms.ToTensor()])
    patch=noise_transform(tensor2pil(torch.randn_like(patch)))
    return patch*patch_mask

def noise_generation(noise_shape=[3,224,224], mode='uniform', pixel_range=[0,1],std=None,mean=None):

    if mode == 'uniform':
        noise=np.random.uniform(pixel_range[0],pixel_range[1],noise_shape)
        noise = torch.tensor(noise)

    if mode == 'normal':
        if std == None and mean == None:
            raise Exception('Please specify the mean and std of noise')

        if type(std) != list or type(mean) != list or noise_shape[0] != len(std) or noise_shape[0] != len(mean):
            raise Exception('std or mean should be the same size list')
        temp=[]
        for i in range(len(std)):
            a=np.random.normal(loc=mean[i] ,scale=std[i],size=[noise_shape[-2],noise_shape[-1]])
            a=np.clip(a=a, a_min=pixel_range[0], a_max=pixel_range[1])
            temp.append(torch.tensor(a))
        noise=torch.stack(temp,dim=0)

    if mode == 'black':
        noise=torch.zeros(noise_shape)

    if mode == 'white':
        noise=torch.ones(noise_shape)

    return noise

def patch_mask_generate(path,size):
    image_pil=PIL.Image.open(path)
    image_pil=image_pil.convert('RGB')
    image_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])
    image_tensor=image_transform(image_pil).unsqueeze(0)
    image_mask=np.clip(a=image_tensor*10,a_max=1,a_min=0)
    image_mask=torch.stack((image_mask[0][0],image_mask[0][0],image_mask[0][0]),0)
    image_mask=image_mask.unsqueeze(0)
    return image_tensor,image_mask

def patch_attach(image,patch,patch_mask,loc):
    patch_mask_im=torch.zeros_like(image).squeeze()
    patch_mask_im[:,loc[0]:loc[0]+patch.shape[-2],loc[1]:loc[1]+patch.shape[-1]]=patch_mask.squeeze()
    patch_mask_im.unsqueeze(0)
    patch_mask_im_reverse=torch.abs(patch_mask_im-1)
    patch_im=torch.zeros_like(image)
    patch_im[0][:,loc[0]:loc[0]+patch.shape[-2],loc[1]:loc[1]+patch.shape[-1]]=patch
    return patch_im,patch_mask_im

def to_cuda(x, GPU_device):
    if torch.cuda.is_available():
        return x.cuda(GPU_device)
    else:
        return x.cpu()

def to_cpu(x):
    return x.cpu()

def tensor2pil(x):
    x = x.squeeze()
    transform = transforms.Compose([
                transforms.ToPILImage('RGB')
            ])
    return transform(x)

def pil2tensor(x):
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    return transform(x).unsqueeze(0)


def restore_patch_im(patch_im,loc,patch_side):
    start_0 = int(loc[0]-patch_side/2)
    end_0 = int(loc[0]+patch_side/2)
    start_1 = int(loc[1]-patch_side/2)
    end_1 = int(loc[1]+patch_side/2)
    patch = patch_im.clone()[: , :, start_0:end_0,start_1:end_1]
    return patch



def jpg2tensor(path='img'):
    img_name = os.listdir(path)[-1]
    img_pil=PIL.Image.open(path+'/'+img_name)
    val_data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ])
    img=val_data_transform(img_pil)
    return img.unsqueeze(0)

def img_load(patch_path,side):
    patch_pil=PIL.Image.open(patch_path)
    patch_transform = transforms.Compose([
                                         transforms.
                transforms.Resize(side),
                transforms.CenterCrop(side),
                transforms.ToTensor()
                ])
    patch=patch_transform(patch_pil)
    return patch.unsqueeze(0)

def path_generate(path):
    for i in os.listdir(path):
        if i.split('.')[-1]=='jpg' or i.split('.')[-1]=='JEPG' or i.split('.')[-1]=='png':
            return(path+'/'+i)
        else:
            continue



def generate_path_im_and_mask(patch, patch_mask, image_size, center_loc):
    patch_im = torch.zeros(image_size).squeeze()
    patch_im_mask = torch.zeros(image_size).squeeze()

    if patch.shape[-1] % 2 == 0:
        for i in range(2):
            globals()['start_%s' % i] = int(
                center_loc[i] - (patch.shape[-2:][i]) / 2)
            globals()['end_%s' % i] = int(
                center_loc[i] + (patch.shape[-2:][i]) / 2)
    else:
        for i in range(2):
            globals()['start_%s' % i] = center_loc[i] - int(
                (patch.shape[-2:][i]) / 2)
            globals()['end_%s' % i] = center_loc[i] + int(
                (patch.shape[-2:][i]) / 2)+1

    if (start_0 < 0 or start_1 < 0) or (
            end_0 > image_size[-2:][0] or end_1 > image_size[-2:][1]):
        raise Exception('Your patch is beyond the image boundary')

    patch_im_mask[:, start_0: end_0, start_1: end_1] = patch_mask
    patch_im[:, start_0: end_0, start_1: end_1] = patch * patch_mask

    if patch.device.type == 'cuda':
        GPU_device = patch.device.index
        patch_im = to_cuda(patch_im, GPU_device)
        patch_im_mask = to_cuda(patch_im_mask, GPU_device)

    return patch_im.unsqueeze(0), patch_im_mask.unsqueeze(0)
