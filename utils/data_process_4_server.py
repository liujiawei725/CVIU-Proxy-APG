import torchvision
import torchvision.transforms as transforms
import os
import torch
import cv2

def model_select(model):

    if model == 'vgg19':
        model=torchvision.models.vgg19(pretrained=True)
        print('================= SELECTED MODEL: vgg19 =================')
    elif model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        print('================= SELECTED MODEL: vgg16 =================')
    elif model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        print('================= SELECTED MODEL: res18 =================')
    elif model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True )
        print('================= SELECTED MODEL: res34 =================')
    elif model == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
        print('================= SELECTED MODEL: GoogleNet =================')
    elif model == 'vitb16':
        model = model = torchvision.models.vit_b_16(pretrained=True)
        print('================= SELECTED MODEL: vit_b_16 =================')
    elif model == 'vitb32':
        model = model = torchvision.models.vit_b_32(pretrained=True)
        print('================= SELECTED MODEL: vit_b_32 =================')
    else:
        raise Exception('unknown model')

    return model

class dataset_load(torch.utils.data.Dataset):
  def __init__(self,img_path,transform=None):
    self.img_path = img_path
    self.transform = transform
    self.file_name_list = os.listdir(self.img_path)


  def __len__(self):
    return len(self.file_name_list )

  def __getitem__(self,idx):
    img_name=self.file_name_list[idx]
    img=cv2.imread(self.img_path+'/'+img_name)

    if self.transform:
      img=self.transform(img)
    return (img,idx)


def proxy_dataset_select(proxy_data):
    if proxy_data == 'ImageNet':
        print('================= selected proxy data type is ImageNet =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/ImageNet',
                                     transform=transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
                               )
    elif proxy_data == 'uniform':
        print('================= selected proxy data type is uniform =================')
        dataset = torch.ones(10000,3,224,224).uniform_(0.0,1.0)

    elif proxy_data == 'white':
        print('================= selected proxy data type is white =================')
        dataset = torch.ones(1000,3,224,224)

    elif proxy_data == 'black':
        print('================= selected proxy data type is black =================')
        dataset = torch.zeros(1000,3,224,224)

    elif proxy_data == 'ensemble':
        print('================= selected proxy data type is ensemble =================')
        ensemble_dataset = torch.ones(3000,3,224,224).uniform_(0.0,1.0)
        ensemble_dataset[:1000] = torch.ones(1000,3,224,224)
        ensemble_dataset[1000:2000] = torch.zeros(1000,3,224,224)
        dataset = ensemble_dataset

    elif proxy_data == 'MSCOCO':
        print('================= selected proxy data type is MSCOCO =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/mscoco/test2017',
                                     transform=transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
                                     )

    elif proxy_data == 'KITTI':
        print('================= selected proxy data type is KITTI =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/KITTI',
                               transform=transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor()])
                               )

    # CT_COVID
    elif proxy_data == 'CT_COVID':
        print('================= selected proxy data type is CT_COVID =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/CT_COVID',
                                     transform=transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
                                     )
    elif proxy_data == 'iChallenge':
        print('================= selected proxy data type is iChallenge =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/iChallenge',
                                     transform=transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
                                     )
    elif proxy_data == 'teeth':
        print('================= selected proxy data type is teeth =================')
        dataset = dataset_load(img_path='/home/liujiawei/DataSet/EoD_DataSet/teeth',
                               transform=transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor()])
                               )

    else:
        raise Exception('unknown proxy dataset')

    return dataset