import os
import sys
import torch
from PIL import Image,ImageDraw
from utils.utils import pil2tensor, tensor2pil, to_cuda
from utils.utils import generate_path_im_and_mask, restore_patch_im
from utils.eot import EoT
from utils.attacker import attacker_select
import random
import argparse

from utils.data_process_4_server import proxy_dataset_select, model_select
from utils.s_map import cal_patch_saliency
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-random-seed', default = 0, type=int)
    parser.add_argument('--GPU-device', default = 0, type=int)
    parser.add_argument('--model', default = 'vgg19', type = str)
    parser.add_argument('--attack-iters', default = 400, type=int)
    parser.add_argument('--patch-diameter', default = 50, type=int)
    parser.add_argument('--proxy-data', default = '', type = str)
    parser.add_argument('--patch-save-dir', default = '', type = str)
    parser.add_argument('--attacker_name', default='', type=str)

    return parser.parse_args()


def main():
    args = get_args()
    target_random_seed = args.target_random_seed
    model = args.model
    diameter = args.patch_diameter
    GPU_device = args.GPU_device
    proxy_data = args.proxy_data
    attack_iters = args.attack_iters
    patch_save_dir = args.patch_save_dir
    attacker_name = args.attacker_name

    attacker_name = 'EoT+'
    attack_iters = 400
    lr_d = 1/255; I_d = 2
    lr_s = 1/255; I_s = 10
    restarts = 10

    if os.path.exists(patch_save_dir) == False:
        os.makedirs(patch_save_dir)

    dataset = proxy_dataset_select(proxy_data)
    model = model_select(model)
    _ = model.eval()
    random.seed(target_random_seed)
    # randomly select 100 target classes
    target_list = random.sample(range(0, 1000), k=100)
    attack = attacker_select(attacker_name)

    for count, target in enumerate(target_list):
        print('current target: ', target,'  ',str(count+1),'/',len(target_list))
        patch = Image.new("RGB", (diameter, diameter),(0,0,0))
        draw = ImageDraw.Draw(patch)
        draw.ellipse(((0, 0), (diameter, diameter)), fill=(255,255,255), outline=None)
        patch = pil2tensor(patch)
        patch_mask = patch.clone()
        patch_batch = torch.stack([patch] * restarts ,dim = 0).squeeze()
        patch_batch = patch_batch.uniform_(0.0, 1.0)
        patch_batch = patch_batch * patch_mask
        model = to_cuda(model, GPU_device)
        patch_batch = to_cuda(patch_batch, GPU_device)
        patch_mask = to_cuda( patch_mask, GPU_device)
        eot = EoT()
        loss_list = []
        for i in range(attack_iters):
            img = eot.img_select(dataset)
            img_batch = torch.stack([img]*restarts)
            img_batch = to_cuda(img_batch, GPU_device)
            center_loc = eot.locate(patch_size=[100,100],img_size=[224,224])
            patch_im_list = []; patch_im_mask_list = []
            for i in patch_batch:
                patch_im, patch_im_mask = generate_path_im_and_mask(patch = i.unsqueeze(0), patch_mask=patch_mask,
                                          image_size=img.shape[-3:], center_loc=center_loc)
                patch_im_list.append(patch_im.squeeze())
                patch_im_mask_list.append(patch_im_mask.squeeze())
            patch_im_batch = torch.stack(patch_im_list, dim=0)
            patch_im_mask_batch = torch.stack(patch_im_mask_list, dim=0)
            patch_im_batch,patch_im_mask_batch,loss = attack(
            img = img_batch, patch_im= patch_im_batch, patch_im_mask= patch_im_mask_batch,
            model = model, I_d = I_d ,I_s = I_s, target=target,
            lr_s = lr_s, lr_d = lr_d, style_loss_weight = 5
            )
            loss_list.extend(loss)
            patch_list = []
            for patch_im in patch_im_batch:
                patch = restore_patch_im(patch_im=patch_im.unsqueeze(0),loc=center_loc,patch_side=diameter)
                patch_list.append(patch.squeeze())
            patch_batch = torch.stack(patch_list)
        patch_saliency_list = []
        for patch in patch_batch:
            patch_im, patch_im_mask = generate_path_im_and_mask(patch=patch.unsqueeze(0),patch_mask=patch_mask,
                                                                        image_size=[3,224,224],center_loc=[112,112])
            patch_saliency =cal_patch_saliency(patch_im = patch_im, patch_im_mask=patch_im_mask,
                                               model = model, target = target, background='black')
            patch_saliency_list.append(patch_saliency)
        index = np.argmax(patch_saliency_list)
        save_patch = tensor2pil(patch_batch[index].cpu())
        path = patch_save_dir+'/'+ str(target)+ '_' + proxy_data + '.png'
        save_patch.save(path)
        print('patch have been saved in: ', path)


if __name__ == "__main__":
    main()