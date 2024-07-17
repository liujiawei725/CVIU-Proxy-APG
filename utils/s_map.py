import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *

def boost_contrast(lower,saliency_map):
    contrast_saliency_map=np.clip(a=saliency_map,a_min=0,a_max=lower)
    filter_matrix=(contrast_saliency_map==lower)
    contrast_saliency_map=contrast_saliency_map*filter_matrix
    contrast_saliency_map=torch.sign(contrast_saliency_map)
    return contrast_saliency_map

def heat_map_plot(saliency_map, title=None):
    x = [i for i in range(saliency_map.shape[0])]
    y = [i for i in range(saliency_map.shape[1])]
    X, Y = np.meshgrid(x, y)  # 网格的创建，这个是关键
    v_map = saliency_map.numpy()
    Z = v_map[np.rot90(X), np.rot90(Y)]

    plt.figure('Imshow', facecolor='lightgray')
    if title==None:
        plt.title('Saliency Map', fontsize=18)
    else:
        plt.title(title, fontsize=18)
    plt.grid(linestyle=":")
    plt.imshow(Z, cmap='jet', origin='lower')
    # 颜色条
    plt.colorbar()
    plt.show()

def I_s_map(img,model,target,k=30):
    start_point = torch.zeros_like(img)
    saliency_sum=0
    for i in range(k):
        adve=start_point+((i+1)/k)*(img-start_point)
        adve.requires_grad_(True)
        output=model(adve)
        loss=output[0][target]
        loss.backward()
        grad=adve.grad.clone()
        saliency_sum+=grad
        adve.requires_grad_(False)
        adve.grad.zero_()
    saliency_map=saliency_sum*(1/k)*torch.abs(img-start_point)
    return saliency_map


def s_map(img, model, index):
    img = to_cuda(img)
    model = to_cuda(model)
    img.requires_grad_(True)
    output = model(img)
    loss = output[0][index]
    loss.backward()
    grad = img.grad.clone()
    _ = img.grad.zero_()
    _ = img.requires_grad_(False)
    saliency_map = torch.abs(grad)
    saliency_map=torch.max(saliency_map.squeeze(),dim=0)[0]
    return saliency_map.cpu()


class GradCAM(object):
    def __init__(self, net, layer_name):
        self.net = to_cuda(net)
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()
    
    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                
    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        inputs=to_cuda(inputs)
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature[0].cpu().data.numpy()
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return torch.tensor(cam)


def cal_patch_saliency(patch_im,patch_im_mask, target, model, background):

    if background == 'black':
        background = torch.zeros_like(patch_im_mask)#.uniform_(0.0,1.0)
    elif background == 'white':
        background = torch.ones_like(patch_im_mask)
    elif background == 'uniform':
        background = torch.ones_like(patch_im_mask).uniform_(0.0,1.0)
    elif background == 'Gaussian':
        background = torch.clamp(input = (0.25*torch.randn_like(patch_im_mask)+0.5) , min=0.0 ,max = 1.0)

    img = patch_im * patch_im_mask + background * torch.abs(patch_im_mask -1 )

    saliency_map = I_s_map(img = img, model = model, target=target, k=50)
    saliency_map = (saliency_map * patch_im_mask).squeeze()
    saliency_map_2d = torch.sum(saliency_map,dim = 0)
    patch_saliency = saliency_map_2d.sum()
    return patch_saliency.item()