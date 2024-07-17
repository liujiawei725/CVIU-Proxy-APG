import torchvision
from utils.utils import *
import random

def patch_perspective(patch, height_scale=None, width_scale=None,startpoints=None,endpoints=None):
    patch = tensor2pil(patch)
    if width_scale is None:
        width_scale = [1, 1]
    if height_scale is None:
        height_scale = [1, 1]
    patch_side = patch.size[-1]
    delta_w_1 = (width_scale[0] - 1) * patch_side
    delta_w_2 = (width_scale[1] - 1) * patch_side
    delta_h_1 = (height_scale[0] - 1) * patch_side
    delta_h_2 = (height_scale[1] - 1) * patch_side
    if startpoints == None:
        startpoints=[[0,0],[0,patch_side],[patch_side,0],[patch_side,patch_side]]
        endpoints=[[0-(delta_w_1/2),0-(delta_h_1/2)],
                   [0-(delta_w_2/2),patch_side+(delta_h_1/2)],
                   [patch_side+(delta_w_1/2),0-(delta_h_2/2)],
                   [patch_side+(delta_w_2/2),patch_side+(delta_h_2/2)]]
    patch=torchvision.transforms.functional.perspective(img=patch,
    startpoints=startpoints,endpoints=endpoints)
    patch=pil2tensor(patch)
    return patch,startpoints,endpoints



class EoT():
    """input:patch,patch_mask,location_range,rotation_range,scale_range,
       output:patch，patch_mask,patch_im,patch_im_mask
       pipeline: scale => rotate => locate
       #patch only start from the minist size
    """
    '''
    example:
    eot = EoT(scale_range=[0.5,2],angle_range=[-180,180])
    patch_eot,_=eot.pipeline(patch,img_size=img.shape)
    patch_restore=eot.restore(patch_eot)
    '''

    def __init__(self, scale_range=None, angle_range=None, location=None):
        if scale_range is None:
            self.scale_range = [1, 1]
        else:
            self.scale_range = scale_range

        if angle_range is None:
            self.angle_range = [0, 0]
        else:
            self.angle_range = angle_range

        if location is None:
            self.location = [112, 112]
        else:
            self.location = location

    def img_select(self, dataset):
        index = torch.randint(high=len(dataset), size=[1, 1])
        return dataset[index][0].squeeze()


    def scale(self, patch, scale_range=None, scale_side=None):
        if scale_range != None:
            if scale_range[0] > scale_range[1]:
                raise Exception('Please input correct scale range')
            scale_list = torch.linspace(scale_range[0], scale_range[1],
                                        steps=int((scale_range[1] - scale_range[0]) * 10) + 1)
            scale_factor = random.choice(scale_list)
            patch = patch.squeeze()
            #######
            self.init_side = patch.shape[-1]
            ######
            scale_side = int(((scale_factor) ** 0.5) * (self.init_side))
            ###
            self.scale_side = scale_side
            ###
        else:
            patch = patch.squeeze()
            self.init_side = patch.shape[-1]
            self.scale_side = scale_side

        patch = tensor2pil(patch)
        scale_transform = transforms.Compose([
            transforms.Resize(self.scale_side),
            transforms.CenterCrop(self.scale_side),
            transforms.ToTensor()
        ])
        return scale_transform(patch)

    def rotate(self, patch, angle_range=None, angle=0):
        if angle_range != None:
            if angle_range[0] > angle_range[1]:
                raise Exception('Please input correct angle range')
            angle_list = torch.linspace(angle_range[0], angle_range[1], steps=angle_range[1] - angle_range[0] + 1)
            ###
            self.angle = random.choice(angle_list)
        else:
            self.angle = angle
        ###
        patch = patch.squeeze()
        rotated_patch = tensor2pil(patch)
        rotated_patch = transforms.functional.rotate(rotated_patch, self.angle, expand=False, center=None)
        rotated_patch = pil2tensor(rotated_patch)
        self.max_side = rotated_patch.shape[-1]
        return rotated_patch

    def locate(self, patch_size, img_size):
        img_size = img_size[-2:]
        pat_size = patch_size[-2:]
        location_0_low = int(pat_size[0] / 2) + 10
        location_0_high = img_size[0] - int(pat_size[0] / 2) - 10
        location_1_low = int(pat_size[1] / 2) + 10
        location_1_high = img_size[1] - int(pat_size[1] / 2) - 10
        location_0_list = torch.linspace(location_0_low, location_0_high,
                                         steps=location_0_high - location_0_low + 1)
        location_1_list = torch.linspace(location_1_low, location_1_high,
                                         steps=location_1_high - location_1_low + 1)
        loc_0 = random.choice(location_0_list)
        loc_1 = random.choice(location_1_list)
        return ([int(loc_0), int(loc_1)])

    def pipeline(self, patch, img_size):
        patch = self.scale(patch, self.scale_range)
        patch = self.rotate(patch, self.angle_range)
        loc = self.locate(patch, img_size=img_size)
        return patch, loc

    def restore(self, patch):
        patch = tensor2pil(patch)
        patch = transforms.functional.rotate(patch, -1*self.angle, expand=True, center=None)
        crop_transform = transforms.Compose([
            transforms.CenterCrop(self.scale_side),
        ])
        resize_transform = transforms.Compose([
            transforms.Resize(self.init_side),
            transforms.ToTensor()
        ])
        patch = resize_transform(crop_transform(patch))
        return patch



def total_variance_matrix(img_shape):
    side=img_shape[-1]
    row_matrix=torch.zeros(side-1,side)
    column_matrix=torch.zeros(side,side-1)
    for i in range(row_matrix.shape[0]):
        row_matrix[i][i]=1
        row_matrix[i][i+1]=-1
    for i in range(column_matrix.shape[1]):
        column_matrix[i][i]=1
        column_matrix[i+1][i]=-1
    return to_cuda(row_matrix),to_cuda(column_matrix)

