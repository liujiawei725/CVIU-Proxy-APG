import torch

def attacker_select(name):
    if name == 'EoT':
        return EoT_patch
    if name == 'EoT+':
        return EoD_patch




def adv_patch(img, patch_im, patch_im_mask, model,loss_fn, iteration, lr, loss_seg=1):
    patch_im_mask_reverse = torch.abs(patch_im_mask - 1)
    loss_list = []

    for i in range(iteration):
        patch_im.requires_grad_(True)
        adve = patch_im_mask_reverse * img + patch_im_mask * patch_im

        output = model(adve)
        loss = loss_fn(output)
        loss_list.append(loss.item())
        loss.backward()

        grad = patch_im.grad.clone()
        # zero the grad,
        _ = patch_im.requires_grad_(False)
        _ = patch_im.grad.zero_()

        patch_im -= lr * torch.sign(grad) * patch_im_mask
        patch_im = patch_im
        patch_im = torch.clamp(input = patch_im, min = 0, max = 1)

        if (i+1) % loss_seg == 0:
            print('iteration:%s  loss:%s' % (i + 1, loss.item()))

    return patch_im, patch_im_mask, loss_list




def EoT_patch(patch_im, patch_im_mask, img, model, I_s, lr_s, target, **kwargs):

    if I_s != 1:
        raise Warning('patch update multiple times for a single image')

    def loss_fn_sal(output):
        loss = -( torch.log( (torch.nn.functional.softmax(output, dim=1)[:,target]) ).sum() ) / (len(output))
        return loss

    def loss_fn_dis(output):
        loss =  ( torch.log( (torch.nn.functional.softmax(output, dim=1)[:,target]) ).sum() ) / (len(output))
        return loss
    EoT_loss_list = []
    tem_patch_im, tem_patch_im_mask, loss_list = adv_patch(img = img,
                                                           patch_im = patch_im,
                                                           patch_im_mask=patch_im_mask,
                                                           model=model,
                                                           iteration=I_s,
                                                           lr=lr_s,
                                                           loss_fn=loss_fn_sal,
                                                           loss_seg=100)
    EoT_loss_list.extend([-1*i for i in loss_list])
    return tem_patch_im, tem_patch_im_mask, EoT_loss_list




# 这个实际上是EoT+
def EoD_patch(patch_im, patch_im_mask, img, model, I_s, I_d, lr_s, lr_d, target, **kwargs):

    def loss_fn_sal(output):
        loss = -( torch.log( (torch.nn.functional.softmax(output, dim=1)[:,target]) ).sum() ) / (len(output))
        return loss

    def loss_fn_dis(output):
        loss =  ( torch.log( (torch.nn.functional.softmax(output, dim=1)[:,target]) ).sum() ) / (len(output))
        return loss

    EoD_loss_list = []

    img_mask = torch.abs(patch_im_mask - 1)
    if (loss_fn_dis == None) or (I_d == 0):
        raise  Exception('You have not specify the objective function for image background or the I_d = 0')
    d_im, d_im_mask, loss_list = adv_patch(img = patch_im, #the patch_im is the image now # untarget_
                                           patch_im = img,  #img is the patch you want to update
                                           patch_im_mask = img_mask,
                                           model=model,
                                           iteration=I_d,
                                           lr=lr_d,
                                           loss_seg=100,
                                           loss_fn=loss_fn_dis)
    img = d_im
    EoD_loss_list.extend(loss_list)
    tem_patch_im, tem_patch_im_mask, loss_list = adv_patch(img = img,
                                                           patch_im = patch_im,
                                                           patch_im_mask=patch_im_mask,
                                                           model=model,
                                                           iteration=I_s,
                                                           lr=lr_s,
                                                           loss_fn=loss_fn_sal,
                                                           loss_seg=100)
    EoD_loss_list.extend([-1*i for i in loss_list])
    return tem_patch_im, tem_patch_im_mask, EoD_loss_list





