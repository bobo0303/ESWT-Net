import os
import random, cv2, lpips
import numpy as np
import torch
from pytorch_msssim import ssim
import colorful as c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def Visualization_of_training_results(first_out, second_out, input_img, input_mask, save_path, iterations):

    # without edge
    current_img = input_img[:4, ...]
    current_img = current_img.permute(0, 2, 3, 1) * 255
    original_img = np.concatenate(current_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT

    mask = input_mask[:4, ...].permute(0, 2, 3, 1)
    current_img = (current_img * (1 - mask)).cpu().detach().numpy().astype(np.uint8)
    current_img = np.concatenate(current_img, axis=0)  # GT with masks

    first_out_output = first_out[:4, :, :, :]
    first_out_output = first_out_output.permute(0, 2, 3, 1) * 255
    first_out_output = np.concatenate(first_out_output.cpu().detach().numpy().astype(np.uint8), axis=0)  # first_out

    if second_out != None:
        second_out_output = second_out[:4, :, :, :]
        second_out_output = second_out_output.permute(0, 2, 3, 1) * 255
        second_out_output = np.concatenate(second_out_output.cpu().detach().numpy().astype(np.uint8), axis=0)  # second_out

        output = np.concatenate([original_img, current_img, first_out_output, second_out_output],
                                axis=1)  # GT + GT with mask + first_out + second_out
    else:
        output = np.concatenate([original_img, current_img, first_out_output],
                             axis=1)  # GT + GT with mask + first_out

    save_path = save_path + '/samples'
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_path + '/' + str(iterations) + '.jpg', output[:, :, ::-1])

    return None

def Visualization_of_training_results_val(first_out, second_out, input_img, input_mask, save_path, iterations):

    # without edge
    current_img = input_img[:4, ...]
    current_img = current_img.permute(0, 2, 3, 1) * 255
    original_img = np.concatenate(current_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT

    mask = input_mask[:4, ...].permute(0, 2, 3, 1)
    current_img = (current_img * (1 - mask)).cpu().detach().numpy().astype(np.uint8)
    current_img = np.concatenate(current_img, axis=0)  # GT with masks

    first_out_output = first_out[:4, :, :, :]
    first_out_output = first_out_output.permute(0, 2, 3, 1) * 255
    first_out_output = np.concatenate(first_out_output.cpu().detach().numpy().astype(np.uint8), axis=0)  # first_out

    if second_out != None:
        second_out_output = second_out[:4, :, :, :]
        second_out_output = second_out_output.permute(0, 2, 3, 1) * 255
        second_out_output = np.concatenate(second_out_output.cpu().detach().numpy().astype(np.uint8),
                                           axis=0)  # second_out

        output = np.concatenate([original_img, current_img, first_out_output, second_out_output],
                                axis=1)  # GT + GT with mask + first_out + second_out
    else:
        output = np.concatenate([original_img, current_img, first_out_output],
                             axis=1)  # GT + GT with mask + first_out

    save_path = save_path + '/samples_val'
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_path + '/' + str(iterations) + '.jpg', output[:, :, ::-1])

    return None

def PSNR(GT, Pred):
    mse = torch.mean((GT - Pred) ** 2)
    PSNR = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return PSNR

def SSIM(GT, Pred):
    SSIM = ssim(GT, Pred, data_range=1.0, size_average=True)    #https://github.com/VainF/pytorch-msssim
    return SSIM

def LPIPS_SET():
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    #loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    return loss_fn_alex

def LPIPS(GT, Pred, alex):
    # LPIPS = loss_fn_vgg(GT, Pred,)  #https://pypi.org/project/lpips/
    LPIPS = alex(GT, Pred)  #https://pypi.org/project/lpips/
    LPIPS = LPIPS.mean()
    return LPIPS

def FID(GT, Pred):
    pass
    return FID  # noused

def save_img(Pred, save_img_path, name):
    for n in range(Pred.shape[0]):
        os.makedirs(save_img_path, exist_ok=True)
        pre_img = Pred[n:n+1, ...]
        pre_img = pre_img.permute(0, 2, 3, 1) * 255
        names = name[n]
        pre_img = np.concatenate(pre_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
        cv2.imwrite(save_img_path+str(names), pre_img[:, :, ::-1])
        print(c.magenta(save_img_path+str(names)))

def save_img_first_stage(Pred, save_img_path, name):
    for n in range(Pred.shape[0]):
        os.makedirs(save_img_path, exist_ok=True)
        pre_img = Pred[n:n+1, ...]
        pre_img = pre_img.permute(0, 2, 3, 1) * 255
        names = name[n]
        pre_img = np.concatenate(pre_img.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
        cv2.imwrite(save_img_path+str(names), pre_img[:, :, ::-1])
        print(c.magenta(save_img_path+str(names)))




