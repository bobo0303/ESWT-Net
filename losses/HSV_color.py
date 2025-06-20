import cv2, torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as FF
import scipy.stats as st
import math

def define_gaussian_kernal(kernel_size=3,sigma=2,channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size -1)/2.
    variance = sigma**2.

    gaussian_kernal = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
    gaussian_kernal = gaussian_kernal / torch.sum(gaussian_kernal)
    gaussian_kernal = gaussian_kernal.view(1, 1, kernel_size, kernel_size)
    gaussian_kernal = gaussian_kernal.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size ,groups=channels, bias=False, padding=kernel_size//2)
    gaussian_filter.weight.data = gaussian_kernal
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def BGR2HSV(img, pred_img, edge, batch):
    img_hsv = []
    pre_img_hsv = []
    edge_mask_list = []
    for n in range(batch):
        numberofimg_from = n
        numberofimg = n+1
        img_ = img[numberofimg_from:numberofimg, ...]
        img_ = img_.permute(0, 2, 3, 1) * 255
        img_ = np.concatenate(img_.cpu().detach().numpy().astype(np.uint8), axis=0)  # GT
        hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

        pred_img_ = pred_img[numberofimg_from:numberofimg, ...]
        pred_img_ = pred_img_.permute(0, 2, 3, 1) * 255
        pred_img_ = np.concatenate(pred_img_.cpu().detach().numpy().astype(np.uint8), axis=0)  # pre
        pre_hsv = cv2.cvtColor(pred_img_, cv2.COLOR_BGR2HSV)

        hsv = FF.to_tensor(hsv).float()
        pre_hsv = FF.to_tensor(pre_hsv).float()
        img_hsv.append(hsv)
        pre_img_hsv.append(pre_hsv)
        if edge != None:
            edge_ = edge[numberofimg_from:numberofimg, ...]
            edge_ = edge_.permute(0, 2, 3, 1) * 255
            edge_ = np.concatenate(edge_.cpu().detach().numpy().astype(np.uint8), axis=0)  # edge
            edge_mask = FF.to_tensor(edge_).float()
            edge_mask_list.append(edge_mask)
    if batch == 8:
        img_hsv = np.stack([img_hsv[0],img_hsv[1],img_hsv[2],img_hsv[3],img_hsv[4],img_hsv[5],img_hsv[6],img_hsv[7]],axis=0)
        pre_img_hsv = np.stack([pre_img_hsv[0],pre_img_hsv[1],pre_img_hsv[2],pre_img_hsv[3],pre_img_hsv[4],pre_img_hsv[5],pre_img_hsv[6],pre_img_hsv[7]],axis=0)
    elif batch == 4:
        img_hsv = np.stack([img_hsv[0],img_hsv[1],img_hsv[2],img_hsv[3]],axis=0)
        pre_img_hsv = np.stack([pre_img_hsv[0],pre_img_hsv[1],pre_img_hsv[2],pre_img_hsv[3]],axis=0)
    elif batch == 16:
        img_hsv = np.stack([img_hsv[0],img_hsv[1],img_hsv[2],img_hsv[3],img_hsv[4],img_hsv[5],img_hsv[6],img_hsv[7],img_hsv[8],img_hsv[9],img_hsv[10],img_hsv[11],img_hsv[12],img_hsv[13],img_hsv[14],img_hsv[15]],axis=0)
        pre_img_hsv = np.stack([pre_img_hsv[0],pre_img_hsv[1],pre_img_hsv[2],pre_img_hsv[3],pre_img_hsv[4],pre_img_hsv[5],pre_img_hsv[6],pre_img_hsv[7],pre_img_hsv[8],pre_img_hsv[9],pre_img_hsv[10],pre_img_hsv[11],pre_img_hsv[12],pre_img_hsv[13],pre_img_hsv[14],pre_img_hsv[15]],axis=0)
    else:
        assert batch != 8 and batch != 4 and batch != 16, \
            'please fix it by yourself ex. like above'

    img_hsv = torch.from_numpy(img_hsv).float()
    pre_img_hsv = torch.from_numpy(pre_img_hsv).float()

    if edge != None:
        edge_mask_list = np.stack([edge_mask_list[0],edge_mask_list[1],edge_mask_list[2],edge_mask_list[3]],axis=0)
        edge_mask_list = torch.from_numpy(edge_mask_list).float()

    return img_hsv, pre_img_hsv, edge_mask_list

def HSV(img, pred_img, edge):
    batch = img.shape[0]
    gauss_blur = define_gaussian_kernal().cuda()
    img = gauss_blur(img)
    pred_img = gauss_blur(pred_img)
    hsv, pre_hsv, edge = BGR2HSV(img, pred_img, edge, batch)
    #####
    losses = F.mse_loss(hsv, pre_hsv, reduction='mean')
    losses_H=losses_S=losses_V= None
    #####
    # loss = F.mse_loss(hsv, pre_hsv, reduction='none')
    # if edge != []:
    #     loss = loss * (1 - edge) + 10 * loss * edge
    # losses_H = loss[0].mean()
    # losses_S = loss[1].mean()
    # losses_V = loss[2].mean()
    # losses = (losses_H + losses_S + losses_V)

    return losses_H,losses_S,losses_V,losses

