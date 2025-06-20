from fid_score.fid_score import FidScore
import cv2, lpips, os
import torch
import torchvision.transforms.functional as F
import colorful as c

'''
pip install fid-score 0.1.3
with scipy 1.3.2
it is clash with scikit-iamge 0.19.3 (scipy>=1.4.1)
'''

# do what set
do_FID = True
do_LPIPS = True

# Source and target path
source = r'G:\FFHQ\val/'
target = r'G:\journal\out\FFHQ_69500_500_10+20'
source_list = os.listdir(source)
target_list = os.listdir(target)

# FID set
score = None
paths = [source,target]
device = torch.device('cuda:0')
batch_size = 32

#LPIPS set
LPIPS_center = []
gpu = torch.device('cuda:0')
alex = loss_fn_alex = lpips.LPIPS(net='alex').to(gpu)  # best forward scores

#FID
if do_FID == True:
    fid = FidScore(paths, device, batch_size)
    score = fid.calculate_fid_score()

#LPIPS
if do_LPIPS == True:
    for GT, Pred in zip(source_list, target_list):
        GT_ = cv2.imread(source+GT).copy()
        GT_ = GT_[:, :, ::-1].copy()
        GT_ = F.to_tensor(GT_).float().to(gpu)
        Pred_ = cv2.imread(target+Pred).copy()
        Pred_ = Pred_[:, :, ::-1].copy()
        Pred_ = F.to_tensor(Pred_).float().to(gpu)
        LPIPS = alex(GT_, Pred_)
        LPIPS_center.append(LPIPS.mean())

print("FID:  ", score)
print("LPIPS:  ", LPIPS_center.mean())