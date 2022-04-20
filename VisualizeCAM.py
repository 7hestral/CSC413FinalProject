import datetime
import pickle

from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import argparse
from torchvision import transforms, datasets, models
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from LayerCAM.cam.layercam import LayerCAM
from LayerCAM.utils import *

from Train import get_model

model_name = 'resnet18'

args1 = {
            'model': model_name,
            'pretrain': True,
            'data_size': 1,
        }
args2 = {
            'model': model_name,
            'pretrain': False,
            'data_size': 1,
        }

model_folder = '2022-04-16-05-43'

device = torch.device('cuda:0')
save_name = f"{args1['model']}-{str(args1['data_size'])}-{str(args1['pretrain'])}-80"
_, model1 = get_model(args1)
model1 = model1.to(device=device)
model1.load_state_dict(torch.load(os.path.join(f'./{model_folder}', f"{save_name}-model.pth")))
model1.eval()
model1_dict = dict(type=model_name, arch=model1, layer_name='conv2', input_size=(299, 299))
model1_layercam = LayerCAM(model1_dict)

model2 = save_name = f"{args2['model']}-{str(args2['data_size'])}-{str(args2['pretrain'])}-80"
_, model2 = get_model(args2)
model2 = model2.to(device=device)
model2.load_state_dict(torch.load(os.path.join(f'./{model_folder}', f"{save_name}-model.pth")))
model2.eval()
model2_dict = dict(type=model_name, arch=model2, layer_name='conv2', input_size=(299, 299))
model2_layercam = LayerCAM(model2_dict)

data_path = './plant-seedlings-classification'
save_folder = 'pt_true_npt_true'
transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.3288, 0.2894, 0.2073), std=(0.1039, 0.1093, 0.1266))
])

whole_ds = datasets.ImageFolder(os.path.join(data_path, 'mytest'), transform=transform)
data_loader = DataLoader(whole_ds,
                         batch_size=1,
                         num_workers=0,
                         shuffle=False)


i = 0
for batch in data_loader:
    img = batch[0].to(device)
    label = batch[1].to(device)

    logits1 = model1(img)
    logits2 = model2(img)

    pred1 = logits1.max(1)[-1].item()
    pred2 = logits2.max(1)[-1].item()
    label = label.item()

    # if pred1 == label and pred2 == pred1:
    if i == 506:
        layer_map1 = model1_layercam(img)
        layer_map2 = model2_layercam(img)
        basic_visualize(img.cpu().detach(), layer_map1.type(torch.FloatTensor).cpu(),
                        save_path=f'./{save_folder}/actual-{label}-i{i}-pt-{pred1}.png', alpha=0.5)
        basic_visualize(img.cpu().detach(), layer_map2.type(torch.FloatTensor).cpu(),
                        save_path=f'./{save_folder}/actual-{label}-i{i}-npt-{pred2}.png', alpha=0.5)
    # if pred1 != label and pred2 == label:
    #     layer_map1 = model1_layercam(img)
    #     layer_map2 = model2_layercam(img)
    #     basic_visualize(img.cpu().detach(), layer_map1.type(torch.FloatTensor).cpu(),
    #                     save_path=f'./npt_true_pt_false/actual-{label}-i{i}-pt-{pred1}.png', alpha=0.5)
    #     basic_visualize(img.cpu().detach(), layer_map2.type(torch.FloatTensor).cpu(),
    #                     save_path=f'./npt_true_pt_false/actual-{label}-i{i}-npt-{pred2}.png', alpha=0.5)
    i += 1


