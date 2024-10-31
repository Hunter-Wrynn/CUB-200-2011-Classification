import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import alpha_clip
from bird2 import Bird

# Step 1: 加载类别名称
def load_class_texts(classes_txt_path):
    with open(classes_txt_path, 'r') as f:
        class_texts = [
            ' '.join(line.strip().split(' ', 1)[1].split('.')[1].replace('_', ' ').split())
            for line in f.readlines()
        ]
    return class_texts

# 加载权重文件并去除 'module.' 前缀
def load_weights(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)

    
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 路径
model_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/models/alpha_clip_last.pth"
classes_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/classes.txt"
images_dir = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/images"
train_txt_path = "/ailab/user/mahaoxuan/BIRD/birdDataSet_mixed/bird/trainDataSet_mixed.txt"

# 加载类别文本
class_texts = load_class_texts(classes_txt_path)

print(class_texts)