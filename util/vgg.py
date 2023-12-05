import os

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

device = "cpu"
data_dir = "./dataset"
cat16_dir = data_dir + "/cat16"

def load_cat16(return_not_normalized = False):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cat16_files = np.sort([f for f in os.listdir(cat16_dir) if f.endswith('.jpeg')]) 
    cat16_raw = [Image.open(os.path.join(cat16_dir, file)) for file in cat16_files]
    cat16_cropped = [preprocess(img) for img in cat16_raw]
    cat16 = norm(torch.stack(cat16_cropped))

    if return_not_normalized:
        return cat16, cat16_cropped

    return cat16