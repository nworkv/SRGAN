# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:59:01 2025

@author: nikita
"""

import torch
from models.generator import Generator
import matplotlib.pyplot as plt
from dsdl import to_device, denorm, transform
from torchvision.utils import make_grid
from PIL import Image


def load_model(model, PATH, device):
  state_dict = torch.load(PATH, weights_only=True, map_location=torch.device(device))
  model.load_state_dict(state_dict)
  return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator()
generator = to_device(load_model(generator, "implementation/generator_20250708_105415", device) , device)
print("generator has been loaded")

image_size = 256
image = Image.open("image.png")
print("image has been opened")


Image_HR = transform(image, image_size, 1)
Image_LR = transform(image, image_size, 0)
image_tensor = to_device(torch.tensor(Image_LR), device).view(1, 3, 64, 64)

GImage_HR = generator(image_tensor).cpu().detach()

fig, axs = plt.subplots(1, 3, figsize=(15, 15))
axs[0].imshow(make_grid(denorm(Image_HR)).permute(1, 2, 0))
axs[1].imshow(make_grid(denorm(Image_LR)).permute(1, 2, 0))
axs[2].imshow(make_grid(denorm(GImage_HR)).permute(1, 2, 0))