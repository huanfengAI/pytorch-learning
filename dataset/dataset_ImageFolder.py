import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from torchvision import datasets,transforms
import random
transform=transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(220),
	transforms.ToTensor(),
	transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])

])

dataset=datasets.ImageFolder('dogcat/',transform=transform)
print(dataset[0][0].shape)
print(dataset[0][1])
print(dataset[1][0].shape)
print(dataset[1][1])
print(dataset[3][0].shape)
print(dataset[3][1])
