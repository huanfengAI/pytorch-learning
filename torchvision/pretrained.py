from torchvision import models
from torch import nn
squeezenet1_1=models.squeezenet1_1(pretrained=True,num_classes=1000)
squeezenet1_1.fc=nn.Linear(512,10)#然后修改其全连接层，使其变为10分类的问题
print(squeezenet1_1)
