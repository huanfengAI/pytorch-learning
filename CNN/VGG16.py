import torch
from torch import nn

def vgg_block(num_convs,in_channels,out_channels):
	net=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
	for i in range(num_convs-1):
		net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
		net.append(nn.ReLU(inplace=True))
	net.append(nn.MaxPool2d(2,2))
	return nn.Sequential(*net)#*net是不要列表只要列表中的元素
	


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.layer1=vgg_block(3,3,64)
        self.layer2=vgg_block(3,64,128)
        self.layer3=vgg_block(4,128,256)
        self.layer4=vgg_block(4,256,512)
        self.layer5=vgg_block(4,512,512)
        self.fc1=nn.Linear(100352,4096)
        self.fc2=nn.Linear(4096,1000)
        self.fc3=nn.Linear(1000,10)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

data=torch.randn(2,3,224,224)        
vgg=VGG16()
out=vgg(data)
print(out.shape)#torch.Size([2, 10])
