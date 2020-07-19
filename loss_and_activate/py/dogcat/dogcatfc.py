#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
#本案例是使用resnet，然后在此基础上只训练全连接层的参数
import os
import shutil
import torch
from torch import nn
from torchvision import datasets,transforms,models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    num_correct = (pred_label == label).sum().float()
    return num_correct / total
#kaggle/train/下面是所有训练数据，但是这个数据集中有猫还有狗，所以我们第一步是数据处理，将猫和狗区分开
datafile=os.listdir('kaggle/train/')#路径不能写为/kaggle/train
c=list(filter(lambda x:x[0:3]=='cat',datafile))
d=list(filter(lambda x:x[0:3]=='dog',datafile))
#现在我们要将数据集分为两部分，一部分作为训练集数据，比例是90%，另外一部分是测试集数据，比例是10%
root='kaggle/'
train_root='dataset/train/'#要将dataset/train和dataset/val目录创建好
val_root  ='dataset/val/'
for i in range(len(c)):
	pic_path =root +'train/'+c[i]#原来的数据集的位置
	if i < len(c)*0.9:#数据集还没有遍历到百分之90%,那么数据存放路径就放在了下面这个文件夹
		obj_path=train_root+'cat/'+c[i]
	else:
		obj_path=val_root+'cat/'+c[i]#如果数据已经遍历到了10%，那么数据路径就放在这个文件夹中
	shutil.move(pic_path,obj_path)#移动数据

for i in range(len(d)):
	pic_path =root+'train/'+ d[i]
	if i< len(d)*0.9:
		obj_path=train_root+'dog/'+d[i]
	else:
		obj_path=val_root+'dog/'+d[i]
	shutil.move(pic_path,obj_path)
	
#		
net=models.resnet18(pretrained=True)
print(net)
import ipdb
ipdb.set_trace()
dim_in=net.fc.in_features#获得最后一层全连接层的fc的输入单元的数量in_features,用于建立新的全连接层

for param in net.parameters():#冻结所有的参数，然后在optimizer指定要进行梯度下降的参数
	param.requires_grad=False
net.fc=nn.Linear(dim_in,2)	
transform =transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.4,0.4,0.4],std=[0.2,0.2,0.2])
])

train_set =ImageFolder('dataset/train/',transform=transform)#加载数据，并且进行数据处理
test_set =ImageFolder('dataset/val/',transform=transform)#加载数据，并且进行数据处理


test_data =DataLoader(test_set, batch_size=128, shuffle=False)
train_data =DataLoader(train_set, batch_size=64, shuffle=True)

criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.fc.parameters(),lr=1e-1)#只训练全连接层的参数
#训练
prev_time=datetime.now()
for epoch in range(30):
	train_loss=0
	train_acc =0
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
		#前向传播并计算损失
		#print(im.size())#im=im.view(im.size(0),-1)torch.Size([64, 1, 28, 28])
		#im=im.view(im.size(0),-1)
		#print(im.size())torch.Size([64, 784])
		output =net(im)
		
		loss =criterion(output ,label)
		#反向传播
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		#print(loss.data)
		train_loss +=loss.data.float()
		train_acc +=get_acc(output,label)
		
		#print(train_acc/len(train_data))
		#print(train_acc/64)
	#测试
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
		#im=im.view(im.size(0),-1)
		output =net(im)
		loss= criterion(output,label)
		valid_loss +=loss.data.float()
		valid_acc +=get_acc(output,label)
	epoch_str=(
			"Epoch %d. Train Loss %f,Train Acc:%f,Valid Loss: %f,Valid Acc: %f ,"
			%(epoch,train_loss/len(train_data),
			  train_acc /len(train_data),
			  valid_loss/len(test_data),
			  valid_acc /len(test_data)))
	prev_time=cur_time
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
