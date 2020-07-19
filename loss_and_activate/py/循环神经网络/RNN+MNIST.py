import torch
from torch import nn
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)#求每行的最大就是最有可能的类别
    num_correct = (pred_label == label).sum().float()
    return num_correct / total

data_tf=transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize([0.5],[0.5])
]
)
train_set =mnist.MNIST('./adata',train=True,transform=data_tf,download=True)
test_set  =mnist.MNIST('./adata',train=False,transform=data_tf,download=True)
train_data=DataLoader(train_set,batch_size=64,shuffle=True)
test_data =DataLoader(test_set,batch_size=128,shuffle=True)





class RNN(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(RNN,self).__init__()
        self.n_layer=n_layer
        self.hidden_dim=hidden_dim
        self.lstm=nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True)
        self.classifier=nn.Linear(hidden_dim,n_class)

    def forward(self,x):
        out,_ =self.lstm(x)
        out =out[:,-1,:]
        out =self.classifier(out)
        return out

net=RNN(28,50,2,10)#因为图片的大小是28，所以输出层要是28
criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),1e-1)


#训练
prev_time=datetime.now()
for epoch in range(30):
	train_loss=0
	train_acc =0
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
                im = im.squeeze(1)
                output =net(im)
		
                loss =criterion(output ,label)
		#反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss +=loss.data.float()
                train_acc +=get_acc(output,label)
	#测试
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
		im = im.squeeze(1)
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
	print(epoch_str+time_str)

