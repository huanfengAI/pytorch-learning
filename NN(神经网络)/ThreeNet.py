import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 


batch_size=64
learning_rate=1e-2
num_epoches=10

#三层的全连接神经网络
class ThreeNet(nn.Module) :
   def __init__ (self,in_dim,n_hidden_1,n_hidden_2,out_dim):
     super(ThreeNet, self).__init__()
     self.layer1 = nn.Linear (in_dim, n_hidden_1) 
     self.layer2 = nn.Linear(n_hidden_1,n_hidden_2) 
     self.layer3 = nn.Linear(n_hidden_2, out_dim)
   def forward(self,x): 
     x =self.layer1(x) 
     x =self.layer2(x) 
     x =self.layer3(x) 
     return x

data_tf =transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5],[0.5])
])

train_dataset=datasets.MNIST(root='./adata',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./adata',train=False,transform=data_tf,download=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size, shuffle=False) 

model=ThreeNet(28 * 28, 300, 100, 10)

if torch.cuda.is_available() : 
   model = model. cuda () 
 
criterion = nn.CrossEntropyLoss() #交叉熵损失
optimizer = optim.SGD(model.parameters() , lr=learning_rate)
#手写字体通道为1,大小是28*28
for epoch in range(num_epoches):
    loss_sum, cort_num_sum,acc = 0.0, 0,0
    for data in train_loader:#train_loader表示所有的数据，在他定义的时候batch_size=4,则遍历的时候data就是多少
        img,label=data#总size=batch×通道数×height×weight
        #print(img.size())#torch.size([batch,通道,height,weight])
        #print(img.size(0))#batch
        #print(img.size(1))#通道
        #print(img.size(2))#height
        #print(img.size(3))#weight
        
        img=img.view(img.size(0),-1)
        #转成batch行，因为通道是1，所以列为height*weight，也就是将图片展开了，这是输入到全连接神经网络必须的操作
        if torch.cuda.is_available():
            inputs = img.cuda()
            target = label.cuda()
        else:
            inputs = img
            target = label
        output =model(inputs)
        print(output.shape)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.data#因为一次没有喂入所有数据，而是喂入batch数据，所以要将损失累加
        a, pred = output.data.max(1)#a表示具体的数值，pred表示具体的索引
        num_correct = pred.eq(target).sum()#求正确的个数
        cort_num_sum += num_correct#将所有的batch正确的个数累加
    acc=cort_num_sum.float()/len(train_dataset)#计算准确率
    print( "After %d epoch , training loss is %.2f , correct_number is %d  accuracy is %.6f. "%(epoch,loss_sum,cort_num_sum,acc))



model.eval()#开启测试
eval_loss=0 
eval_acc=0
for data in test_loader: 
  img, label=data 
  img=img.view(img.size(0),-1) 
  if torch.cuda.is_available( ) : 
     img=img.cuda () 
     label=label.cuda() 
  else: 
     img=img
     label=label
    

  out=model(img)
  loss=criterion(out,label)
  print(loss)
  print(loss.data)
 
  eval_loss+=loss.data*label.size(0)
  _,pred=out.data.max(1)
  num_correct=pred.eq(label).sum()
  eval_acc+=num_correct.data
print('Test loss: {:.6f},ACC: {:.6f}'.format(eval_loss.float()/(len(test_dataset)),eval_acc.float()/(len(test_dataset))))



