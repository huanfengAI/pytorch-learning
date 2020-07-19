import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from  torch.autograd import Variable 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 


batch_size=64
learning_rate=1e-2
num_epoches=10






#带有皮标准层和激活函数的三层全连接神经网络
class BatchNet(nn.Module):
  def __init__(self, in_dim,n_hidden_1, n_hidden_2, out_dim) : 
      
      super(BatchNet,self).__init__()
      self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
   
      self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2), nn.BatchNorm1d(n_hidden_2),nn.ReLU(True)) 
      self.layer3 = nn.Sequential(nn.Linear (n_hidden_2,out_dim)) 
  def forward(self,x): 
      x=self.layer1(x)
      x=self.layer2(x) 
      x=self.layer3(x) 
      return x 
  



data_tf =transforms.Compose(
 [
  transforms.ToTensor(),
  transforms.Normalize([0.5],[0.5])
]
)


train_dataset=datasets.MNIST(root='./adata',train=True,transform=data_tf,download=True) 

test_dataset=datasets.MNIST(root='./adata',train=False,transform=data_tf) 

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size, shuffle=False) 



model=BatchNet(28 * 28, 300, 100, 10)

if torch.cuda.is_available() : 
   model = model. cuda () 
 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters() , lr=learning_rate)



for epoch in range(num_epoches):
    loss_sum, cort_num_sum,acc = 0.0, 0,0
    for data in train_loader:#train_loader表示所有的数据，在他定义的时候batch_size=4,则遍历的时候data就是多少
        img,label=data
       # print(img.size())#batch总size=batch×通道×height×weight
       # print(img.size(1))#通道
       # print(img.size(2))#height
       # print(img.size(3))#weight
        
        img=img.view(img.size(0),-1)#转成batch行，因为通道是1，所以列为height*weight
        if torch.cuda.is_available():
            inputs = Variable(img).cuda()
            target = Variable(label).cuda()
        else:
            inputs = Variable(img)
            target = Variable(label)
        output =model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.data
        a, pred = output.data.max(1)#a表示具体的数值，pred表示具体的索引
        num_correct = pred.eq(target).sum()
        cort_num_sum += num_correct
    acc=cort_num_sum.float()/len(train_dataset)
    print( "After %d epoch , training loss is %.2f , correct_number is %d  accuracy is %.6f. "%(epoch,loss_sum,cort_num_sum,acc))



model.eval()
eval_loss=0 
eval_acc=0
for data in test_loader: 
  img, label=data 
  img=img.view(img.size(0),-1) 
  if torch.cuda.is_available( ) : 
     img=Variable(img,volati1e=True).cuda () 
     label=Variable(label,volatile=True).cuda() 
  else: 
     img=Variable(img, volatile=True) 
     label=Variable(label,volatile=True) 
    

  out=model(img)
  loss=criterion(out,label)
  print(loss)
  print(loss.data)
 
  eval_loss+=loss.data*label.size(0)
  _,pred=out.data.max(1)
  num_correct=pred.eq(label).sum()
  eval_acc+=num_correct.data
print('Test loss: {:.6f},ACC: {:.6f}'.format(eval_loss.float()/(len(test_dataset)),eval_acc.float()/(len(test_dataset))))



