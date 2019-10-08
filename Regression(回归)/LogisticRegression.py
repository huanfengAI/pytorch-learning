import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


with open('data','r') as f:
   data_list=f.readlines()#读取所有返回一个列表,txt每一行表示一个整体
   data_list=[i.split('\n')[0] for i in data_list]#去掉每一行之后的\n
   data_list=[i.split(',') for i in data_list]
   #print(data_list)#[[x1,x2,y],[],[]...[]]
   data=[]
   #过滤掉缺失的数据
   for i in data_list :
       if len(i)!=1:
           data.append(i)
   data=[(float(i[0]),float(i[1]),float(i[2])) for i in data]
   





x_data=[[i[0],i[1]] for i in data]
y_data=[[i[2]] for i in data]
x_data=np.array(x_data)
x_data=torch.from_numpy(x_data)
x_data=x_data.float()
y_data=np.array(y_data)
y_data=torch.from_numpy(y_data)
y_data=y_data.float()





class LogisticRegression(nn.Module) : 

  def __init__(self) :
    super (LogisticRegression, self). __init__()
    self.lr=nn.Linear(2,1)
    self.sm=nn.Sigmoid()

  def forward(self,x): 
    x=self.lr(x)
    x=self.sm(x) 
    return x 

logistic_model = LogisticRegression() 
if torch.cuda.is_available() : 
       logistic_model.cuda() 
criterion = nn.BCELoss () 
optimizer = torch.optim.SGD(logistic_model.parameters() , lr=1e-3)
for epoch in range(100000): 
  if torch.cuda.is_available() :
     x=x_data.cuda() 
     y=y_data.cuda() 
  else: 
     x = x_data
     y = y_data 
     
  out = logistic_model(x)
  loss= criterion(out, y) 
  print_loss = loss.item()
  mask=out.ge(0.5).float() 
  correct=(mask == y).sum () 
  acc = correct.item()/x.size(0) 
  optimizer.zero_grad() 
  loss.backward () 
  optimizer.step () 
  if (epoch+1) %1000==0: 
    print('*'*10) 
    print('epoch {}' . format (epoch+1)) 
    print ('loss is {:.4f}'.format(print_loss)) 
    print ( 'acc i8 {:.4f} '. format (acc) ) 


x0=list(filter(lambda x :x[-1]==0.0,data))
x1=list(filter(lambda x :x[-1]==1.0,data))
plot_x0_0=[i[0] for i in x0]
plot_x0_1=[i[1] for i in x0]
plot_x1_0=[i[0] for i in x1]
plot_x1_1=[i[1] for i in x1]
plt.plot(plot_x0_0,plot_x0_1,'ro',label='x_0')
plt.plot(plot_x1_0,plot_x1_1,'bo',label='x_1')
plt.legend(loc='best')


w0,w1 = logistic_model.lr.weight.data[0]
w0 = w0.item()
w1 = w1.item()
b = logistic_model.lr.bias.data[0] 
plot_x = np.arange(30, 100, 0.1) 
plot_x=torch.from_numpy(plot_x)
plot_x=plot_x.float()
plot_y=(-w0*plot_x-b)/w1
plot_x=plot_x.numpy()
plot_y=plot_y.numpy()
plt.plot(plot_x, plot_y) 
plt.show()






