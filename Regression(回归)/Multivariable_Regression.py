import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

input=torch.randn(64,3)
W_target = torch.FloatTensor([0.7, 2.5, 2.4]) .unsqueeze(1) 
b_target = torch.FloatTensor([0.8])
#b_target表示偏置项的b，y=w_target*x+b_target 
def f (x) :    
    return x.mm (W_target) + b_target[0] #x.mm()表示做矩阵乘法,这样就构成了我们的模型f
    
target=f(input)
print(input.shape)
print(target.shape)
class Multivariable_Regression(nn.Module):
   def __init__(self):
     super(Multivariable_Regression,self).__init__()
     self.poly=nn.Linear(3,1)#这里的模型输入是 3 维，输山是 1 维，
   def forward(se1f,x): 
     out = se1f.poly(x) 
     return out 
 
if torch.cuda.is_available() : 
   model = Multivariable_Regression().cuda() 
else: 
   model = Multivariable_Regression() 
 
criterion=nn.MSELoss()#定义损失函数，均方误差
optimizer=optim.SGD(model.parameters(),lr=1e-3)#定义优化器，梯度下降


while True:
   output =model(input)#使用模型进行预测
   loss=criterion(output,target)#计算误差
   print_loss=loss.item()
   print("当前损失",print_loss)
   optimizer.zero_grad()#在优化之前需要先将梯度归零a
   loss.backward()#反向传播
   optimizer.step()#完成参数的更新
   if print_loss <1e-3:
      break
#模型保存 
torch.save(model,'./model.pth')
torch.save(model.state_dict(),'./model_state.pth') 
