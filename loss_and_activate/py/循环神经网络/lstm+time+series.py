import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable

data_csv =pd.read_csv('./data.csv', usecols=[1])

data_csv = data_csv.dropna()#缺失值处理函数dropna：去除数据结构中值为空得数据。

dataset = data_csv.values

dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))#进行均值归一化操作


#创建数据集
#将前两个月的数据作为输入，然后第三个月的数据作为标签
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 创建好输入输出
data_X, data_Y = create_dataset(dataset)
print(len(data_X))
# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
print(len(train_X))
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

#所有的数据只有一个batch
train_X = train_X.reshape(-1, 1, 2)

#print(train_Y.shape)(99, 1)


train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)


train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        #print(x.shape)#torch.Size([99, 1, 2])

        x, _ = self.rnn(x) # (seq, batch, hidden)
        #print(x.shape)#torch.Size([99, 1, 4])
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
       
        #print(x.shape)torch.Size([99, 4])
        x = self.reg(x)

        #print(x.shape)torch.Size([99, 1])
        x = x.view(s, b, -1)
       
        #print(x.shape)#torch.Size([99, 1, 1])
        return x

net = lstm_reg(2, 4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
   
    #print(var_y.shape)
    #torch.Size([99, 1, 1])

    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

net = net.eval() # 转换成测试模式
data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()


