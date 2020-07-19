import torch
from torch import nn
dict= {'huan': 0 , 'feng': 1}
embeds =nn.Embedding(2,5)
#获取huan的词向量
index=torch.LongTensor([dict['huan']])
print(index)#0
vector=embeds(index)
print(vector)
#tensor([[-0.6942, -0.4719, -0.0518, -0.4731,  0.5416]], grad_fn=<EmbeddingBackward>)

