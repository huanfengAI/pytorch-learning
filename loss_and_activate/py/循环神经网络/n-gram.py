import torch
from torch import nn
from torch.autograd import Variable

sentence = """When I do count the clock that tells the time,
And see the brave day sunk in hideous night;
When I behold the violet past prime,
And sable curls all silver'd o'er with white:
When lofty trees I see barren of leaves,
Which erst from heat did canopy the herd,
And summer's green, all girded up in sheaves,
Born on the bier with white and bristly beard;
Then of thy beauty do I question make,
That thou among the wastes of time must go,
Since sweets and beauties do themselves forsake,
And die as fast as they see others grow;
And nothing 'gainst Time's scythe can make defence
Save breed, to brave him when he takes thee hence.""".split()

print(sentence)#生成一个列表



sample=[((sentence[i],sentence[i+1]),sentence[i+2]) for i in range(len(sentence)-2)]




print(sample)#((第一个词，第二个词),第三个词)


vocb=set(test_sentence)

#建立字典

word_to_idx={word : i for i ,word in enumerate(vocb)}
print(word_to_idx)

idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
print(idx_to_word)

CONTEXT_SIZE = 2 
EMBEDDING_DIM = 10 
class n_gram(nn.Module):
        #vocab_size表示字典中词的个数
        #context_size表示依据的单词数，也就是我们使用多少个单词来预测下一个单词
        #n_dim 表示词向量的维度
	def __init__(self,vocab_size,context_size=CONTEXT_SIZE,n_dim=EMBEDDING_DIM):
             super(n_gram,self).__init__()
            
             self.embed=nn.Embedding(vocab_size,n_dim)
             self.classify=nn.Sequential(
			nn.Linear(context_size* n_dim,128),
			nn.ReLU(True),
			nn.Linear(128,vocab_size)
		)
	def forward(self,x):
             #print(x.shape)#torch.Size([2])
             voc_embed=self.embed(x)
             #print(voc_embed)#torch.Size([2, 10])
             voc_embed=voc_embed.view(1,-1)
             out =self.classify(voc_embed)
             return out



net=n_gram(len(word_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)
for e in range(100):
    train_loss = 0
    for word, label in trigram: 
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        #print(word)#tensor([35, 85])
        #print("-----------")
        #print(label)#tensor([48])
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.data[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))


#获取clock的词向量
index=torch.LongTensor([test_sentence.index('clock')])
print(index)#0
vector=net.embed(index)
print(vector)
