import torch
from torch import nn
from torch.autograd import Variable
training_data = [("The monkey ate the banana".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("The dog ate the bones".split(), 
                  ["DET", "NN", "V", "DET", "NN"])]

#print(training_data)

word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)

#print(word_to_idx)
#{'the': 0, 'dog': 1, 'ate': 2, 'apple': 3, 'everybody': 4, 'read': 5, 'that': 6, 'book': 7}
#print(tag_to_idx)
#{'det': 0, 'nn': 1, 'v': 2}

z = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(z)):
    char_to_idx[z[i]] = i


def make_vector(x, dic): # 字符编码
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx


class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()
        self.char_embed = nn.Embedding(n_char, char_dim)
        self.lstm = nn.LSTM(char_dim, char_hidden)
        
    def forward(self, x):
        #可以输入给embedding层任意的维度，然后输出为输入维度×词向量的维度
        x = self.char_embed(x)
        #print(x.shape)#torch.Size([4, 1, 10])
        out, _ = self.lstm(x)
        return out[-1]


class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, word_dim, 
                 char_hidden, word_hidden, n_tag):
        super(lstm_tagger, self).__init__()
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_embed = nn.Embedding(n_word, word_dim)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)
        
    def forward(self, x, word):
        char = []
        for w in word: # 对于每个单词做字符的 lstm  
            char_list = make_vector(w, char_to_idx)
           # print(char_list.shape)#torch.Size([4])
            char_list = char_list.unsqueeze(1) 
            #print(char_list.shape)#torch.Size([4, 1])
            char_infor = self.char_lstm(Variable(char_list)) # (batch, char_hidden)
            #print(char_infor.shape)#torch.Size([1, 50])
            char.append(char_infor)
       
        char = torch.stack(char, dim=0) # (seq, batch, feature)
        #print(char)#torch.Size([4, 1, 50])
        #print(x.shape)#torch.Size([1, 5])
        x = self.word_embed(x) # (batch, seq, word_dim)
       # print(x.shape)#torch.Size([1, 5,100])
        x = x.permute(1, 0, 2) # 改变顺序
        x = torch.cat((x, char), dim=2) 
        #print(x.shape)#torch.Size([5, 1, 150])
        x, _ = self.word_lstm(x)
        #print(x.shape)#torch.Size([5, 1, 128])
        s, b, h = x.shape
        x = x.view(-1, h)#torch.Size([5,128])5可以看成是batch,5其实是每一个单词，也就是说每一个单词都会有一个输出的
        out = self.classify(x)
        return out



net = lstm_tagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 开始训练
for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_vector(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_vector(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        #print(tag)
        # 前向传播
        out = net(word_list, word)
        #print(out)
        loss = criterion(out, tag)
        train_loss += loss.data[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))



net = net.eval()
test_sent = 'The dog ate the banana'
test = make_vector(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
print(out.max(1)[1].data)

