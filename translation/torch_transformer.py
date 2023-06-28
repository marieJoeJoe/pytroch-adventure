#! /usr/bin/env python3

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
import torch.nn as nn
import torch.optim as optim

# 定义分词器和词汇表
tokenizer = get_tokenizer('spacy')
train_iter = IMDB(split='train')
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# 定义数据加载器
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.float64).to(device), nn.utils.rnn.pad_sequence(text_list, padding_value=1.0).permute(1, 0).to(device)

from torch.utils.data import DataLoader
train_iter = IMDB(split='train')
dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# 定义一个基于transformer的模型
class Transformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, n_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, n_heads), 1)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output = self.transformer(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一步的输出
        return output


# 初始化模型
INPUT_DIM = len(vocab)
EMBEDDING_DIM = 96   # 更改嵌入维度使其能被头数整除
OUTPUT_DIM = 1
N_HEADS = 8
DROPOUT = 0.2
model = Transformer(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, N_HEADS, DROPOUT)

# 训练模型
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    for label, text in iterator:
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

for epoch in range(5):
    train_loss = train(model, dataloader, optimizer, criterion)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')



