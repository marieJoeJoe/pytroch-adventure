#! /usr/bin/env python3
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import inspect

# 构建数据集
EN_TEXT = torchtext.datasets.IWSLT2016(split='train', language_pair=('en', 'fr'))
FR_TEXT = torchtext.datasets.IWSLT2016(split='train', language_pair=('fr', 'en'))

print(f'Epoch: { inspect.currentframe().f_lineno}')
# 创建分词器
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

print(f'Epoch: { inspect.currentframe().f_lineno}')
# 创建词汇表
def build_vocab(text, tokenizer):
    def yield_tokens(data_iter):
        for text_entry in data_iter:
            yield tokenizer(text_entry[0])
    return build_vocab_from_iterator(yield_tokens(text), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

en_vocab = build_vocab(EN_TEXT, en_tokenizer)
fr_vocab = build_vocab(FR_TEXT, fr_tokenizer)

# 创建数据加载器
def data_process(filepaths):
    raw_en_iter = iter(filepaths[0])
    raw_fr_iter = iter(filepaths[1])
    data = []
    for (raw_en, raw_fr) in zip(raw_en_iter, raw_fr_iter):
        en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en[0])],
                                 dtype=torch.long)
        fr_tensor = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr[0])],
                                 dtype=torch.long)
        data.append((en_tensor, fr_tensor))
    return data

train_data = data_process((EN_TEXT, FR_TEXT))

# 创建批次数据
BATCH_SIZE = 128
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

def generate_batch(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
    return en_batch, fr_batch

print(f'Epoch: { inspect.currentframe().f_lineno}')
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
print(f'Epoch: { inspect.currentframe().f_lineno}')
# 初始化模型参数
d_model = 512
nhead = 8
num_layers = 6
dropout = 0.5
dim_feedforward = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化一个Transformer模型
model = nn.Transformer(d_model, nhead, num_layers, dim_feedforward, dropout).to(device)

# 定义训练过程
def train(model, iterator, optim):
    model.train()
    total_loss = 0
    for idx, (src, tgt) in enumerate(iterator):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_real = tgt[1:, :]
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        optim.zero_grad()
        out = model(src, tgt_input, tgt_mask=tgt_mask)
        loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)(out.reshape(-1, out.shape[-1]), tgt_real.reshape(-1))
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(iterator)

print(f'Epoch: { inspect.currentframe().f_lineno}')
# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters())

print(f'Epoch: { inspect.currentframe().f_lineno}')
# 开始训练
EPOCHS = 10
for epoch in range(EPOCHS):
    print(f'Epoch: { inspect.currentframe().f_lineno}')
    loss = train(model, train_iter, optimizer)
    print(f'Epoch: {epoch}, Loss: {loss}')
