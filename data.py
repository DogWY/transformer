from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

# 获取分词器
tokenizer = Tokenizer()
# 制作loader
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en = tokenizer.tokenize_en,
                    tokenize_de = tokenizer.tokenize_de,
                    init_token = '<sos>',
                    eos_token = '<eos>',)

# 制作数据集
train, valid, test = loader.make_dataset()
# 根据训练集构建词表，词表只能在训练时构建，因此训练集和测试集都用训练集构建词表
loader.build_vocab(train_data=train, min_freq=2)
# 制作dataloader
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test, batch_size=batch_size, device=device)

# 获取填充的词表索引
src_pad_idx = loader.source.vocab.stoi['<pad>']
# sos标记用于表示生成开始，一般只有在解码器端才需要，因此不需要获取源词表的sos索引
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

# 获取词表大小
enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
