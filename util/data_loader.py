from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

class DataLoader:
    source: Field = None
    target: Field = None
    
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token) -> None:
        """
        ext: 一个元组，表示源语言和目标语言的文件扩展名
        tokenize_en: 英文分词函数
        tokenize_de: 德文分词函数
        init_token: 句子的开始标记
        eos_token: 句子的结束标记
        """
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print("dataset initializing start")
        
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            """
            Field:
                使用tokenize对数据进行分词，该参数接收一个可调用对象
                init_token: 句子的开始标记
                eos_token: 句子的结束标记
                lower: 是否将所有字符转换为小写
                batch_first: 是否将batch的维度放在第一维度
            """
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, lower=True, batch_first=True)
            
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        """
        Multi30k.splits:
            该函数用于下载并返回Multi30k数据集的训练集、验证集和测试集。
            exts: 一个元组，表示源语言和目标语言的文件扩展名
            fields: 一个元组，表示源语言和目标语言的Field对象
        """
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        """
        Field.build_vocab:
            该函数用于构建词表，并将词表中的词映射到相应的索引。
            train_data: 训练数据集
            min_freq: 词频阈值，低于该阈值的词将被过滤掉
        """
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        """
        BucketIterator.splits:
            该类用于将数据集划分为多个batch，并在每个batch上进行迭代。
            它会自动将数据集按照长度进行排序，并将长度相同的句子放在同一个batch中。这么做能够避免由于句子长短不一而需要填补的需求（一个batch的输入样本之间的size要一样）
            它还可以将数据集划分为多个设备，并在每个设备上进行迭代。
            batch_size: 每个batch的大小
            device: 设备
        """
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator        