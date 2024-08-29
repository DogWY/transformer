from torch import nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        # padding_idx：填充符号在vocab中的索引
        # nn.Embedding可以理解为一个可学习的映射/字典，将idx映射到d_model维度的向量
