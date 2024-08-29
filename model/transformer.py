import re
import torch
from torch import nn

from model.decoder import Decoder
from model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device) -> None:
        """
        Args:
            src_pad_idx (_type_): 源序列谈冲索引。在处理变长序列时，通常会用一个特殊的标记来填充短序列，使其长度一致
            trg_pad_idx (_type_): 目标序列的填充索引，与src_pad_idx类似
            trg_sos_idx (_type_): 目标序列的“序列开始”标记索引。在生成目标序列时，通常会在序列开头添加一个特殊的标记来表示序列的开始
            enc_voc_size (_type_): 编码器词汇表大小，编码器输入的词汇表中不同单词的数量
            dec_voc_size (_type_): 解码器词汇表大小，解码器输入的词汇表中不同单词的数量
            d_model (_type_): 模型中嵌入向量和前馈网络的维度
            n_head (_type_): 多头注意力的头数
            max_len (_type_): 最大序列长度，模型能够处理的最大序列长度
            ffn_hidden (_type_): 前馈网络的隐藏层维度
            n_layers (_type_): 编码器和解码器的层数
            drop_prob (_type_): dropout概率
            device (_type_): 设备
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)
        return dec_output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask