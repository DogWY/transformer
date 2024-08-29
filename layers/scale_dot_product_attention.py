import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    点积缩放注意力机制
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 多个样本(batch)和多头注意力下，反转2、3维度等效于对k_i进行转制
        k_t = k.transpose(2, 3) 
        # @ 表示矩阵乘法
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            """
            对score中的元素应用mask，传入一个callable，masked_fill自动对每个元素应用callable，如果callable返回值为True，则将值填充为传入的指定值
            """
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        v = score @ v

        return v, score
