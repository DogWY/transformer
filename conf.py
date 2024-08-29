import torch

# GPU device setting
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

# model parameter setting
# 批处理大小，即每次训练使用的样本数量
batch_size = 128
# 输入序列的最大长度
max_len = 256
# 模型的维度，通常指嵌入层和前馈网络的维度
d_model = 512
# 模型中使用的层数，例如在Transformer模型中指编码器和解码器的层数
n_layers = 6
# 多头注意力机制中的头数
n_heads = 8
# 前馈网络的隐藏层维度
ffn_hidden = 2048
#  dropout概率，用于防止过拟合
drop_prob = 0.1

# optimizer parameter setting
# 初始学习率
init_lr = 1e-5
# 学习率衰减因子
factor = 0.9
# Adam优化器的epsilon参数，用于数值稳定性
adam_eps = 5e-9
# 早停机制中的耐心值，即在验证损失不再改善时等待的epoch数
patience = 10
# 学习率预热期，即在前几个epoch中逐渐增加学习率
warmup = 100
# 训练的总epoch数
epoch = 1000
# 梯度裁剪阈值，用于防止梯度爆炸
clip = 1.0
# 权重衰减，用于防止过拟合
weight_decay = 5e-4
# 表示无穷大
inf = float('inf')
