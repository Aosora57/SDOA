import torch
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim

        # 修改线性层的维度
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, n_filters, inner_dim)
        batch_size, n_filters, inner_dim = x.size()

        # 重塑输入以适应注意力机制
        x = x.transpose(1, 2)  # (batch_size, inner_dim, n_filters)
        x = x.reshape(batch_size, -1, self.input_dim)  # (batch_size, seq_len, input_dim)

        # 计算注意力
        query = self.query(x)  # (batch_size, seq_len, input_dim)
        key = self.key(x)  # (batch_size, seq_len, input_dim)
        value = self.value(x)  # (batch_size, seq_len, input_dim)

        # 重塑为多头形式
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # 重塑回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.input_dim)
        attn_output = self.out(attn_output)

        # 恢复原始形状
        attn_output = attn_output.view(batch_size, inner_dim, n_filters)
        attn_output = attn_output.transpose(1, 2)  # (batch_size, n_filters, inner_dim)

        return attn_output


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        # 1x1 卷积用于通道调整
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)
        return out


# SEBlock 是 Squeeze-and-Excitation Block 的缩写，用于增强卷积神经网络的特征表示能力。
# 它通过引入注意力机制来调整特征图的权重，从而提高网络的性能。
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class spectrumModule(nn.Module):
    def __init__(self, signal_dim=8, n_filters=32, n_layers=6, inner_dim=32, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        self.inner_dim = inner_dim

        # 确保 inner_dim 是 num_heads 的倍数
        self.inner_dim = (inner_dim // 4) * 4

        # 保持其他初始化代码不变
        self.in_layer = nn.Sequential(
            nn.Linear(2 * signal_dim, self.inner_dim * n_filters),
            nn.LayerNorm(self.inner_dim * n_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.blocks = nn.ModuleList([
            ImprovedResidualBlock(n_filters, n_filters) for _ in range(n_layers)
        ])

        self.attention = MultiHeadAttention(input_dim=self.inner_dim)

        self.out_layer = nn.Sequential(
            nn.Linear(self.inner_dim * n_filters, self.inner_dim * n_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.inner_dim * n_filters // 2, 2 * signal_dim)
        )

    def forward(self, inp):
        bsz = inp.size(0)

        # 输入处理
        x = self.in_layer(inp.view(bsz, -1))
        x = x.view(bsz, self.n_filters, self.inner_dim)

        # 残差块处理
        for block in self.blocks:
            x = block(x)

        # 注意力处理
        x = self.attention(x)

        # 使用 reshape 代替 view，并确保张量连续
        x = x.contiguous().reshape(bsz, -1)
        x = self.out_layer(x)

        return x

# 创建模型实例并生成可视化图 batch_size=64, n_layers=6, n_filters=2, kernel_size=3, inner_dim=32,
model = spectrumModule(signal_dim=16, n_filters=2, n_layers=6)
sample_input = torch.randn(64, 32)  # 假设批量大小为64，输入信号维度为 (64, 32)
output = model(sample_input)

# # 使用 torchviz 可视化模型结构
# dot_graph = make_dot(output.mean(), params=dict(model.named_parameters()))
# dot_graph.render("spectrum_module", format="png")  # 保存为 PNG 格式的图像文件

# 使用 torchviz 可视化模型结构
dot_graph = make_dot(output.mean(), params=dict(model.named_parameters()))

# 简化可视化：只显示主要模块而不显示所有参数
for node in dot_graph.body:
    if 'Linear' in node or 'Conv' in node:
         continue  # 忽略具体参数信息

dot_graph.render("spectrum_module_simplified", format="png")  # 保存为 PNG 格式的图像文件