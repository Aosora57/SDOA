import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(input_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: 输入特征，形状为 (batch_size, n_filters, seq_len)
        scores = torch.tanh(self.Wa(x))
        attention_weights = F.softmax(self.Ua(scores), dim=2)
        weighted_input = attention_weights * x
        output = weighted_input.sum(dim=2)  # 聚合加权输入
        return output


class SpectrumModule(nn.Module):
    def __init__(self, signal_dim=8, n_filters=2, n_layers=6, inner_dim=32, kernel_size=3):
        super().__init__()
        self.n_filters = n_filters
        dropout_rate = 0.5

        # 输入层
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)

        # 卷积层序列
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding='same', bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
        self.mod = nn.Sequential(*mod)

        # 注意力层，确保输入维度与卷积输出匹配
        self.attention = Attention(input_dim=n_filters, hidden_dim=16)

        # 输出层
        self.out_layer = nn.Linear(inner_dim * n_filters, 2 * signal_dim)

    def forward(self, inp):
        bsz = inp.size(0)

        # 输入层线性变换
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)

        # 卷积层序列
        x = self.mod(x)

        print(f"Input shape to attention: {x.shape}")

        # 注意力机制
        x = self.attention(x)

        print(f"Output shape after attention: {x.shape}")

        # 展平以适应输出层
        x = x.view(bsz, -1)  # 确保展平以适应线性层

        # 输出层线性变换，得到最终输出
        x = self.out_layer(x)

        return x


# 示例使用
if __name__ == "__main__":
    model = SpectrumModule(signal_dim=8, n_filters=2, n_layers=6, inner_dim=32)
    sample_input = torch.randn(64, 16)  # 假设批量大小为64，输入信号维度为16（即2*signal_dim）
    output = model(sample_input)
    print(output.shape)  # 输出的形状应为 (64, 16)，对应于 2*signal_dim
