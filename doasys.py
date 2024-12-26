import time
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 生成目标方向角 (DOA) 的函数。它在给定的范围内随机生成目标方向角，并确保它们之间的最小间隔
def gen_doa(target_num, doa, ant_num):
    doa_min = -45
    doa_max = 45
    # min_sep = 102.0 / ((ant_num-1)*super_ratio)
    min_sep = 102.0 / (ant_num - 1) * 2.0
    for n in range(target_num):
        condition = True
        doa_new = 0
        while condition:
            doa_new = np.random.rand() * (doa_max - doa_min) + doa_min
            condition = (np.min(np.abs(doa - doa_new)) < min_sep)
        doa[n] = doa_new
    # for n in range(target_num):
    #     doa_new = np.random.rand() * (doa_max - doa_min) + doa_min
    #     doa[n] = doa_new

# 生成方向矢量的函数。它根据给定的 DOA 角度、天线间距和天线数量生成方向矢量。
def steer_vec(doa_deg, d, ant_num, d_per):
    st = np.exp(1j * 2 * np.pi * (d * np.arange(ant_num).T + d_per) * np.sin(np.deg2rad(doa_deg)))
    return st


# 生成信号函数。它生成给定数量的数据样本，每个样本包含目标方向角、信号和目标数量。
# 信号包括相位和幅度的扰动以及互耦效应。
def gen_signal(data_num, args):
    target_num = np.random.randint(1, args.max_target_num + 1, data_num)
    doa = np.ones((data_num, args.max_target_num)) * np.inf
    s = np.zeros((data_num, 2, args.ant_num))
    for n in range(data_num):
        gen_doa(target_num[n], doa[n], args.ant_num)
        # perturbation
        d_per = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_per_std
        # phase and amplitude
        amp = np.ones(args.ant_num).T + np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_amp_std
        pha = np.random.randn(args.ant_num).T * np.random.rand(1) * args.max_phase_std
        amp_phase = amp * np.exp(1j * pha)
        for m in range(target_num[n]):
            st = amp_phase * steer_vec(doa[n, m], args.d, args.ant_num, d_per)
            s[n, 0] = s[n, 0] + st.real
            s[n, 1] = s[n, 1] + st.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
        s_comp = s[n, 0] + 1j * s[n, 1]
        # mutual coupling
        max_mc_power = np.power(args.max_mc, np.arange(args.ant_num))
        mc_mat = np.zeros((args.ant_num, args.ant_num), dtype=complex)
        for idx_ant1 in range(args.ant_num):
            for idx_ant2 in range(args.ant_num):
                if idx_ant1 == idx_ant2:
                    mc_mat[idx_ant1, idx_ant2] = 1
                else:
                    mc_power = np.random.rand(1) * max_mc_power[np.abs(idx_ant2 - idx_ant1)]
                    mc_mat[idx_ant1, idx_ant2] = np.sqrt(mc_power) * np.exp(1j * np.random.rand(1) * 2 * np.pi)
        s_comp_mc = np.matmul(mc_mat, s_comp)
        s[n, 0] = s_comp_mc.real
        s[n, 1] = s_comp_mc.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

        # non linear function
        if args.is_nonlinear:
            s[n] = np.tanh(args.nonlinear*s[n])
            s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))

    doa[doa == float('inf')] = -100
    doa = np.sort(doa, axis=1)
    return s.astype('float32'), doa.astype('float32'), target_num

# 生成参考空间谱的函数。它根据 DOA 和 DOA 网格生成参考空间谱。
def gen_refsp(doa, doa_grid, sigma):
    ref_sp = np.zeros((doa.shape[0], doa_grid.shape[0]))
    for i in range(doa.shape[1]):
        dist = np.abs(doa_grid[None, :] - doa[:, i][:, None])
        ref_sp += np.exp(- dist ** 2 / sigma ** 2)
    return ref_sp

# 为信号添加噪声的函数。它根据给定的信噪比 (SNR) 为信号添加噪声
def noise_torch(s, snr):
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    sigma_max = np.sqrt(1. / snr)
    sigmas = sigma_max * torch.rand(bsz, device=s.device, dtype=s.dtype)
    # sigmas = math.sqrt(1.0/snr)

    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    mult = sigmas * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    noise = noise * mult[:, None]
    return (s + noise).view(bsz, -1, signal_dim)

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(input_dim, hidden_dim)  # 用于计算注意力分数的线性层
        self.Ua = nn.Linear(hidden_dim, input_dim)  # 用于计算注意力权重的线性层
        self.output_layer = nn.Linear(input_dim, output_dim)  # 用于将输出映射到所需维度

    def forward(self, x):
        # x: 输入特征，形状为 (batch_size, n_filters, seq_len)
        batch_size, n_filters, seq_len = x.size()

        # 将 x 转换为 (batch_size * seq_len, n_filters) 以适应线性层
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * seq_len, n_filters)

        scores = torch.tanh(self.Wa(x_reshaped))
        attention_weights = F.softmax(self.Ua(scores), dim=1)  # 注意这里的 dim=1

        # 将 attention_weights 的形状调整回 (batch_size, seq_len, n_filters)
        attention_weights = attention_weights.view(batch_size, seq_len, n_filters)

        weighted_input = attention_weights * x.permute(0, 2, 1).contiguous()  # 确保维度匹配
        output = weighted_input.sum(dim=1)  # 聚合加权输入

        # print(f"Output shape after attention before final layer: {output.shape}")  # 应该是 (batch_size, n_filters)

        # 将输出映射到所需维度
        output = self.output_layer(output)  # 输出形状为 (batch_size, output_dim)

        # print(f"Output shape after attention: {output.shape}")

        return output

# MultiHeadAttention 是多头注意力机制的实现。它将输入张量分割成多个头，并计算每个头的注意力分数。
# 然后，它将这些注意力分数与值相乘，并合并结果。
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
        key = self.key(x)      # (batch_size, seq_len, input_dim)
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

# 定义了一个频谱模块的神经网络类。它包括输入层、多个卷积层和输出层。
# mod 变量是一个列表，其中包含了多个神经网络层。
# 它在 spectrumModule 和 DeepFreq 类的构造函数中被定义，用于存储卷积层、批标准化层和 ReLU 激活函数。
# 随后，mod 被传递给 nn.Sequential 创建一个顺序容器，存储在 self.mod 属性中，用于定义模型的中间层序列

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入和输出通道数不匹配，使用卷积调整输入形状
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x  # 保存输入以便后续相加
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

# ImprovedResidualBlock 是 ResidualBlock 的改进版本，引入了 SEBlock 来增强特征表示能力。
# 它通过引入注意力机制来调整特征图的权重，从而提高网络的性能。
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
    def __init__(self, signal_dim=8, n_filters=32, n_layers=6, inner_dim=32,kernel_size=3):
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

# 定义了一个深度频率模块的神经网络类。它包括输入层、多个卷积层和一个转置卷积层。
class DeepFreq(nn.Module):
    def __init__(self, signal_dim=8, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    # padding = (kernel_out - upsampling + 1) // 2
    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x

# 空间谱中提取 DOA 的函数。它使用峰值检测算法从空间谱中提取 DOA。
def get_doa(sp, doa_num, doa_grid, max_target_num, ref_doa):
    est_doa = -100 * np.ones((doa_num.shape[0], max_target_num))
    for n in range(len(doa_num)):
        find_peaks_out = scipy.signal.find_peaks(sp[n], height=(None, None))
        num_spikes = min(len(find_peaks_out[0]), int(doa_num[n]))
        idx = np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:]
        est_doa[n, :num_spikes] = np.sort(doa_grid[find_peaks_out[0][idx]])
        tmp = est_doa[n].copy()
        for idx_tmp in range(len(ref_doa[n])):
            est_doa[n, idx_tmp] = tmp[np.argmin(np.abs(ref_doa[n, idx_tmp] - tmp))]

    # est_doa = np.sort(est_doa, axis=1)
    return est_doa

# 训练神经网络的函数。它包括训练和验证过程，并计算训练和验证损失
def train_net(args, net, optimizer, criterion, train_loader, val_loader,
              doa_grid, epoch, train_num, train_type, net_type):
    epoch_start_time = time.time()
    net.train()
    loss_train = 0
    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    if net_type == 0:
        for n in range(doa_grid.size):
            tmp = steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
            dic_mat[n, 0] = tmp.real
            dic_mat[n, 1] = tmp.imag
        dic_mat_torch = torch.from_numpy(dic_mat).float()
        if args.use_cuda:
            dic_mat_torch = dic_mat_torch.cuda()
    '''
        数据加载: 从 train_loader 中获取一个批次的数据，包括干净信号、目标谱和方向信息。
        添加噪声: 使用 noise_torch 函数向干净信号添加噪声以生成带噪信号。
        清除梯度: 在每个批次开始时清除之前计算的梯度，以准备新的反向传播。
    '''
    for batch_idx, (clean_signal, target_sp, doa) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_sp = clean_signal.cuda(), target_sp.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr)#向干净信号添加噪声以生成带噪信号
        optimizer.zero_grad()
        # print(f"input shape noisy_signal: {noisy_signal.shape}")
        # print(f"input shape batch_size: {args.batch_size}")
        # 前向传播: 将带噪信号传递给网络，得到输出并重塑为(batch_size, 2, -1)的形状。这里2是特征数量，后面的维度会根据实际情况自动推断。
        output_net = net(noisy_signal)
        output_net = output_net.reshape(args.batch_size, 2, -1)  # 使用 reshape 代替 view
        # 打印输出形状
        # print(f"Output shape before view: {output_net.shape}")
        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
            # loss = criterion(torch.pow(mm_real, 2) + torch.pow(mm_imag, 2), target_sp)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)

        loss.backward()
        optimizer.step()
        loss_train += loss.data.item()

        # plt.figure()
        # plt.plot(sp.cpu().detach().numpy()[0])
        # plt.plot(target_sp.cpu().detach().numpy()[0])
        # plt.show()

    net.eval()   # 设置网络为评估模式
    loss_val, fnr_val = 0, 0

    for batch_idx, (noisy_signal, _, target_sp, doa) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_sp = noisy_signal.cuda(), target_sp.cuda()
        with torch.no_grad():
            output_net = net(noisy_signal)
            output_net = output_net.reshape(args.batch_size, 2, -1)  # 使用 reshape 代替 view

        if net_type == 0:
            mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 1, :].T)
            mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                         dic_mat_torch[:, 0, :].T)
        else:
            mm_real = output_net[:, 0, :]
            mm_imag = output_net[:, 1, :]
        sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
        loss = criterion(sp, target_sp)
        loss_val += loss.data.item()

        doa_num = (doa >= -90).sum(dim=1)
        est_doa = get_doa(sp.cpu().detach().numpy(), doa_num, doa_grid, args.max_target_num, doa)

    loss_train /= args.n_training
    loss_val /= args.n_validation

    print("TTrain_Num: %d, rain_Type: %d, Epochs: %d / %d, Time: %.1f, Training Loss: %.2f, Validation Loss:  %.2f" % (
        train_num,
        train_type,
        epoch, args.n_epochs,
        time.time() - epoch_start_time,
        loss_train,
        loss_val))
    # print(np.sort(doa[0]))
    # print(np.sort(est_doa[0]))
    return net, loss_train, loss_val
