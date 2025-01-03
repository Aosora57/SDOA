import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import doasys
import os

if __name__ == '__main__':
    # 解析命令行参数：使用argparse 模块解析命令行参数，设置训练和验证数据的数量、网络结构参数、天线参数、噪声参数等
    # parser.add_argument 函数用于向命令行参数解析器添加参数。它定义了程序可以接受的命令行参数及其属性。以下是每个参数的解释：
    parser = argparse.ArgumentParser()

    # parser.add_argument('--numpy_seed', type=int, default=222)
    # parser.add_argument('--torch_seed', type=int, default=333)

    parser.add_argument('--n_training', type=int, default=8000, help='# of training data')  # 8000
    parser.add_argument('--n_validation', type=int, default=64, help='# of validation data')

    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=float, default=100, help='the size of grids')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')#64

    # module parameters
    parser.add_argument('--n_layers', type=int, default=6, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=2, help='number of filters per layer in the module')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs used to train the module')  # 100

    # array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    # parser.add_argument('--super_ratio', type=float, default=1, help='super-resolution ratio based on 102/(ant_num-1)')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--snr', type=float, default=1., help='the maximum SNR')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # imperfect parameters
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    # training policy
    parser.add_argument('--new_train', type=int, default=0, help='train a new network')
    parser.add_argument('--train_num', type=int, default=80, help='train a new network')
    parser.add_argument('--net_type', type=int, default=0, help='the type of network')

    args = parser.parse_args()

    # 设置设备：根据是否有可用的 GPU，设置 use_cuda 标志
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    # np.random.seed(args.numpy_seed)
    # torch.manual_seed(args.torch_seed)

    doa_grid = np.linspace(-50, 50, args.grid_size, endpoint=False)
    # ref_grid = np.linspace(-50, 50, 16, endpoint=False)
    ref_grid = doa_grid

    # if args.new_train:
    if True:
        if args.net_type == 0:
            net = doasys.spectrumModule(signal_dim=args.ant_num, n_filters=args.n_filters, inner_dim=args.inner_dim,
                                        n_layers=args.n_layers, kernel_size=args.kernel_size)
        else:
            net = doasys.DeepFreq(signal_dim=args.ant_num, n_filters=args.n_filters, inner_dim=args.inner_dim,
                                  n_layers=args.n_layers, kernel_size=args.kernel_size,
                                  upsampling=int(args.grid_size / args.ant_num),
                                  kernel_out=int(args.grid_size / args.ant_num + 3))
    else:
        # if args.net_type == 0:
        #     net = torch.load(('net_layer%d.pkl' % args.n_layers))
        # else:
        #     net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers))

        if args.use_cuda:
            if args.net_type == 0:
                net = torch.load('net.pkl')
            else:
                net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers))

            # net = torch.load('net_layer2.pkl')
        else:
            if args.net_type == 0:
                net = torch.load('net.pkl', map_location=torch.device('cpu'))
            else:
                net = torch.load(('deepfreq_layer%d.pkl' % args.n_layers), map_location=torch.device('cpu'))

            # net = torch.load('net_layer2.pkl', map_location=torch.device('cpu'))

    if args.use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)

    max_per_std = args.max_per_std
    max_amp_std = args.max_amp_std
    max_phase_std = args.max_phase_std
    max_mc = args.max_mc
    nonlinear = args.nonlinear
    is_nonlinear = args.is_nonlinear


    '''
    train_type == 0：所有参数设为0，即无位置扰动、无幅度扰动、无相位扰动、无互耦、无非线性效应。
    train_type == 1：只有位置扰动，其他参数设为0。
    train_type == 2：只有幅度扰动，其他参数设为0。
    train_type == 3：只有相位扰动，其他参数设为0。
    train_type == 4：只有互耦，其他参数设为0。
    train_type == 5：只有非线性效应，其他参数设为0。
    train_type == 6：所有参数都使用最大值，即包含位置扰动、幅度扰动、相位扰动、互耦和非线性效应
    
    每个 train_num 是对同一个 net 进行更新。代码中定义的 net 对象在整个训练过程中保持不变，并且在每个训练周期（epoch）中都会对其进行更新。
    '''
    for idx in range(args.train_num):
        for train_type in range(7):
            if train_type == 0:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 1:
                args.max_per_std = max_per_std
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 2:
                args.max_per_std = 0
                args.max_amp_std = max_amp_std
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 3:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = max_phase_std
                args.max_mc = 0
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 4:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = max_mc
                args.nonlinear = 0
                args.is_nonlinear = 0
            elif train_type == 5:
                args.max_per_std = 0
                args.max_amp_std = 0
                args.max_phase_std = 0
                args.max_mc = 0
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear
            else:
                args.max_per_std = max_per_std
                args.max_amp_std = max_amp_std
                args.max_phase_std = max_phase_std
                args.max_mc = max_mc
                args.nonlinear = nonlinear
                args.is_nonlinear = is_nonlinear

            # generate the training data 生产训练数据部分
            signal, doa, target_num = doasys.gen_signal(args.n_training, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            doa = torch.from_numpy(doa).float()
            ref_sp = torch.from_numpy(ref_sp).float()
            dataset = data_utils.TensorDataset(signal, ref_sp, doa)
            train_loader = data_utils.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            # 获取 train_loader 中的一个批次的数据
            # for batch in train_loader:
            #     signal_batch, ref_sp_batch, doa_batch = batch  # 解包批次数据
            #     print("Signal shape:", signal_batch.shape)  # 打印信号的形状
            #     print("Reference Spectrum shape:", ref_sp_batch.shape)  # 打印参考谱的形状
            #     print("DOA shape:", doa_batch.shape)  # 打印方向信息的形状
            #
            #     # 输出一条数据
            #     print("Single Signal Example:", signal_batch[0])  # 输出第一个样本的信号
            #     print("Single Reference Spectrum Example:", ref_sp_batch[0])  # 输出第一个样本的参考谱
            #     print("Single DOA Example:", doa_batch[0])  # 输出第一个样本的方向信息
            #     break  # 只获取第一条数据，之后退出循环

            # Save the training data to a file
            # torch.save({'signal': signal, 'doa': doa, 'ref_sp': ref_sp}, 'training_data.pt')

            # generate the validation data
            signal, doa, target_num = doasys.gen_signal(args.n_validation, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            doa = torch.from_numpy(doa).float()
            ref_sp = torch.from_numpy(ref_sp).float()
            noisy_signals = doasys.noise_torch(signal, args.snr)
            dataset = data_utils.TensorDataset(noisy_signals, signal, ref_sp, doa)
            val_loader = data_utils.DataLoader(dataset, batch_size=args.batch_size)

            # Save the validation data to a file
            # torch.save({'noisy_signals': noisy_signals, 'signal': signal, 'doa': doa, 'ref_sp': ref_sp},'validation_data.pt')

            start_epoch = 1
            criterion = torch.nn.MSELoss(reduction='sum')
            loss_train = np.zeros((args.n_epochs, 1))
            loss_val = np.zeros((args.n_epochs, 1))
            print("args", args)
            for epoch in range(start_epoch, args.n_epochs):
                net, loss_train[epoch - 1], loss_val[epoch - 1] = doasys.train_net(args=args, net=net,
                                                                                   optimizer=optimizer,
                                                                                   criterion=criterion,
                                                                                   train_loader=train_loader,
                                                                                   val_loader=val_loader,
                                                                                   doa_grid=doa_grid, epoch=epoch,
                                                                                   train_num=idx, train_type=train_type,
                                                                                   net_type=args.net_type)
                # if (epoch % 10 == 0):
                #     plt.figure()
                #     plt.semilogy(loss_train[0:epoch-1])
                #     plt.semilogy(loss_val[0:epoch-1])
                #     plt.show()
            if args.net_type == 0:
                # 创建保存目录
                save_dir = 'checkpoints'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # 构建包含 train_num 的文件名
                loss_filename = os.path.join(save_dir, f'loss_attention_resnet_train_{idx}.npz')
                net_filename = os.path.join(save_dir, f'net_attention_resnet_train_{idx}.pkl')
                
                # 保存损失和网络模型
                np.savez(loss_filename, loss_train=loss_train, loss_val=loss_val)
                torch.save(net, net_filename)
                
                print(f"Saved model and loss for train_num {args.train_num}")
                print(f"Loss file: {loss_filename}")
                print(f"Model file: {net_filename}")
            else:
                np.savez(('deepfreq_loss_layer%d.npz' % args.n_layers), loss_train, loss_val)
                torch.save(net, ('deepfreq__layer%d.pkl' % args.n_layers))
