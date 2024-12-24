# SDOA-Net: An Efficient Deep Learning-Based DOA Estimation Network for Imperfect Array

The estimation of direction of arrival (DOA) is a crucial issue in conventional radar, wireless communication, and integrated sensing and communication (ISAC) systems. However, low-cost systems often suffer from imperfect factors, such as antenna position perturbations, mutual coupling effect, inconsistent gains/phases, and non-linear amplifier effect, which can significantly degrade the performance of DOA estimation. This paper proposes a DOA estimation method named super-resolution DOA network (SDOA-Net) based on deep learning (DL) to characterize the realistic array more accurately. Unlike existing DL-based DOA methods, SDOA-Net uses sampled received signals instead of covariance matrices as input to extract data features. Furthermore, SDOA-Net produces a vector that is independent of the DOA of the targets but can be used to estimate their spatial spectrum. Consequently, the same training network can be applied to any number of targets, reducing the complexity of implementation. The proposed SDOA-Net with a low-dimension network structure also converges faster than existing DL-based methods. The simulation results demonstrate that SDOA-Net outperforms existing DOA estimation methods for imperfect arrays.



方向到达（DOA）估计在传统雷达、无线通信和集成传感与通信（ISAC）系统中是一个关键问题。然而，低成本系统常常受到不完美因素的影响，例如天线位置扰动、互耦效应、不一致的增益/相位以及非线性放大器效应，这些因素会显著降低DOA估计的性能。本文提出了一种基于深度学习（DL）的DOA估计方法，称为超分辨DOA网络（SDOA-Net），以更准确地表征实际阵列。与现有的基于DL的DOA方法不同，SDOA-Net使用采样接收信号而不是协方差矩阵作为输入来提取数据特征。此外，SDOA-Net生成一个与目标的DOA无关的向量，但可以用于估计它们的空间谱。因此，相同的训练网络可以应用于任意数量的目标，从而减少实现的复杂性。所提出的具有低维网络结构的SDOA-Net收敛速度也比现有基于DL的方法快。仿真结果表明，SDOA-Net在不完美阵列上的性能优于现有的DOA估计方法。

ref_sp 是指参考空间谱（Reference Spatial Spectrum）。在信号处理和目标检测中，空间谱用于表示信号在空间中的分布情况。ref_sp 通常是通过对信号进行处理和计算得到的，用于与实际接收到的信号进行比较，以便确定信号的方向或位置


原网络结构 

```python
spectrumModule(
  (in_layer): Linear(in_features=32, out_features=64, bias=False)
  (mod): Sequential(
    (0): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (4): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (7): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (10): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
    (12): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (13): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (16): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): ReLU()
  )
  (out_layer): Linear(in_features=64, out_features=32, bias=False)
)
```

循环填充（Cyclic Padding）是一种在卷积神经网络（CNN）中使用的填充方式，主要用于处理一维或二维数据。其特点是将输入数据的边缘部分进行循环连接，以保持数据的连续性和完整性。这种填充方式在处理具有周期性特征的数据时特别有效。