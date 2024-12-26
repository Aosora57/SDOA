# SDOA-Net: An Efficient Deep Learning-Based DOA Estimation Network for Imperfect Array

The estimation of direction of arrival (DOA) is a crucial issue in conventional radar, wireless communication, and integrated sensing and communication (ISAC) systems. However, low-cost systems often suffer from imperfect factors, such as antenna position perturbations, mutual coupling effect, inconsistent gains/phases, and non-linear amplifier effect, which can significantly degrade the performance of DOA estimation. This paper proposes a DOA estimation method named super-resolution DOA network (SDOA-Net) based on deep learning (DL) to characterize the realistic array more accurately. Unlike existing DL-based DOA methods, SDOA-Net uses sampled received signals instead of covariance matrices as input to extract data features. Furthermore, SDOA-Net produces a vector that is independent of the DOA of the targets but can be used to estimate their spatial spectrum. Consequently, the same training network can be applied to any number of targets, reducing the complexity of implementation. The proposed SDOA-Net with a low-dimension network structure also converges faster than existing DL-based methods. The simulation results demonstrate that SDOA-Net outperforms existing DOA estimation methods for imperfect arrays.



方向到达（DOA）估计在传统雷达、无线通信和集成传感与通信（ISAC）系统中是一个关键问题。然而，低成本系统常常受到不完美因素的影响，例如天线位置扰动、互耦效应、不一致的增益/相位以及非线性放大器效应，这些因素会显著降低DOA估计的性能。本文提出了一种基于深度学习（DL）的DOA估计方法，称为超分辨DOA网络（SDOA-Net），以更准确地表征实际阵列。与现有的基于DL的DOA方法不同，SDOA-Net使用采样接收信号而不是协方差矩阵作为输入来提取数据特征。此外，SDOA-Net生成一个与目标的DOA无关的向量，但可以用于估计它们的空间谱。因此，相同的训练网络可以应用于任意数量的目标，从而减少实现的复杂性。所提出的具有低维网络结构的SDOA-Net收敛速度也比现有基于DL的方法快。仿真结果表明，SDOA-Net在不完美阵列上的性能优于现有的DOA估计方法。

ref_sp 是指参考空间谱（Reference Spatial Spectrum）。在信号处理和目标检测中，空间谱用于表示信号在空间中的分布情况。ref_sp 通常是通过对信号进行处理和计算得到的，用于与实际接收到的信号进行比较，以便确定信号的方向或位置


```markdown
Signal shape: torch.Size([64, 2, 16])
这表示生成的信号数据是一个三维张量，其中：
64：批量大小（batch size），表示一次处理的样本数量。
2：通道数（channels），可能表示不同的传感器或特征。
16：每个样本的特征长度（features），表示每个通道包含的特征数。
Reference Spectrum shape: torch.Size([64, 10000])
这表示参考谱（reference spectrum）数据也是一个二维张量，其中：
64：批量大小，与信号数据相同。
10000：每个样本的参考谱长度，可能表示在某个频率范围内的谱值数量。
DOA shape: torch.Size([64, 3])
这表示方向信息（Direction of Arrival, DOA）数据是一个二维张量，其中：
64：批量大小，与其他数据相同。
3：每个样本的方向特征数，可能表示在三维空间中接收到信号的方向。
单个样本输出
Single Signal Example:
text
tensor([[ 1.4142,  1.2440,  0.7745,  0.1185, -0.5660, -1.1142, -1.3944, -1.3389,
           -0.9612, -0.3522,  0.3416,  0.9531,  1.3353,  1.3961,  1.1210,  0.5760],
        [ 0.0000, -0.6726, -1.1833, -1.4092, -1.2960, -0.8709, -0.2362,  0.4554,
           1.0373,  1.3697,  1.3723,  1.0448,  0.4657, -0.2254, -0.8622, -1.2916]])
输出为一个形状为 (2, 16) 的张量，表示该样本在两个通道上的信号值。
每一行代表一个通道的数据，包含16个特征值。这些值可能代表在不同时间点或频率下测得的信号强度。
Single Reference Spectrum Example:
text
tensor([2.4046e-19, 2.4555e-19, 2.5075e-19, ..., 
        1.7240e-39, 1.6726e-39, 
        1.6228e-39])
输出为一个一维张量，形状为 (10000,)，表示该样本的参考谱值。
每个值代表在某个频率点上的谱强度，通常用于后续处理，例如信号重建或目标检测。
Single DOA Example:
text
tensor([-100.0000, -100.0000,   -9.0772])
输出为一个一维张量，形状为 (3,)，表示该样本的方向信息。
每个值可能代表不同方向上的到达角度（DOA），通常以度为单位。这里有三个值，可能对应于三维空间中的三个方向。
```
```python
spectrumModule(
  (in_layer): Sequential(
    (0): Linear(in_features=32, out_features=64, bias=True)
    (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (blocks): ModuleList(
    (0-5): 6 x ImprovedResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (se): SEBlock(
        (avg_pool): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=2, out_features=0, bias=False)
          (1): ReLU(inplace=True)
          (2): Linear(in_features=0, out_features=2, bias=False)
          (3): Sigmoid()
        )
      )
      (relu): ReLU(inplace=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (attention): MultiHeadAttention(
    (query): Linear(in_features=32, out_features=32, bias=True)
    (key): Linear(in_features=32, out_features=32, bias=True)
    (value): Linear(in_features=32, out_features=32, bias=True)
    (out): Linear(in_features=32, out_features=32, bias=True)
  )
  (out_layer): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=32, out_features=32, bias=True)
  )
)
```

循环填充（Cyclic Padding）是一种在卷积神经网络（CNN）中使用的填充方式，主要用于处理一维或二维数据。其特点是将输入数据的边缘部分进行循环连接，以保持数据的连续性和完整性。这种填充方式在处理具有周期性特征的数据时特别有效。

硬件环境：

    CPU 12th Gen Intel(R)Core(TM)i7-12700H(20 CPUs),~2.3GHz
    GPU NVIDIA GeForce RTX 3060 Laptop GPU
    内存 16384MB RAM
    操作系统 Windows 11 64-bit

软件环境：

    MATLAB R2021b
    Python                    3.8.8
    anyio                     4.6.2
    argon2-cffi               21.3.0
    argon2-cffi-bindings      21.2.0
    asttokens                 2.0.5
    async-lru                 2.0.4
    attrs                     24.2.0
    autocommand               2.2.2
    Babel                     2.11.0
    backcall                  0.2.0
    backports.tarfile         1.2.0
    beautifulsoup4            4.12.3
    bleach                    4.1.0
    blis                      0.7.11
    Bottleneck                1.4.2
    Brotli                    1.0.9
    catalogue                 2.0.10
    certifi                   2024.8.30
    cffi                      1.17.1
    charset-normalizer        3.3.2
    click                     7.1.2
    cloudpickle               3.1.0
    colorama                  0.4.6
    comm                      0.2.1
    contourpy                 1.2.0
    cycler                    0.11.0
    cymem                     2.0.8
    debugpy                   1.6.7
    decorator                 5.1.1
    defusedxml                0.7.1
    en_core_web_sm            3.0.0
    exceptiongroup            1.2.0
    executing                 0.8.3
    fastjsonschema            2.16.2
    filelock                  3.13.1
    fonttools                 4.51.0
    fsspec                    2024.12.0
    gmpy2                     2.1.2
    graphviz                  0.20.3
    gym                       0.26.1
    gym-notices               0.0.8
    h11                       0.14.0
    httpcore                  1.0.2
    httpx                     0.27.0
    idna                      3.7
    importlib-metadata        7.0.1
    importlib_resources       6.4.0
    inflect                   7.3.1
    ipykernel                 6.29.5
    ipython                   8.15.0
    ipywidgets                8.1.2
    jaraco.collections        5.1.0
    jaraco.context            5.3.0
    jaraco.functools          4.0.1
    jaraco.text               3.12.1
    jedi                      0.19.1
    Jinja2                    3.1.4
    joblib                    1.4.2
    json5                     0.9.25
    jsonschema                4.23.0
    jsonschema-specifications 2023.7.1
    jupyter                   1.0.0
    jupyter_client            8.6.0
    jupyter-console           6.6.3
    jupyter_core              5.7.2
    jupyter-events            0.10.0
    jupyter-lsp               2.2.0
    jupyter_server            2.14.1
    jupyter_server_terminals  0.4.4
    jupyterlab                4.2.5
    jupyterlab-pygments       0.1.2
    jupyterlab_server         2.27.3
    jupyterlab-widgets        3.0.10
    kiwisolver                1.4.4
    MarkupSafe                2.1.3
    matlabengineforpython     9.13
    matplotlib                3.9.2
    matplotlib-inline         0.1.6
    mistune                   2.0.4
    mkl_fft                   1.3.10
    mkl_random                1.2.7
    mkl-service               2.4.0
    more-itertools            10.3.0
    mpmath                    1.3.0
    murmurhash                1.0.10
    nbclient                  0.8.0
    nbconvert                 7.16.4
    nbformat                  5.10.4
    nest-asyncio              1.6.0
    networkx                  3.2.1
    notebook                  7.2.2
    notebook_shim             0.2.3
    numexpr                   2.10.1
    numpy                     1.26.4
    overrides                 7.4.0
    packaging                 24.1
    pandas                    2.2.2
    pandocfilters             1.5.0
    parso                     0.8.3
    pathlib_abc               0.1.1
    pathy                     0.11.0
    pickleshare               0.7.5
    pillow                    10.4.0
    pip                       24.2
    platformdirs              3.10.0
    ply                       3.11
    preshed                   3.0.9
    prometheus-client         0.14.1
    prompt-toolkit            3.0.43
    psutil                    5.9.0
    pure-eval                 0.2.2
    pycparser                 2.21
    pydantic                  1.7.4
    Pygments                  2.15.1
    pyparsing                 3.2.0
    PyQt5                     5.15.10
    PyQt5-sip                 12.13.0
    PySocks                   1.7.1
    python-dateutil           2.9.0.post0
    python-json-logger        2.0.7
    pytz                      2024.1
    pywin32                   305.1
    pywinpty                  2.0.10
    PyYAML                    6.0.2
    pyzmq                     25.1.2
    qtconsole                 5.6.0
    QtPy                      2.4.1
    referencing               0.30.2
    requests                  2.32.3
    rfc3339-validator         0.1.4
    rfc3986-validator         0.1.1
    rpds-py                   0.10.6
    scikit-learn              1.5.1
    scipy                     1.13.1
    seaborn                   0.13.2
    Send2Trash                1.8.2
    setuptools                75.1.0
    sip                       6.7.12
    six                       1.16.0
    smart-open                6.4.0
    sniffio                   1.3.0
    soupsieve                 2.5
    spacy                     3.0.0
    spacy-legacy              3.0.12
    srsly                     2.4.8
    stack-data                0.2.0
    sympy                     1.13.1
    terminado                 0.17.1
    thinc                     8.0.17
    threadpoolctl             3.5.0
    tinycss2                  1.2.1
    tomli                     2.0.1
    torch                     2.5.1
    torchaudio                2.5.1
    torchtext                 0.6.0
    torchvision               0.20.1
    torchviz                  0.0.3
    tornado                   6.4.1
    tqdm                      4.66.5
    traitlets                 5.14.3
    typeguard                 4.3.0
    typer                     0.3.2
    typing_extensions         4.11.0
    tzdata                    2023.3
    unicodedata2              15.1.0
    urllib3                   2.2.3
    wasabi                    0.10.1
    wcwidth                   0.2.5
    webencodings              0.5.1
    websocket-client          1.8.0
    wheel                     0.44.0
    widgetsnbextension        4.0.10
    win-inet-pton             1.1.0
    zipp                      3.20.2
