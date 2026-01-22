# 文物种类识别项目

本项目旨在使用多种深度学习模型对文物种类进行自动识别。项目中实现了三种不同的网络架构，分别探索了不同的特征提取和分类策略。

## 支持的模型

本项目包含以下三种深度学习模型的实现：

1.  **CNN-Transformer**: 结合了卷积神经网络（CNN）提取局部特征的能力和 Transformer 捕捉全局依赖关系的能力。
2.  **EfficientNet**: 使用了 MBConvBlock 和 SEBlock (Squeeze-and-Excitation) 的高效网络结构，旨在实现更好的性能与计算效率平衡。
3.  **ResNet152**: 一个深度的残差网络，使用了自定义的 Bottleneck 结构，通过残差连接解决了深层网络的退化问题。

## 目录结构

```text
.
├── 模型使用/                   # 存放模型权重和 GUI 软件
│   ├── efficientnet_b0_final.pth  # 示例模型权重
│   ├── gui.exe                    # 可视化识别软件
│   └── (其他 .pth 文件需本地训练生成)
├── CNN-Transformer/            # CNN-Transformer 模型相关代码
│   ├── CNN-Transformer .py     # 主训练/推理脚本
│   └── external.py             # 依赖安装和环境配置脚本
├── EfficientNet/               # EfficientNet 模型相关代码
│   ├── EfficientNet.py         # 主训练/推理脚本
│   └── external.py             # 依赖安装和环境配置脚本
└── ResNet152/                  # ResNet152 模型相关代码
    ├── ResNet152.py            # 主训练/推理脚本
    └── external.py             # 依赖安装和环境配置脚本
```

## 环境要求

本项目主要依赖以下 Python 库：

*   Python 3.x
*   torch (PyTorch)
*   torchvision
*   numpy
*   matplotlib
*   scikit-learn

## 使用说明

### 1. 准备数据集

请确保将数据集文件（通常是图片文件夹或压缩包）放置在与你要运行的模型脚本（`.py` 文件）**相同的目录**下。

### 2. 环境配置

每个模型文件夹下都有一个 `external.py` 脚本。在运行主模型脚本之前，建议先运行该脚本以自动检查并安装所需的依赖库。

```bash
# 以 EfficientNet 为例
cd EfficientNet
python external.py
```

### 3. 运行模型

配置好环境并放置好数据集后，即可运行相应的模型脚本进行训练或推理。

```bash
# 运行 EfficientNet
python EfficientNet.py

# 运行 CNN-Transformer
cd ../CNN-Transformer
python "CNN-Transformer .py"

# 运行 ResNet152
cd ../ResNet152
python ResNet152.py
```

### 4. 使用预训练模型

训练好的模型权重文件保存在 `模型使用/` 目录下。你可以在代码中加载这些权重文件进行推理或继续微调。

> **注意**：
> 1. 为了减小项目体积，本项目 Github 仓库仅上传了 `efficientnet_b0_final.pth` 作为示例模型权重。
> 2. `Antique.zip` 等数据集文件未包含在本项目中，请自行准备数据集。

### 5. 可视化识别软件 (GUI)

本项目提供了一个无需配置环境即可使用的可视化识别工具，方便用户快速体验模型效果。

*   **软件位置**: `模型使用/gui.exe`
*   **功能说明**: 该软件封装了训练好的深度学习模型，用户只需选择一张文物图片，即可获得识别结果。
*   **使用方法**:
    1. 确保 `gui.exe` 与模型权重文件（如 `efficientnet_b0_final.pth`）在同一目录下。
    2. 双击运行 `gui.exe`。
    3. 在软件界面中点击“选择图片”按钮，上传本地图片。
    4. 软件将自动调用模型进行推理，并显示预测的文物种类。
