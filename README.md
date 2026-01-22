# 文物种类识别项目

本项目旨在使用多种深度学习模型对文物种类进行自动识别。项目中实现了三种不同的网络架构，分别探索了不同的特征提取和分类策略。

## 支持的模型

本项目包含以下三种深度学习模型的实现：

1.  **CNN-Transformer**: 结合了卷积神经网络（CNN）提取局部特征的能力和 Transformer 捕捉全局依赖关系的能力。
2.  **EfficientNet**: 使用了 MBConvBlock 和 SEBlock (Squeeze-and-Excitation) 的高效网络结构，旨在实现更好的性能与计算效率平衡。
3.  **ResNet152**: 一个深度的残差网络，使用了自定义的 Bottleneck 结构，通过残差连接解决了深层网络的退化问题。

## 目录结构

`\
.
 Notice.txt                  # 简要注意事项
 模型使用/                   # 存放训练好的模型权重文件 (.pth)
    efficientnet_b0_final.pth
    enhanced_cnn_transformer.pth
    resnet152_model.pth
 CNN-Transformer/            # CNN-Transformer 模型相关代码
    CNN-Transformer .py     # 主训练/推理脚本
    external.py             # 依赖安装和环境配置脚本
 EfficientNet/               # EfficientNet 模型相关代码
    EfficientNet.py         # 主训练/推理脚本
    external.py             # 依赖安装和环境配置脚本
 ResNet152/                  # ResNet152 模型相关代码
     ResNet152.py            # 主训练/推理脚本
     external.py             # 依赖安装和环境配置脚本
`\

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

请确保将数据集文件（通常是图片文件夹或压缩包）放置在与你要运行的模型脚本（\".py\" 文件）**相同的目录**下。

### 2. 环境配置

每个模型文件夹下都有一个 \"external.py\" 脚本。在运行主模型脚本之前，建议先运行该脚本以自动检查并安装所需的依赖库。

`ash\
# 以 EfficientNet 为例\
cd EfficientNet\
python external.py\
`\

### 3. 运行模型

配置好环境并放置好数据集后，即可运行相应的模型脚本进行训练或推理。

`ash\
# 运行 EfficientNet\
python EfficientNet.py\
\
# 运行 CNN-Transformer\
cd ../CNN-Transformer\
python \"CNN-Transformer .py\"\
\
# 运行 ResNet152\
cd ../ResNet152\
python ResNet152.py\
`\

### 4. 使用预训练模型

训练好的模型权重文件保存在 \"模型使用/\" 目录下。你可以在代码中加载这些权重文件进行推理或继续微调。

> **注意**：\
> 1. 为了减小项目体积，本项目仅上传了 \"efficientnet_b0_final.pth\" 作为示例模型权重。\
> 2. \"Antique.zip\" 等数据集文件未包含在本项目中，请自行准备数据集。

### 5. GUI 软件使用

我们在 \"模型使用/\" 目录下提供了一个可视化的文物种类识别软件 \"gui.exe\"。\
用户可以直接运行该程序，程序会加载同目录下的权重文件（如 \"efficientnet_b0_final.pth\"），你可以通过界面上传图片进行快速识别。

---
**注意**: 在运行脚本前，请查阅 \"Notice.txt\" 获取最新的简要提示。
