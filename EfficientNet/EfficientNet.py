# EfficientNet.py
import os
import zipfile
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import math

# =========================
# EfficientNet网络结构实现
# =========================

class Swish(nn.Module):
    # Swish激活函数
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            Swish(),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block，EfficientNet的主力模块"""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_channels == out_channels and stride == 1
        # 扩展阶段
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        # 深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se_block = SEBlock(expanded_channels, r=expanded_channels//se_channels)
        else:
            self.se_block = nn.Identity()
        # 输出阶段
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def _drop_connect(self, x):
        # DropConnect正则化
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor
    def forward(self, x):
        identity = x
        # 扩展
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        # 深度卷积
        x = self.depthwise_conv(x)
        # SE注意力
        x = self.se_block(x)
        # 投影
        x = self.project_conv(x)
        # 残差连接
        if self.use_residual:
            x = identity + self._drop_connect(x)
        return x

class EfficientNet(nn.Module):
    """EfficientNet主网络结构"""
    # EfficientNet-B0/B1配置参数
    # 格式: (宽度系数, 深度系数, 输入分辨率, dropout率)
    model_configs = {
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
    }
    # EfficientNet基础结构配置
    # 格式: (扩展率, 卷积核, 步长, 输入通道, 输出通道, 层数, SE比例)
    base_architecture = [
        (1, 3, 1, 32, 16, 1, 0.25),    # MBConv1, 3x3
        (6, 3, 2, 16, 24, 2, 0.25),    # MBConv6, 3x3
        (6, 5, 2, 24, 40, 2, 0.25),    # MBConv6, 5x5
        (6, 3, 2, 40, 80, 3, 0.25),    # MBConv6, 3x3
        (6, 5, 1, 80, 112, 3, 0.25),   # MBConv6, 5x5
        (6, 5, 2, 112, 192, 4, 0.25),  # MBConv6, 5x5
        (6, 3, 1, 192, 320, 1, 0.25)   # MBConv6, 3x3
    ]
    def __init__(self, version='b0', num_classes=1000, dropout_rate=None):
        super().__init__()
        # 获取模型配置
        width_factor, depth_factor, resolution, default_dropout = self.model_configs[version]
        dropout_rate = dropout_rate or default_dropout
        # Stem卷积层
        out_channels = self._round_filters(32, width_factor)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        # 构建主干block
        self.blocks = nn.Sequential()
        in_channels = out_channels
        for i, (expand_ratio, kernel_size, stride, _, filters, num_layers, se_ratio) in enumerate(self.base_architecture):
            out_channels = self._round_filters(filters, width_factor)
            num_layers = self._round_repeats(num_layers, depth_factor)
            block_sequence = []
            for j in range(num_layers):
                block_stride = stride if j == 0 else 1
                block_in_channels = in_channels if j == 0 else out_channels
                block_sequence.append(
                    MBConvBlock(
                        block_in_channels, out_channels,
                        expand_ratio, kernel_size, block_stride, se_ratio
                    )
                )
            self.blocks.add_module(f'block{i+1}', nn.Sequential(*block_sequence))
            in_channels = out_channels
        # Head输出层
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self._round_filters(1280, width_factor), 1, bias=False),
            nn.BatchNorm2d(self._round_filters(1280, width_factor)),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self._round_filters(1280, width_factor), num_classes)
        )
    def _round_filters(self, filters, width_factor):
        """根据宽度系数调整通道数"""
        filters *= width_factor
        new_filters = max(8, int(filters + 4) // 8 * 8)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)
    def _round_repeats(self, repeats, depth_factor):
        """根据深度系数调整重复次数"""
        return int(math.ceil(depth_factor * repeats))
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# 工厂函数，快速创建EfficientNet模型
# 创建EfficientNet-B0
def efficientnet_b0(num_classes=1000):
    return EfficientNet(version='b0', num_classes=num_classes)
# 创建EfficientNet-B1
def efficientnet_b1(num_classes=1000):
    return EfficientNet(version='b1', num_classes=num_classes)

# =========================
# 数据集处理与增强部分
# =========================
# 解压数据集
# zip_pattern: 压缩包通配符，data_dir: 解压目标文件夹
# 用于首次运行时自动解压数据集

def extract_dataset(zip_pattern="*.zip", data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    zip_files = glob.glob(zip_pattern)
    if not zip_files:
        raise FileNotFoundError("No zip files found in current directory")
    print(f"Extracting {zip_files[0]} to {data_dir}...")
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete!")

# 自定义数据集，支持transform和异常图片跳过
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = datasets.folder.default_loader(img_path)
            if self.transform:
                image = self.transform(image)
            return image, label
        except (OSError, Exception) as e:
            print(f"跳过损坏的图像: {img_path}, 错误: {e}")
            return None

# DataLoader的collate函数，跳过None样本
# 用于过滤掉损坏图片

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# DataLoader的worker初始化函数，保证多进程随机性

def worker_init_fn(worker_id, seed=42):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# worker初始化类，便于传递seed
class WorkerInitializer:
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)

# 数据准备函数，包含数据增强、随机划分训练/验证/测试集
# 返回DataLoader和类别信息

def prepare_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42):
    # 设置随机种子，保证可复现
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    # 验证集增强
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 测试集增强
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 先加载全部样本
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    all_samples = [(full_dataset.imgs[i][0], full_dataset.targets[i]) for i in range(len(full_dataset))]
    random.shuffle(all_samples)
    train_size = int(train_ratio * len(all_samples))
    val_size = int(val_ratio * len(all_samples))
    test_size = len(all_samples) - train_size - val_size
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    train_dataset = CustomDataset(train_samples, transform=train_transform)
    val_dataset = CustomDataset(val_samples, transform=val_transform)
    test_dataset = CustomDataset(test_samples, transform=test_transform)
    worker_init = WorkerInitializer(seed)
    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn_skip_none
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init,
        collate_fn=collate_fn_skip_none
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init,
        collate_fn=collate_fn_skip_none
    )
    print(f"Dataset loaded with {len(train_dataset)} training, {len(val_dataset)} validation, "
          f"and {len(test_dataset)} testing samples")
    print(f"Classes: {class_names}")
    print("Data properly randomized before any splitting occurred")
    return train_loader, val_loader, test_loader, num_classes, class_names

# =========================
# 训练与评估部分
# =========================
# 训练函数，带有验证集监控和早停

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30, patience=10):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        # 更新学习率
        scheduler.step()
        # 打印训练与验证信息
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        # 早停与模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_loss_model.pth')
            print(f"Saved best loss model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    # 绘制训练与验证曲线
    plot_metrics_with_validation(train_losses, train_accuracies, val_losses, val_accuracies)
    return train_losses, train_accuracies, val_losses, val_accuracies

# 测试集评估函数

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 绘制训练/验证损失与准确率曲线

def plot_metrics_with_validation(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_validation_metrics.png')
    plt.show()

# =========================
# 模型创建与主程序入口
# =========================
# 创建EfficientNet模型，支持冻结特征层

def create_efficient_net(model_variant, num_classes, freeze_features=False):
    if model_variant == 'b0':
        model = efficientnet_b0(num_classes=num_classes)
    else:
        model = efficientnet_b1(num_classes=num_classes)
    if freeze_features:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    return model

# 主程序入口
# 包含数据解压、数据准备、模型训练、评估与保存

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    extract_dataset()
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_data(
        "data",
        batch_size=48,
        train_ratio=0.7,
        val_ratio=0.15
    )
    model_variant = 'b0'
    model = create_efficient_net(model_variant, num_classes, freeze_features=False)
    model = model.to(device)
    print(f"Using EfficientNet-{model_variant.upper()} architecture")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=128,
        eta_min=1e-5
    )
    print("Starting training with validation monitoring...")
    train_with_validation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs=128, patience=12
    )
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    print("Evaluating model on test data...")
    test_accuracy = evaluate_model(model, test_loader, device)
    torch.save(model.state_dict(), f'efficientnet_{model_variant}_final.pth')
    print(f"Model saved with test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()