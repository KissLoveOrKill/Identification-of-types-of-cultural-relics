# CNN-Transformer.py
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


# Residual block for deeper CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


# Enhanced CNN-Transformer Model
class CNNTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=512, depth=6, heads=12, mlp_dim=1024, dropout=0.2):
        super(CNNTransformer, self).__init__()

        # Deeper CNN Feature Extractor with proper channel dimensions
        self.feature_extractor = nn.Sequential(
            # Initial conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 1
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            # Block 4 (new) - IMPORTANT: make sure embed_dim matches properly
            ResidualBlock(512, embed_dim),
            ResidualBlock(embed_dim, embed_dim),  # Fixed input channels
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Rest of the implementation remains the same
        self.seq_length = 7 * 7
        self.embedding = nn.Linear(embed_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features with CNN
        x = self.feature_extractor(x)

        # Reshape to sequence
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, -1, x.shape[-1])

        # Project embedding
        x = self.embedding(x)

        # Add positional embedding
        x = x + self.pos_embedding

        # Layer normalization
        x = self.norm(x)

        # Apply transformer
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classify
        x = self.mlp_head(x)

        return x


# Extract dataset zip file
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

# Define this class at module level (outside any function)
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

# Define collate function at module level
def collate_fn_skip_none(batch):
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# Define worker initialization function at module level
def worker_init_fn(worker_id, seed=42):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# Add this class at module level
class WorkerInitializer:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)

# Prepare data with enhanced randomization and validation split
def prepare_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Define transforms separately
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

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset with no transform first
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Get all samples and labels
    all_samples = [(full_dataset.imgs[i][0], full_dataset.targets[i]) for i in range(len(full_dataset))]

    # Shuffle samples before splitting
    random.shuffle(all_samples)

    # Calculate split sizes
    train_size = int(train_ratio * len(all_samples))
    val_size = int(val_ratio * len(all_samples))
    test_size = len(all_samples) - train_size - val_size

    # Split samples into train, val and test sets
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]

    # Create datasets with appropriate transforms
    train_dataset = CustomDataset(train_samples, transform=train_transform)
    val_dataset = CustomDataset(val_samples, transform=val_transform)
    test_dataset = CustomDataset(test_samples, transform=test_transform)

    # Create data loaders
    worker_init = WorkerInitializer(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init,  # Use class instance instead of lambda
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn_skip_none
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_skip_none
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_skip_none
    )

    print(f"Dataset loaded with {len(train_dataset)} training, {len(val_dataset)} validation, "
          f"and {len(test_dataset)} testing samples")
    print(f"Classes: {class_names}")
    print("Data properly randomized before any splitting occurred")

    return train_loader, val_loader, test_loader, num_classes, class_names


# Enhanced training function with validation monitoring
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=25,
                          patience=5):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
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

            # Gradient clipping
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

        # Validation phase
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

        # Update learning rate
        scheduler.step()

        # Print stats
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping and model saving logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_loss_model.pth')
            print(f"Saved best loss model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                # Load best model - fixed without invalid parameter
                model.load_state_dict(torch.load('best_model.pth', weights_only=True))
                break

    # Plot metrics with validation
    plot_metrics_with_validation(train_losses, train_accuracies, val_losses, val_accuracies)

    return train_losses, train_accuracies, val_losses, val_accuracies


# Evaluation function
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


# Enhanced plotting function with validation metrics
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


# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract dataset
    extract_dataset()

    # Prepare data with augmentation, randomization and validation split
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_data(
        "data",
        batch_size=64,  # Increased batch size
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Initialize CNN-Transformer model with improved parameters
    model = CNNTransformer(
        num_classes=num_classes,
        embed_dim=512,  # Keep this
        depth=6,  # More transformer layers
        heads=8,  # Changed from 12 to 8 (512 ÷ 8 = 64)
        mlp_dim=1024,  # Larger feedforward dimension
        dropout=0.2  # Slightly increased dropout
    )
    model = model.to(device)
    print("Using enhanced CNN-Transformer architecture")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=256, eta_min=1e-6)

    # Train with validation monitoring
    print("Starting training with validation monitoring...")
    train_with_validation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs=256, patience=16  # Increased epochs and patience
    )

    # Load best model for final evaluation - fixed without invalid parameter
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # Evaluate on test data
    print("Evaluating model on test data...")
    test_accuracy = evaluate_model(model, test_loader, device)

    # Save the final model
    torch.save(model.state_dict(), 'enhanced_cnn_transformer.pth')
    print(f"Model saved with test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()