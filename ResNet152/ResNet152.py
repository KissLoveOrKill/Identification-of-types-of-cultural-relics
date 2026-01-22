# ResNet152.py
import os
import zipfile
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import functools  # Add this import
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Define a top-level worker init function for proper pickling
def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)


# Custom ResNet152 Implementation
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet152(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet152, self).__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet152 layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 8, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 36, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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


# Prepare data loaders with enhanced image augmentation and validation set
def prepare_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Training transforms with extensive augmentation
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

    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with temporary transform
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Calculate sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create shuffled indices
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=data_dir, transform=train_transform),
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=data_dir, transform=test_transform),
        val_indices
    )

    test_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=data_dir, transform=test_transform),
        test_indices
    )

    # Create data loaders with properly pickable worker init function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=functools.partial(worker_init_fn, seed=seed),
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(
        f"Dataset loaded with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} testing samples")
    print(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, num_classes, class_names


# Train with validation monitoring
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10,
                          patience=5):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    patience_counter = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')

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
                # Load best model
                model.load_state_dict(torch.load('best_model.pth', weights_only=True))
                break

    # Plot metrics with validation
    plot_metrics_with_validation(train_losses, train_accuracies, val_losses, val_accuracies)

    return train_losses, train_accuracies, val_losses, val_accuracies


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


# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract dataset
    extract_dataset()

    # Prepare data with augmentation, randomization and validation split
    train_loader, val_loader, test_loader, num_classes, class_names = prepare_data(
        "data",
        batch_size=64,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )

    # Initialize ResNet152 model
    model = ResNet152(num_classes=num_classes)
    model = model.to(device)
    print("Using ResNet152 architecture")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Train with validation monitoring
    print("Starting training with validation monitoring...")
    train_with_validation(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs=50, patience=10
    )

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # Evaluate on test data
    print("Evaluating model on test data...")
    test_accuracy = evaluate_model(model, test_loader, device)

    # Save the final model
    torch.save(model.state_dict(), 'resnet152_model.pth')
    print(f"Model saved with test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()