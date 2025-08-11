import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ModulationDataset(Dataset):
    """Custom dataset for modulation classification"""

    def __init__(self, filepath, sequence_length=128):
        """
        Args:
            filepath: Path to the text file containing I/Q data
            sequence_length: Number of I/Q samples per sequence
        """
        self.sequence_length = sequence_length
        self.data, self.labels = self._load_data(filepath)

    def _load_data(self, filepath):
        """Load and preprocess the I/Q data"""
        print("Loading I/Q data...")

        # Read the text file
        with open(filepath, 'r') as f:
            raw_data = f.read().strip().split()

        # Convert to float and reshape into I/Q pairs
        float_data = [float(x) for x in raw_data]

        # Ensure even number of values (I/Q pairs)
        if len(float_data) % 2 != 0:
            float_data = float_data[:-1]

        # Reshape into I/Q pairs
        iq_data = np.array(float_data).reshape(-1, 2)
        print(f"Total I/Q samples loaded: {len(iq_data)}")

        # Create labels based on the data structure described
        total_samples = len(iq_data)
        samples_per_class = 500000  # Updated to 500,000 samples per class

        # Ensure we have enough data
        expected_samples = 1500000  # 3 classes × 500,000 samples each
        if total_samples < expected_samples:
            print(f"Warning: Expected {expected_samples} samples, got {total_samples}")
            # Adjust samples_per_class if we don't have enough data
            samples_per_class = total_samples // 3
            print(f"Adjusted to {samples_per_class} samples per class")

        # Create labels: 0=BPSK, 1=QPSK, 2=16QAM
        labels = []
        labels.extend([0] * samples_per_class)  # BPSK
        labels.extend([1] * samples_per_class)  # QPSK
        labels.extend([2] * samples_per_class)  # 16QAM

        # Trim data to match labels
        total_used_samples = len(labels)
        iq_data = iq_data[:total_used_samples]
        labels = np.array(labels)

        print(f"Using {total_used_samples} total samples ({samples_per_class} per class)")

        # Create sequences
        sequences, seq_labels = self._create_sequences(iq_data, labels)

        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        print(f"Class distribution: BPSK={np.sum(seq_labels==0)}, QPSK={np.sum(seq_labels==1)}, 16QAM={np.sum(seq_labels==2)}")

        return sequences, seq_labels

    def _create_sequences(self, iq_data, labels):
        """Create overlapping sequences from I/Q data"""
        sequences = []
        seq_labels = []

        # Create sequences with 50% overlap
        step = self.sequence_length // 2

        print(f"Creating sequences with step size: {step}")
        print("This may take a few minutes for large datasets...")

        for i in range(0, len(iq_data) - self.sequence_length, step):
            sequence = iq_data[i:i + self.sequence_length]
            # Use the label of the middle sample
            label = labels[i + self.sequence_length // 2]

            sequences.append(sequence)
            seq_labels.append(label)

            # Progress indicator for large datasets
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1:,} samples...")

        return np.array(sequences), np.array(seq_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Normalize the sequence (zero mean, unit variance)
        sequence = self.data[idx].copy()
        sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

        return torch.FloatTensor(sequence), torch.LongTensor([self.labels[idx]])[0]

class ModulationCNN(nn.Module):
    """1D CNN for modulation classification"""

    def __init__(self, sequence_length=128, num_classes=3):
        super(ModulationCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(2, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second conv block
            nn.Conv1d(64, 128, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Third conv block
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Calculate the size after convolutions
        self.conv_output_size = self._get_conv_output_size(sequence_length)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_conv_output_size(self, sequence_length):
        """Calculate the output size after conv layers"""
        # Simulate forward pass through conv layers
        x = torch.randn(1, 2, sequence_length)
        x = self.conv_layers(x)
        return x.numel()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 2)
        # Transpose for Conv1d: (batch_size, 2, sequence_length)
        x = x.transpose(1, 2)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)

        return x

class ModulationLSTM(nn.Module):
    """LSTM-based modulation classifier"""

    def __init__(self, input_size=2, hidden_size=128, num_layers=3, num_classes=3):
        super(ModulationLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the modulation classification model"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None

    print("Starting training...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress tracking for large datasets
        batch_count = 0
        print_interval = max(1, len(train_loader) // 5)  # Print 5 times per epoch for very large dataset

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            batch_count += 1

            # Progress update for large datasets
            if batch_count % print_interval == 0:
                current_acc = 100. * train_correct / train_total
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print("Running validation...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best validation accuracy: {best_val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}] Summary: '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        print("-" * 80)

    # Load best model
    model.load_state_dict(best_model_state)

    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    model.eval()
    all_predictions = []
    all_targets = []

    print("Evaluating model...")
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"Processed {batch_count}/{len(test_loader)} test batches...")

    # Calculate accuracy
    accuracy = 100. * np.sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)

    # Class names
    class_names = ['BPSK', 'QPSK', '16-QAM']

    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, all_predictions, all_targets

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Main training and evaluation pipeline"""

    # Configuration - Optimized for very large dataset (500k per class)
    FILEPATH = r"data1.txt"  # UPDATE THIS PATH
    SEQUENCE_LENGTH = 128
    BATCH_SIZE = 256  # Further increased batch size for efficiency with 1.5M samples
    NUM_EPOCHS = 20   # Reduced epochs since we have massive amount of data
    LEARNING_RATE = 0.001

    print("Modulation Classification Neural Network")
    print("="*50)
    print(f"Configuration:")
    print(f"- Expected samples per class: 500,000")
    print(f"- Total expected samples: 1,500,000")
    print(f"- Sequence length: {SEQUENCE_LENGTH}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Number of epochs: {NUM_EPOCHS}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print("="*50)

    # Create dataset
    print("Creating dataset...")
    dataset = ModulationDataset(FILEPATH, sequence_length=SEQUENCE_LENGTH)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Dataset split - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")

    # Create data loaders - Reduced num_workers to avoid memory issues with large dataset
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    # Choose model architecture
    print("\nSelect model architecture:")
    print("1. CNN (Recommended)")
    print("2. LSTM")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '2':
        model = ModulationLSTM(input_size=2, hidden_size=128, num_layers=3, num_classes=3)
        model_name = "LSTM"
    else:
        model = ModulationCNN(sequence_length=SEQUENCE_LENGTH, num_classes=3)
        model_name = "CNN"

    model = model.to(device)

    print(f"\nUsing {model_name} architecture")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train the model
    print("\nStarting training with large dataset...")
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, predictions, targets = evaluate_model(model, test_loader)

    # Save the model
    model_path = f'modulation_classifier_{model_name.lower()}_500k.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model_name,
        'sequence_length': SEQUENCE_LENGTH,
        'test_accuracy': test_accuracy,
        'class_names': ['BPSK', 'QPSK', '16-QAM'],
        'samples_per_class': 500000
    }, model_path)

    print(f"\nModel saved as: {model_path}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Training completed with 500,000 samples per modulation class!")

if __name__ == "__main__":
    main()
