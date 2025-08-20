import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

class MultiRateCNN(nn.Module):
    def __init__(self, output_steps=17, num_classes=2):
        super().__init__()

        # 200 Hz branch: downsample by 10x
        self.branch_200Hz = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),  # -> 100 Hz
            nn.BatchNorm1d(8),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),                           # -> 50 Hz
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1), # -> 25 Hz
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(output_steps)                     # -> 20 Hz-like output
        )

        # 50 Hz branch: downsample by 2.5x
        self.branch_50Hz = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),  # -> 50 Hz
            nn.BatchNorm1d(8),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2),                           # -> 25 Hz
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1), # keep 25 Hz
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(output_steps)                     # match 20 Hz
        )

        # 20 Hz branch: no downsampling
        self.branch_20Hz = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(output_steps)  # directly to same number of time steps
        )

        # Fusion: concatenate along channel dimension
        self.fusion = nn.Sequential(
            nn.Conv1d(16 * 3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),  # collapse time dimension
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, x_20, x_200, x_50):
        out_200 = self.branch_200Hz(x_200.unsqueeze(1))  # [B, 64, T]
        out_50  = self.branch_50Hz(x_50.unsqueeze(1))
        out_20  = self.branch_20Hz(x_20.unsqueeze(1))
        fused = torch.cat([out_200, out_50, out_20], dim=1)  # [B, 192, T]
        return self.fusion(fused)

class EarlyStopping:
    def __init__(self, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.epochs_no_improve = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
                return True  # Stop training
        return False
    
class MultiModalDataset(Dataset):
    def __init__(self, ppg, pcg, acc, labels):
        self.ppg = ppg
        self.pcg = pcg
        self.acc = acc
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'ppg': self.ppg[idx],
            'pcg': self.pcg[idx],
            'acc': self.acc[idx],
            'label': self.labels[idx]
        }

def train_model(model, criterion, optimizer, scheduler, x_train, y_train, x_val, y_val, batch_size=32, epochs=20):
    train_dataset = MultiModalDataset(x_train[0], x_train[1], x_train[2], y_train)
    val_dataset = MultiModalDataset(x_val[0], x_val[1], x_val[2], y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            ppg = batch['ppg']
            pcg = batch['pcg']
            acc = batch['acc']
            labels = batch['label'].float().unsqueeze(1)
            #labels = batch['label'].long()
            optimizer.zero_grad()
            outputs = model(ppg, pcg, acc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            #predicted = torch.argmax(outputs, dim=1).long()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ppg = batch['ppg']
                pcg = batch['pcg']
                acc = batch['acc']
                labels = batch['label'].float().unsqueeze(1)
                #labels = batch['label'].long()
                outputs = model(ppg, pcg, acc)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                #predicted = torch.argmax(outputs, dim=1).long()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total

        print(
            f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

        scheduler.step(val_acc)

        if early_stopping(val_loss, model):
            print("Early stopping triggered. Restoring best weights.")
            break

    return model

def evaluate_model(model, criterion, x_test, y_test, batch_size=32):
    test_dataset = MultiModalDataset(x_test[0], x_test[1], x_test[2], y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
            for batch in test_loader:
                ppg = batch['ppg']
                pcg = batch['pcg']
                acc = batch['acc']
                labels = batch['label'].float().unsqueeze(1)
                #labels = batch['label'].long()
                outputs = model(ppg, pcg, acc)
                loss = criterion(outputs, labels)

                predicted = (torch.sigmoid(outputs) > 0.5).long()
                #predicted = (torch.argmax(outputs, dim=1)).long()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
    return correct / total

def model_output(model, criterion, x, y, batch_size=32):
    test_dataset = MultiModalDataset(x[0], x[1], x[2], y)
    test_loader = DataLoader(test_dataset, batch_size=y.size()[0])

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
            for batch in test_loader:
                ppg = batch['ppg']
                pcg = batch['pcg']
                acc = batch['acc']
                labels = batch['label'].float().unsqueeze(1)
                #labels = batch['label'].long()

                outputs = model(ppg, pcg, acc)
                #predicted = torch.argmax(outputs, dim=1).long()
                predicted = outputs
    return predicted