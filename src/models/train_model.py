from .model_config import ModelConfig
from config import MODELS_DIR
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datetime import datetime
import os
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

class ModelTrainer:
  def __init__(self, config: ModelConfig):
    self.model_datetime = datetime.now().strftime("%Y%m%d_%H%M")

    self.model = config.model
    self.device = config.device
    self.criterion = config.criterion
    self.optimizer = config.optimizer
    self.scheduler = config.scheduler
    self.train_loader = config.train_loader
    self.val_loader = config.val_loader
    self.num_epochs = config.num_epochs

    self.best_val_loss = float('inf')
    self.best_val_acc = 0.0
    self.train_loss_history = []
    self.val_loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

  def train_epoch(self):
    self.model.train()
    total_epoch_loss = 0
    all_preds = []
    all_targets = []

    for inputs, labels in self.train_loader:
      # zero gradient
      self.optimizer.zero_grad()
      
      # forward
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      outputs = self.model(inputs) # Shape: (batch_size, 2)
      loss = self.criterion(outputs, labels)
      
      # backpropagation
      loss.backward()
      clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient clipping, prevent gradient explosion
      self.optimizer.step()
      # self.scheduler.step()

      # update metrics
      total_epoch_loss += loss.item()
      preds = torch.argmax(outputs, dim=1)  # Get class predictions (0 or 1)
      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(labels.cpu().numpy())

    avg_loss = total_epoch_loss / len(self.train_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy
  
  def val_epoch(self):
    self.model.eval()
    total_epoch_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
      for inputs, targets in self.val_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)  # Get class predictions (0 or 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_epoch_loss / len(self.val_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, np.array(all_targets), np.array(all_preds), accuracy
  
  def train(self, patience=10):
    best_epoch = -1
    best_preds = None
    best_targets = None
    early_stop_counter = 0

    for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training Epochs"):
      train_loss, train_acc = self.train_epoch()
      val_loss, targets, preds, val_acc = self.val_epoch()

      self.train_loss_history.append(train_loss)
      self.val_loss_history.append(val_loss)
      self.train_acc_history.append(train_acc)
      self.val_acc_history.append(val_acc)

      print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

      if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        self.best_val_acc = val_acc
        best_preds = preds
        best_targets = targets
        best_epoch = epoch
        early_stop_counter = 0
        torch.save({
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'epoch': epoch,
          'val_loss': val_loss,
          'val_acc': val_acc,
          'datetime': self.model_datetime
        }, os.path.join(MODELS_DIR, f"{self.model_name}_best_model_{self.model_datetime}.pth"))
      else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
          print("Early stopping triggered")
          break

      self.scheduler.step()  # Step scheduler after each epoch

    print(f"\n✅ Best Validation Loss: {self.best_val_loss:.4f}, Best Validation Acc: {self.best_val_acc:.4f} at epoch {best_epoch}")
    self.evaluate(best_targets, best_preds)
    self.plot_training_curve()
    self.plot_predictions(best_targets, best_preds)

  def evaluate(self, y_true, y_pred):
    if y_true is None or y_pred is None:
      print("No valid predictions to evaluate")
      return
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    print(f"Evaluation Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}")

  def plot_training_curve(self):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(self.train_loss_history, label='Train Loss')
    plt.plot(self.val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(self.train_acc_history, label='Train Accuracy')
    plt.plot(self.val_acc_history, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f"{self.model_name}_training_curve_{self.model_datetime}.png"))
    plt.close()

  def plot_predictions(self, y_true, y_pred):
    if y_true is None or y_pred is None:
      print("No valid predictions to plot")
      return
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_true)), y_true, label='True Labels', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted Labels', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Class (0: Down, 1: Up)')
    plt.title('True vs Predicted Labels')
    plt.legend()
    plt.savefig(os.path.join(MODELS_DIR, f"{self.model_name}_predictions_{self.model_datetime}.png"))
    plt.close()

  def load_best_model(self):
    checkpoint_path = os.path.join(MODELS_DIR, f"{self.model_name}_best_model_{self.model_datetime}.pth")
    if not os.path.exists(checkpoint_path):
      print(f"No checkpoint found at {checkpoint_path}")
      return False
    checkpoint = torch.load(checkpoint_path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.best_val_loss = checkpoint['val_loss']
    self.best_val_acc = checkpoint['val_acc']
    print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}, "
          f"val acc {checkpoint['val_acc']:.4f}")
    return True