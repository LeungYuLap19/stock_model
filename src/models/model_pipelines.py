import os
import torch
from torch.utils.data import DataLoader

from feature_engineering.create_sequences import SequencesCreator
from feature_engineering.split_sequences import SequenceSplitter
from datasets.stock_dataset import StockDataset
from .model_config import ModelConfig
from .build_model import LSTMModel
from .train_model import ModelTrainer
from config import PROCESSED_DATA_DIR, BATCH_SIZE

class ModelPipeline:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.symbol = None
    self.model = LSTMModel()
    self.sequences_creator = SequencesCreator()
    self.sequence_splitter = SequenceSplitter()

  def run(self):
    self._get_user_input()
    print("📈 Creating sequences...")
    X, y = self.sequences_creator.run(self.symbol)

    print("✂️ Splitting into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = self.sequence_splitter.run(X, y)

    print("📦 Building DataLoaders...")
    train_loader, val_loader = self._build_dataloaders(X_train, y_train, X_val, y_val)

    print("🧠 Initializing model config and trainer...")
    model_config = ModelConfig(self.model, self.device, train_loader, val_loader)
    model_trainer = ModelTrainer(model_config)

    print("🚀 Starting training...")
    model_trainer.train()

    print("✅ Training completed!")

  def _get_user_input(self):
    print("📌 This operation will perform the following steps:")
    print("  1. Create sequences")
    print("  2. Train-test split")
    print("  3. Train and evaluate model")
    print("  4. Save model after training")

    while True:
      # symbol = input("Enter stock symbol: ").strip().upper()
      symbol = "AAPL"
      if not symbol:
        print("❌ Symbol cannot be empty.")
        continue

      file_path = os.path.join(PROCESSED_DATA_DIR, f'{symbol}.csv')
      if not os.path.exists(file_path):
        print(f"❌ CSV for {symbol} not found in processed data directory.")
        continue

      self.symbol = symbol
      break

  def _build_dataloaders(self, X_train, y_train, X_val, y_val):
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader
