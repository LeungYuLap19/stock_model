import os
from feature_engineering.create_sequences import SequencesCreator
from feature_engineering.split_sequences import SequenceSplitter
from config import (
  PROCESSED_DATA_DIR
)

class ModelPipelines:
  def __init__(self):
    self.sequences_creator = SequencesCreator()
    self.sequence_splitter = SequenceSplitter()
    self.processed_data_dir = PROCESSED_DATA_DIR
    self.symbol = None

  def run(self):
    self._user_input()
    X, y = self.sequences_creator.run(self.symbol)
    X_train, X_val, X_test, y_train, y_val, y_test = self.sequence_splitter.run(X, y)

  def _user_input(self):
    print("This operation will perform the following:")
    print("1. Create sequences")
    print("2. Train test split")
    print("3. Train and evaluation model")
    print("4. Save model")

    while True:
      self.symbol = input("Enter stock symbol: ").strip().upper()
      if not self.symbol:
        print("❌ Symbol cannot be empty.")
        continue

      # Check file existence here
      file_path = os.path.join(self.processed_data_dir, f'{self.symbol}.csv')
      if not os.path.exists(file_path):
          print(f"❌ CSV for {self.symbol} not found in processed data directory.")
          continue

      break