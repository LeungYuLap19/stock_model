import os
import pandas as pd
import numpy as np
from config import (
  SEQUENCE_LENGTH,
  HORIZON,
  PROCESSED_DATA_DIR
)

class SequencesCreator:
  def __init__(self):
    self.sequence_length = SEQUENCE_LENGTH
    self.horizon = HORIZON
    self.processed_data_dir = PROCESSED_DATA_DIR
    self.symbol = None
    self.df = None

  def run(self, symbol):
    self.symbol = symbol
    self._load_csv()
    return self._create_sequences()

  def _load_csv(self):
    file_path = os.path.join(self.processed_data_dir, f'{self.symbol}.csv')
    self.df = pd.read_csv(file_path)
    self.df = self.df.drop('timestamp', axis=1)

  def _create_sequences(self):
    X, y = [], []
    for i in range(len(self.df) - self.sequence_length - self.horizon):
      x_seq = self.df[i:i + self.sequence_length].copy()
      y_target = self.df.iloc[i + self.sequence_length + self.horizon - 1][['high', 'low']].copy()
      X.append(x_seq.values)
      y.append(y_target.values)

    return np.array(X), np.array(y)