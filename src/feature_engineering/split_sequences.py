from sklearn.model_selection import train_test_split
from config import (
  TEST_SIZE,
  VAL_SIZE
)

class SequenceSplitter:
  def __init__(self):
    self.test_size = TEST_SIZE
    self.val_size = VAL_SIZE

    self.X = None
    self.y = None

  def run(self, X, y):
    self.X = X
    self.y = y
    return self._split()

  def _split(self):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        self.X, self.y, test_size=self.test_size, shuffle=False
    )

    val_ratio = self.val_size / (1 - self.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, shuffle=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test