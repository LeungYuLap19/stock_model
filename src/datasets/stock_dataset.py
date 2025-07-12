import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
  def __init__(self, X, y):
    super(StockDataset, self).__init__()
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, index):
    x = torch.tensor(self.X[index], dtype=torch.float32)
    y = torch.tensor(self.y[index], dtype=torch.float32)

    return x, y