from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn import CrossEntropyLoss
from config import (
  LEARNING_RATE,
  EPOCHS,
  WEIGHT_DECAY
)

class ModelConfig:
  def __init__(self, model, device, train_loader, val_loader):
    self.model = model.to(device)
    self.device = device
    self.train_loader = train_loader
    self.val_loader = val_loader
    
    self.num_epochs = EPOCHS
    self.lr = LEARNING_RATE
    self.weight_decay = WEIGHT_DECAY
    
    self.criterion = CrossEntropyLoss()
    self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    self.warmup_epochs = max(1, self.num_epochs // 10)
    self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: min(epoch / self.warmup_epochs, 1.0))
    self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)