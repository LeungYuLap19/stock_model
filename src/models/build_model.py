import torch
import torch.nn as nn
from config import (
  HORIZON,
  INPUT_SIZE,
  HIDDEN_SIZE,
  OUTPUT_SIZE,
  NUM_LAYERS,
  DROPOUT_RATE
)

class LSTMModel(nn.Module):
  def __init__(self):
    super(LSTMModel, self).__init__()
    self.input_size = INPUT_SIZE
    self.hidden_size = HIDDEN_SIZE
    self.output_size = OUTPUT_SIZE
    self.num_layers = NUM_LAYERS
    self.dropout_rate = DROPOUT_RATE

    self.lstm = nn.LSTM(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout=self.dropout_rate if self.num_layers > 1 else 0,
        batch_first=True
    )

    self.dropout = nn.Dropout(self.dropout_rate)
    self.fc = nn.Linear(self.hidden_size, self.output_size) 

  def forward(self, x):
    # x shape: (batch_size, seq_length, input_size)
    lstm_out, (h_n, c_n) = self.lstm(x)

    # Take the last time step's output
    last_hidden = lstm_out[:, -1, :]

    # Apply dropout and final linear layer
    out = self.dropout(last_hidden)
    pred = self.fc(out)  # Shape: (batch_size, 1)

    return pred  # Single value for percentage change