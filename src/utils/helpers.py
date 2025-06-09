import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ( RAW_DATA_DIR, REPORT_FIGURES_DV_DIR )

class Visualizer:
  def __init__(self):
    self.raw_data_dir = RAW_DATA_DIR
    self.report_figures_dv_dir = REPORT_FIGURES_DV_DIR
    os.makedirs(self.report_figures_dv_dir, exist_ok=True)

  def run(self, filename: str):
    df = self._load_csv(filename=filename)
    self._plot_closing_price(
      df=df,
      symbol=filename.split('_')[0],
      save_name=filename + "_fig.png"
    )

  def run_all(self):
    csv_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith(".csv")]
    if not csv_files:
      print("⚠️ No CSV files found in RAW_DATA_DIR.")
      return

    for filename in csv_files:
      try:
        print(f"📈 Visualizing: {filename}")
        self.run(filename)
      except Exception as e:
        print(f"❌ Error visualizing {filename}: {e}")
    
  def _load_csv(self, filename: str) -> pd.DataFrame:
    filepath = os.path.join(self.raw_data_dir, filename)
    if not os.path.exists(filepath):
      raise FileNotFoundError(f"CSV file not found: {filepath}")
    return pd.read_csv(filepath)
  
  def _plot_closing_price(
    self, 
    df: pd.DataFrame,
    symbol: str,
    save_name: str
  ):
    if "timestamp" in df.columns:
      df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    plt.title(f'{symbol} Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(self.report_figures_dv_dir, save_name)
    plt.savefig(save_path)
    print(f"📊 Plot saved to: {save_path}")

    plt.close()