import pandas as pd
import ta
import os
from config import RAW_DATA_DIR

class FeaturesCreator:
  def __init__(self):
    self.raw_data_dir = RAW_DATA_DIR

  def run(self):
    csv_files = [
      f for f in os.listdir(self.raw_data_dir)
      if f.endswith(".csv") and os.path.isfile(os.path.join(self.raw_data_dir, f))
    ]

    if not csv_files:
      print("❌ No CSV files found in the raw data directory.")
      return

    for file_name in csv_files:
      path = os.path.join(self.raw_data_dir, file_name)
      try:
        print(f"🔧 Processing: {file_name}")
        self._add_features_to_csv(path)
      except Exception as e:
        print(f"❌ Failed to process {file_name}: {e}")

  def _add_features_to_csv(self, path: str):
    df = pd.read_csv(path)
    df_with_features = self._compute_ohlcv_features(df)
    df_with_features.to_csv(path, index=False)
    print(f"Saved enriched data with features to: {path}")

  def _compute_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Ensure correct column types
    df = df.copy()
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Simple Moving Averages
    df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)

    # Exponential Moving Averages
    df['EMA_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

    # MACD and Signal
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Average Directional Index (ADX)
    df['ADX_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    # Relative Strength Index (RSI)
    df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)

    # Rate of Change (ROC)
    df['ROC_12'] = ta.momentum.roc(df['close'], window=12)

    # Stochastic Oscillator (%K and %D)
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['STOCH_%K'] = stoch.stoch()
    df['STOCH_%D'] = stoch.stoch_signal()

    # Momentum
    df['MOM_10'] = ta.momentum.roc(df['close'], window=10)  # ROC used as momentum proxy

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_width'] = df['BB_high'] - df['BB_low']

    # Average True Range (ATR)
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Volatility (Rolling Std Dev of Close)
    df['Volatility_20'] = df['close'].rolling(window=20).std()

    # On-Balance Volume (OBV)
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # Chaikin Money Flow (CMF)
    df['CMF_20'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)

    # Volume Rate of Change (VROC)
    df['VROC_12'] = df['volume'].pct_change(periods=12) * 100

    # Price-derived Ratios
    df['Close_Open_Ratio'] = df['close'] / df['open']
    df['High_Low_Ratio'] = df['high'] / df['low']
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3

    # Williams %R
    df['Williams_%R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

    # Commodity Channel Index (CCI)
    df['CCI_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

    # Pivot Points (using typical price)
    df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['Support1'] = 2 * df['Pivot'] - df['high']
    df['Resistance1'] = 2 * df['Pivot'] - df['low']
    df['Support2'] = df['Pivot'] - (df['high'] - df['low'])
    df['Resistance2'] = df['Pivot'] + (df['high'] - df['low'])

    return df