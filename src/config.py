import os
from dotenv import load_dotenv
load_dotenv("env\.env")

# --- Alpha Vantage API Constants ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_FUNCTION = "TIME_SERIES_INTRADAY"
DEFAULT_INTERVAL = "15min"
DEFAULT_OUTPUT_SIZE = "full"
DEFAULT_DATATYPE = "csv"

# --- Model Hyperparameters ---
HORIZON = 4 # Predict 4 time steps ahead (e.g., 4×15min = 1 hour ahead)

# SEQUENCE_LENGTH = 
# SEQUENCE_LENGTH = 
# LSTM_UNITS = 
# DROPOUT_RATE = 
# LEARNING_RATE = 
# BATCH_SIZE = 
# EPOCHS = 
# EARLY_STOPPING_PATIENCE = 

# --- File Paths ---
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"
REPORT_FIGURES_DV_DIR = "reports/figures/data_visualizations/"
REPORT_FIGURES_MR_DIR = "reports/figures/model_results"
SCALER_PARAMS_PATH = "data/processed/scaler_params.pkl"

# --- Others ---
PRED_HL_SELECTED_FEATURES = [
  'timestamp',
  'open', 'high', 'low', 'close', 'volume',
  'EMA_10', 'EMA_20',
  'MACD', 'MACD_signal', 'MACD_diff',
  'RSI_14', 'ROC_12', 'MOM_10', 'Williams_%R',
  'ATR_14', 'Volatility_20', 'BB_width',
  'OBV', 'CMF_20', 'VROC_12',
  'Close_Open_Ratio', 'High_Low_Ratio', 'Typical_Price'
]