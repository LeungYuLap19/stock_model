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