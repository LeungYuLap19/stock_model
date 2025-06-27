import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import (
  RAW_DATA_DIR,
  PROCESSED_DATA_DIR,
  PRED_HL_SELECTED_FEATURES,
  SCALER_PARAMS_PATH
)

class Preprocessor:
  def __init__(self):
    self.raw_dfs = {}
    self.processed_dfs = {}
    self.raw_data_dir = RAW_DATA_DIR
    self.processed_data_dir = PROCESSED_DATA_DIR

  def run(self):
    self._load_csv()
    self._inspect_raw_data()

    processing_dfs = self.raw_dfs
    processing_dfs = self._drop_columns(processing_dfs)
    processing_dfs = self._handle_missing_values(processing_dfs)
    processing_dfs = self._scale_data(processing_dfs)
    self.processed_dfs = processing_dfs

    self._inspect_processed_data()
    self._save_csv()

  def _load_csv(self):
    for filename in tqdm(os.listdir(self.raw_data_dir)):
      if filename.endswith('.csv'):
        file_path = os.path.join(self.raw_data_dir, filename)
        self.raw_dfs[filename] = pd.read_csv(file_path)

  def _save_csv(self):
    for filename, df in tqdm(self.processed_dfs.items()):
      file_path = os.path.join(self.processed_data_dir, filename)
      df.to_csv(file_path, index=False)
  
  def _inspect_raw_data(self):
    for filename, df in self.raw_dfs.items():
      print(f"Inspecting {filename}")
      print(df.head())
      print(df.columns)
      print(df.info())
  
  def _inspect_processed_data(self):
    for filename, df in self.processed_dfs.items():
      print(f"Inspecting {filename}")
      print(df.head())
      print(df.columns)
      print(df.info())

  def _drop_columns(self, processing_dfs):
    for filename, df in processing_dfs.items():
      processing_dfs[filename] = df[PRED_HL_SELECTED_FEATURES]
    
    return processing_dfs
  
  def _handle_missing_values(self, processing_dfs):
    for filename, df in processing_dfs.items():
        processing_dfs[filename] = df.dropna()
    
    return processing_dfs
  
  def _scale_data(self, processing_dfs):
    scalers = {}

    for filename, df in processing_dfs.items():
      feature_cols = [col for col in df.columns if col not in ['timestamp', 'high', 'low']]
      target_cols = ['timestamp', 'high', 'low']

      feature_scaler  = MinMaxScaler()
      scaled_features = feature_scaler .fit_transform(df[feature_cols])
      scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)

      target_df = df[target_cols]

      processing_dfs[filename] = pd.concat([scaled_features_df, target_df], axis=1)[df.columns]

      scalers[filename] = { 'feature_scaler': feature_scaler } 

    joblib.dump(scalers, SCALER_PARAMS_PATH)

    return processing_dfs