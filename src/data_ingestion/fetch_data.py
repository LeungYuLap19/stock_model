import os
import requests
import logging
import time
import pandas as pd
from io import StringIO 
from datetime import date, timedelta
from config import (
  ALPHA_VANTAGE_API_KEY,
  ALPHA_VANTAGE_BASE_URL,
  DEFAULT_FUNCTION,
  DEFAULT_INTERVAL,
  DEFAULT_OUTPUT_SIZE,
  DEFAULT_DATATYPE,
  RAW_DATA_DIR,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntradayFetcher:
  def __init__(self):
    self.symbol = None
    self.months = 6
    self.raw_data_dir = RAW_DATA_DIR

  def run(self):
    self._user_input()
    self._fetch_api()

  def _user_input(self):
    print("Get Intraday Stock Data (15-min Interval)")

    while True:
      self.symbol = input("Enter stock symbol: ").strip().upper()
      if not self.symbol:
        print("❌ Symbol cannot be empty.")
        continue

      months_input = input("Input number of months to fetch [default 6]: ").strip()
      if not months_input:
        self.months = 6
      elif months_input.isdigit() and int(months_input) > 0:
        self.months = int(months_input)
      else:
        print("❌ Invalid months input. Please enter a positive integer.")
        continue

      break

  def _fetch_api(self):
    today = date.today()
    all_dfs = []

    for i in range(self.months):
      month_date = (today.replace(day=1) - timedelta(days=30 * i))
      month_str = month_date.strftime("%Y-%m")

      params = {
        "function": DEFAULT_FUNCTION,
        "symbol": self.symbol,
        "interval": DEFAULT_INTERVAL,
        "adjusted": "true",
        "extended_hours": "true",
        "month": month_str,
        "outputsize": DEFAULT_OUTPUT_SIZE,
        "datatype": DEFAULT_DATATYPE,
        "apikey": ALPHA_VANTAGE_API_KEY,
      }

      try:
        logger.info(f"Fetching data for {self.symbol}, month: {month_str}")
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)

        if response.status_code != 200:
          logger.error(f"Failed to fetch data for {self.symbol} ({month_str}): Status {response.status_code}")
          continue

        df = pd.read_csv(StringIO(response.text))
        all_dfs.append(df)

        logger.info(f"✔️ Loaded {len(df)} rows for {month_str}")
        time.sleep(12)

      except Exception as e:
        logger.error(f"Error fetching data for {self.symbol} ({month_str}): {e}")
        continue

    if all_dfs:
      self.dataframe = pd.concat(all_dfs, ignore_index=True)

      if "timestamp" in self.dataframe.columns:
        self.dataframe['timestamp'] = pd.to_datetime(self.dataframe['timestamp'], errors='coerce')
        self.dataframe = self.dataframe.sort_values(by='timestamp').reset_index(drop=True)

      path = os.path.join(self.raw_data_dir, f"{self.symbol}.csv")
      try:
        self.dataframe.to_csv(path, index=False)
        logger.info(f"✅ Merged data saved to {path}")
      except Exception as e:
        logger.error(f"Failed to save merged CSV: {e}")
    else:
        logger.warning("⚠️ No data collected to merge.")