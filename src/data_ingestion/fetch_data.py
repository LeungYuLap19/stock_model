import os
import requests
import logging
import time
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
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

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

    for i in range(self.months):
      month_date = (today.replace(day=1) - timedelta(days=30 * i))
      month_str = month_date.strftime("%Y-%m")

      print(ALPHA_VANTAGE_API_KEY)

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

        file_path = os.path.join(RAW_DATA_DIR, f"{self.symbol}_{month_str}.csv")
        with open(file_path, "w", encoding="utf-8") as f:
          f.write(response.text)

        logger.info(f"Saved raw data to {file_path}")

        time.sleep(12)

      except Exception as e:
        logger.error(f"Error fetching data for {self.symbol} ({month_str}): {e}")
        continue