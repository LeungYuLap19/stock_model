from data_ingestion.fetch_data import IntradayFetcher
from utils.helpers import Visualizer
import os
from config import RAW_DATA_DIR

if __name__ == "__main__":
  # fetcher = IntradayFetcher()
  # fetcher.run()

  visualizer = Visualizer()
  visualizer.run_all()