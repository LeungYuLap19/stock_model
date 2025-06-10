from data_ingestion.fetch_data import IntradayFetcher
from utils.helpers import Visualizer
from feature_engineering.create_features import FeaturesCreator

if __name__ == "__main__":
  # fetcher = IntradayFetcher()
  # fetcher.run()

  # features_creator = FeaturesCreator()
  # features_creator.run()

  visualizer = Visualizer()
  visualizer.run()