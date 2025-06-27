from data_ingestion.fetch_data import IntradayFetcher
from utils.helpers import Visualizer
from feature_engineering.create_features import FeaturesCreator
from data_ingestion.preprocess_raw_data import Preprocessor

if __name__ == "__main__":
  # fetcher = IntradayFetcher()
  # fetcher.run()

  # features_creator = FeaturesCreator()
  # features_creator.run()

  # visualizer = Visualizer()
  # visualizer.run()

  preprocessor = Preprocessor()
  preprocessor.run()