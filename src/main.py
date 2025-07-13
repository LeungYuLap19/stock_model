from InquirerPy import prompt
from data_ingestion.fetch_data import IntradayFetcher
from utils.helpers import Visualizer
from feature_engineering.create_features import FeaturesCreator
from data_ingestion.preprocess_raw_data import Preprocessor
from models.model_pipelines import ModelPipeline

if __name__ == "__main__":
  model_pipelines = ModelPipeline()
  model_pipelines.run()

  # questions = [
  #   {
  #     "type": "list",
  #     "name": "operation",
  #     "message": "Select an operation:",
  #     "choices": [
  #       "Fetch - Preprocess",
  #       "Split - Train",
  #       "Exit"
  #     ]
  #   }
  # ]

  # while True:
  #   answer = prompt(questions)['operation']

  #   if answer == "Fetch - Preprocess":
  #     fetcher = IntradayFetcher()
  #     fetcher.run()

  #     features_creator = FeaturesCreator()
  #     features_creator.run()

  #     visualizer = Visualizer()
  #     visualizer.run()

  #     preprocessor = Preprocessor()
  #     preprocessor.run()

  #   elif answer == "Split - Train":
  #     model_pipelines = ModelPipeline()
  #     model_pipelines.run()

  #   elif answer == "Exit":
  #     print("Exiting.")
  #     break
  


