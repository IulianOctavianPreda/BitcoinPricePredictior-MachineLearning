from packages.trainingAlgorithm import TrainingAlgorithm

from pandas import DataFrame
from sklearn import preprocessing
import pickle


class ModelHelper:
    @staticmethod
    def loadModel(self, path: str) -> TrainingAlgorithm:
        return pickle.load(open(path, 'rb'))

    @staticmethod
    def saveModel(self, model, path: str):
        pickle.dump(model, open(path, 'wb'))

    @staticmethod
    def scaleDataset(self, dataset: DataFrame) -> DataFrame:
        # Get column names first
        names = dataset.columns
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        scaled_dataset = scaler.fit_transform(dataset)
        scaled_dataset = DataFrame(scaled_dataset, columns=names)
        return scaled_dataset

    @staticmethod
    def scaleArray(self, array: list) -> list:
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        scaled_array = scaler.fit_transform(array)
        return scaled_array
