import packages.helpers.datasetHelper as datasetHelper
from pandas import DataFrame
from numpy import array

import copy


class Dataset:
    _dataset: DataFrame
    _features: list
    _label: str

    def __init__(self, path: str, features: list, label: str):
        self._dataset = datasetHelper.readCsv(path)
        self._features = features
        self._label = label

        self.prepareDataset()

    def getDataset(self):
        return self._dataset

    def getDatasetCopy(self):
        return copy.copy(self._dataset)

    def prepareDataset(self):
        self._dataset = datasetHelper.prepareDataset(
            self.getDatasetCopy(), self._features)

    def getTrainingData(self):
        X = array(self._dataset.drop([self._label], 1))
        X = self.scaleArray(X)
        y = array(self._dataset[self._label])
        return X, y

    def shape(self):
        print('Dataset shape:\n', self._dataset.shape, '\n')

    def head(self, rows: int):
        print('Dataset HEAD:\n', self._dataset.head(rows), '\n')

    def description(self):
        print('Dataset statistics:\n', self._dataset.describe(), '\n')

    def distribution(self, classification: str):
        print('Class distribution:\n', self._dataset.groupby(
            classification).size(), '\n')
