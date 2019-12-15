from .helpers.datasetHelper import DatasetHelper
from .helpers.datetimeHelper import DateTimeHelper
from pandas import DataFrame, concat
from numpy import array, ndarray

import copy


class Dataset:
    _dataset: DataFrame
    _features: str
    _label: str

    def __init__(self, path: str, features: str, label: str, ):
        self._dataset = DatasetHelper.readCsv(path)
        self._features = features
        self._label = label

        self.prepareDataset()

    def getDataset(self):
        """Gets a reference of the dataset object"""
        return self._dataset

    def getDatasetCopy(self):
        """Gets a copy of the dataset object"""
        return copy.copy(self._dataset)

    def copy(self):
        return copy.copy(self)

    def prepareDataset(self):
        """Modifies the dataset object with a user defined logic"""
        self._dataset = DatasetHelper.prepareDataset(self.getDatasetCopy())

    def getFeatureData(self):
        """Gets the data held in the feature columns"""
        return array(self._dataset.drop([self._label], 1))

    def getLabelData(self):
        return array(self._dataset[self._label])

    def updateLabelData(self, values: ndarray):
        self._dataset[self._label] = values

    def truncateDataset(self, before=DateTimeHelper.inPastAsString(months=1), after=DateTimeHelper.nowAsString()):
        self._dataset = self._dataset.truncate(before=before, after=after)

    def addRow(self, row: DataFrame):
        self._dataset = concat([row, self._dataset.iloc[:]])

    def shape(self):
        print('Dataset shape:\n', self._dataset.shape, '\n')

    def head(self, rows: int = 10, truncate: bool = False):
        if truncate:
            DatasetHelper.resetPrintingOptions()
        else:
            DatasetHelper.printFullOutput(rows)
        print('Dataset HEAD:\n', self._dataset.head(rows), '\n')

    def description(self):
        print('Dataset statistics:\n', self._dataset.describe(), '\n')

    def distribution(self, classification: str):
        print('Class distribution:\n', self._dataset.groupby(
            classification).size(), '\n')
