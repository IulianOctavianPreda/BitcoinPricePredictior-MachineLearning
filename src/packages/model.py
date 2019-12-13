from packages.trainingAlgorithm import TrainingAlgorithm
from sklearn import model_selection
from numpy import array

import packages.helpers.modelHelper as modelHelper


class Model:

    def __init__(self, algorithm: TrainingAlgorithm = None, X: array = None, y: array = None, testSize: number = 0.2, pathToModel: str = None, saveTrainedModel: bool = False, loadTrainedModel: bool = False):
        if loadTrainedModel is False:
            self.trainModel(X, y, testSize)
        else:
            modelHelper.loadModel(pathToModel)

    def trainModel(self, X, y, testSize, algorithm, saveTrainedModel, pathToModel):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=testSize)
        algorithm.fit(X_train, y_train)
        confidence = algorithm.score(X_test, y_test)

        if(saveTrainedModel is True and pathToModel is not None):
            modelHelper.saveModel(algorithm, pathToModel)

    def forcast(self, algorithm, forcastingDataframe):
        algorithm.predict(forcastingDataframe)
