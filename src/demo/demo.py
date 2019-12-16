from ..dataset import Dataset
from ..model import Model
from ..scaler import Scaler
from ..helpers.plotterHelper import PlotterHelper
from ..types.modelType import ModelType
from ..types.scalerType import ScalerType
from operator import itemgetter


class Demo:
    @staticmethod
    def demo(pathToDataset: str = "./src/demo/dataset/bitcoin_price.csv",
             pathToPredictionDataset: str = "./src/demo/dataset/bitcoin_price_for_prediction.csv",
             features: str = "['Open',  'High',  'Low',  'Close', 'Volume', 'Market Cap']",
             label: str = "Close",
             indexColumn: str = "Date"):
        """Static method that will test all the predictions algorithms"""
        dataset = Dataset(pathToDataset, features, label)
        dataset.truncateDataset()
        scaler = Scaler(ScalerType.StandardScaler, dataset.getFeatureData())
        predictionDataset = Dataset(pathToPredictionDataset, features, label)

        print(dataset.head())
        print(dataset.shape())
        print(predictionDataset.head())
        print(predictionDataset.shape())

        scores = {}

        for types in ModelType:
            copyOfDataset = dataset.copy()
            copyOfPredictionDataset = predictionDataset.copy()

            X = scaler.transform(copyOfDataset.getFeatureData())
            y = copyOfDataset.getLabelData()

            predictionData = scaler.transform(copyOfPredictionDataset.getFeatureData())

            model = Model(types, X, y, 0.2)

            prediction = model.predict(predictionData)
            scoreList = model.evaluate(X,y)
            print(model.getAlgorithmUsed(), " Accuracy:", scoreList)
            scores[model.getAlgorithmUsed()] = scoreList
            # print("predictionData", predictionData)
            # print("prediction", prediction)

            copyOfPredictionDataset.updateLabelData(prediction)
            copyOfPredictionDataset.addRow(copyOfDataset.getDataset().tail(1))

            # print(copyOfPredictionDataset.head())
            # print(copyOfPredictionDataset.getDataset()[label])
            # print(copyOfPredictionDataset.getDataset()[label])


            # For plotting purposes copy the last line of the dataset to the predicted values so the plotter will show an
            # uninterrupted line
            PlotterHelper.plot(copyOfDataset.getDataset()[label], copyOfPredictionDataset.getDataset()[label], label,
                               indexColumn, model.getAlgorithmUsed(), True)
        PlotterHelper.plotEvaluations(list(map(itemgetter(0), scores.items())),
                                      list(map(itemgetter(1), scores.items())), True)
#