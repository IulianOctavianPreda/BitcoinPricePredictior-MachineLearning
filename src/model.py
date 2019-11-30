import csv
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

from enum import Enum

style.use('ggplot')


class Strategy(Enum):
    All = 0
    LinearRegression = 1
    BLUE = 3


class Model:
    _features: list = None
    _label: str = None
    _path: str = None
    _strategy: Strategy = None

    _dataframe: pd.DataFrame = None

    _forecastColumn: str = None
    _forcastedPeriod: int = None
    _X: np.array = None
    _y: np.array = None
    _X_lately: np.array = None
    _X_train: list = None
    _X_test: list = None
    _y_train: list = None
    _y_test: list = None
    _confidence: float = None

    def __init__(self, pathToDataset: str, features: list, label: str, strategy: Strategy):
        self._features = features
        self._label = label
        self._path = pathToDataset
        self._strategy = strategy
        self.setUp()

    def setUp(self):
        df = self.createDataframe(self._path)
        self.regressionModel(
            *self.prepareTrainingSets(self.prepareDataframe(df)))

    def createDataframe(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, header=0, index_col='Date', parse_dates=True)
        # with open(path, 'r') as file:
        #     header = list(file.readline().split(','))
        #     header = [x.replace(' ', '').rstrip() for x in header]
        # return pd.read_csv(path, names=header, skiprows=1)

    def prepareDataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[self._features]

        df['HighLowPercentage'] = (
            df['High'] - df['Low']) / df['Close'] * 100.0
        df['PercentageChange'] = (
            df['Close'] - df['Open']) / df['Open'] * 100.0

        df = df[['Close', 'HighLowPercentage',
                 'PercentageChange', 'Volume', 'Market Cap']]

        # fill NA values with an obvious different value that will not be taken in consideration
        df.fillna(value=-99999, inplace=True)

        return df

    def prepareTrainingSets(self, df: pd.DataFrame):
         # number of days in the future
        forcastedPeriod = int(math.ceil(0.01 * len(df)))

        # X = np.array(df.drop(['label'], 1))
        X = np.array(df)
        # scale the data set
        X = preprocessing.scale(X)
        # select all rows from the dataframe from forcastedPeriod days ago
        # since we have the date in descending order we will select the first forcastedPeriod entries
        X_lately = X[-forcastedPeriod:]
        # X_lately = X[:forcastedPeriod]

        # select all rows from the dataframe from the past until forcastedPeriod days
        X = X[:-forcastedPeriod]
        # X = X[forcastedPeriod:]

        # to predict in the future
        # this will create the column label and will append at the end ( we use ascending date) forcastedPeriod lines
        df['label'] = df[self._label].shift(-forcastedPeriod)

        df.dropna(inplace=True)
        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2)

        return df, X_train, X_test, y_train, y_test, X_lately

    def regressionModel(self, df, X_train, X_test, y_train, y_test, X_lately):
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print(confidence)
        forecast_set = clf.predict(X_lately)
        df['Forecast'] = np.nan

        last_date = df.iloc[-1].name
        # last_date = datetime.datetime(2019, 11, 28)
        # print(last_date)
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        for i in forecast_set:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += 86400
            df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

        df['Close'].plot()
        df['Forecast'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


model = Model("./dataset/bitcoin_price_20170101_20191129_asc.csv",
              ['Open',  'High',  'Low',  'Close', 'Volume', 'Market Cap'], "Close", Strategy.All)
