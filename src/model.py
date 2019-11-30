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
    _X: np.array = None
    _y: np.array = None
    _X_lately: np.array = None
    _X_train: list = None
    _X_test: list = None
    _y_train: list = None
    _y_test: list = None

    def __init__(self, pathToDataset: str, features: list, label: str, strategy: Strategy):
        self._features = features
        self._label = label
        self._path = pathToDataset
        self._strategy = strategy
        self.setUp()

    def setUp(self):
        _df, _X_train, _X_test, _y_train, _y_test, _X_lately = self.prepareTrainingSets(
            self.prepareDataframe(self.createDataframe(self._path)))

        self.regressionModel(_df, _X_train, _X_test,
                             _y_train, _y_test, _X_lately)

    def createDataframe(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, header=0, index_col='Date', parse_dates=True)

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

    def scaleDataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get column names first
        names = df.columns
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        scaled_df = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_df, columns=names)
        return scaled_df

    def scaleArray(self, array):
        # Create the Scaler object
        scaler = preprocessing.StandardScaler()
        # Fit your data on the scaler object
        scaled_array = scaler.fit_transform(array)
        return scaled_array

    def prepareTrainingSets(self, df: pd.DataFrame):
        # number of days in the future
        forcastedPeriod = int(math.ceil(0.01 * len(df)))
        # to predict in the future
        # this will create the column label and will append at the end ( we use ascending date) <forcastedPeriod> lines
        df['label'] = df[self._label].shift(-forcastedPeriod)

        X = np.array(df.drop(['label'], 1))
        # df.to_csv('./adjustedDataset.csv')
        # X = np.array(df)

        # scale the data set
        X = self.scaleArray(X)
        # select all rows from the dataframe from <forcastedPeriod> days ago
        # since we have the date in descending order we will select the first <forcastedPeriod> entries
        X_lately = X[-forcastedPeriod:]
        # select all rows from the dataframe from the past until <forcastedPeriod> days
        X = X[:-forcastedPeriod]

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
              ['Open',  'High',  'Low',  'Close', 'Volume', 'Market Cap'], 'Close', Strategy.All)
