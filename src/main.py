import csv
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')


# class Model:
#     _dataframe: pd.DataFrame = None
#     _forecastColumn: str = None
#     _forecastNumber: int = None
#     _X: np.array = None
#     _y: np.array = None
#     _X_lately: np.array = None
#     _X_train: list = None
#     _X_test: list = None
#     _y_train: list = None
#     _y_test: list = None
#     _confidence: float = None

#     def __init__(self, pathToFileWithHeader):
#         pass


def createDataframe(pathToFileWithHeader: str) -> pd.DataFrame:
    with open(pathToFileWithHeader, 'r') as file:
        header = list(file.readline().split(','))
        header = [x.replace(' ', '').rstrip() for x in header]
    return pd.read_csv(pathToFileWithHeader, names=header, skiprows=1)


def setUp():
    path = "./dataset/bitcoin_price_20170101_20191129.csv"
    df = createDataframe(path)
    regressionModel(*prepareTrainingSets(*prepareDataframe(df)))


def prepareDataframe(df: pd.DataFrame):
    df = df[['Open',  'High',  'Low',  'Close', 'Volume', 'MarketCap']]

    df['HighLowPercentage'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PercentageChange'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HighLowPercentage',
             'PercentageChange', 'Volume', 'MarketCap']]

    forecastColumn = 'Close'

    # fill NA values with an obvious different value that will not be taken in consideration
    df.fillna(value=-99999, inplace=True)

    # number of days in the future
    forecastNumber = int(math.ceil(0.01 * len(df)))

    # to predict in the future
    df['label'] = df[forecastColumn].shift(-forecastNumber)

    return df, forecastNumber


def prepareTrainingSets(df: pd.DataFrame, forecastNumber: int):
    X = np.array(df.drop(['label'], 1))
    # scale the data set
    X = preprocessing.scale(X)
    # select all rows from the dataframe from forecastNumber days ago
    # since we have the date in descending order we will select the first forecastNumber entiries
    X_lately = X[-forecastNumber:]
    # X_lately = X[:forecastNumber]

    # select all rows from the dataframe from the past until forecastNumber days
    X = X[:-forecastNumber]
    # X = X[forecastNumber:]

    df.dropna(inplace=True)
    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    return df, X_train, X_test, y_train, y_test, X_lately


def regressionModel(df, X_train, X_test, y_train, y_test, X_lately):
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)
    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    # last_date = df.iloc[-1]
    last_date = datetime.datetime(2019, 11, 28)
    print(last_date)
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


setUp()
