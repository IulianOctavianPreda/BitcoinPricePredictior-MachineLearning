from pandas import DataFrame, read_csv, to_csv


class DatasetHelper:

    @staticmethod
    def readCsv(self, path: str) -> DataFrame:
        return read_csv(
            path, header=0, index_col='Date', parse_dates=True)

    @staticmethod
    def saveCsv(self, dataset: DataFrame, path: str) -> None:
        dataset.to_csv(path)

    @staticmethod
    def prepareDataset(self, dataset: DataFrame, features: list) -> DataFrame:
        dataset['HighLowPercentage'] = (
            dataset['High'] - dataset['Low']) / dataset['Close'] * 100.0
        dataset['PercentageChange'] = (
            dataset['Close'] - dataset['Open']) / dataset['Open'] * 100.0
        dataset = dataset[['Close', 'HighLowPercentage',
                           'PercentageChange', 'Volume', 'Market Cap']]
        dataset.fillna(value=-99999, inplace=True)
        return dataset
