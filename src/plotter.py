from matplotlib import pyplot
from pandas import DataFrame
from pandas.plotting import scatter_matrix, register_matplotlib_converters
from matplotlib import style
style.use('ggplot')


class Plotter:
    _dataset: DataFrame

    def __init__(self, dataset: DataFrame):
        self._dataset = dataset
        register_matplotlib_converters()

    def boxPlot(self):
        self._dataset.plot(kind='box', vert=True)
        pyplot.title('Box plot')
        pyplot.xticks(rotation='vertical')
        pyplot.show()

    def histogram(self, column: str):
        self._dataset.hist(column=column, bins=3)
        pyplot.title('Histogram for column ' + column)
        pyplot.show()

    def scatterMatrix(self):
        scatter_matrix(self._dataset, figsize=(20, 20))
        pyplot.show()
