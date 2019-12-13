from matplotlib import pyplot


class PlotterHelper:

    @staticmethod
    def plot_evaluations(results: list, names: list):
        pyplot.boxplot(results, labels=names)
        pyplot.title('Algorithm Comparison')
        pyplot.show()

    @staticmethod
    def standard_plot(values: list, x_label: str, y_label: str):
        pyplot.plot(values)
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        pyplot.show()
