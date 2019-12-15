# Bitcoin Price Predictor - MachineLearning

To get the dataset please visit and run the code found [here](https://github.com/IulianOctavianPreda/CryptoCurrencyHistoricalDataRetriever)

# Introduction

This application is developed using python 3+. To be able to run it you need to install the packages from [requirements.txt](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/blob/master/requirements.txt). You can install them using:

`$ pip install -r requirements.txt`

These packages are:

-   Pandas
-   Numpy
-   Matplotlib
-   Scikit-learn
-   Joblib

# Architecture

The code is split in multiple packages:

-   main - the main entry point in the application
-   src - package containing the main classes used by the application
-   types - contains the types used by the scaler and models
-   helpers - is the package that contains static methods useful for the classes found in the src package
-   demo - contains the dataset and demo class that can be used for presentations

# Demo

The demo class will load all the data from the demo dataset. It will create multiple models using that data and after training them, the result accuracies will be compared. The dataset will be printed in the console and the results of each model as well as the comparison can be found in the folder [Figures](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/tree/master/Figures) as generated plots.

# Application flow

-   The CSV dataset will be loaded in the object [Dataset](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/tree/master/src/dataset.py)
-   Using this object the test sets will be generated and then scaled using the [Scaler](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/tree/master/src/scaler.py)
-   Then a [Model](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/tree/master/src/model.py) is selected and trained.
-   To predict future values it is necessary to load a second dataset and use the predict function of the trained model.

By default the trained model and the fitted scaler are saved in the [AppData](https://github.com/IulianOctavianPreda/BitcoinPricePredictor-MachineLearning/tree/master/AppData) folder.

If a model and scaler are already saved they can be loaded directly without needing a the training dataset.
