from enum import Enum


class ModelType(Enum):
    LinearRegression = 0
    BayesianRidge = 1
    LassoLars = 2
    ARDRegression = 3
    # PassiveAggressiveRegressor = 4
    TheilSenRegressor = 5
    # DecisionTreeRegressor = 6
