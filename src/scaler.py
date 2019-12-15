from .types.scalerType import ScalerType
from .helpers.scalerHelper import ScalerHelper
from pandas import DataFrame
from numpy import ndarray
from typing import Union


class Scaler:
    _scaler = None

    def __init__(self, scalerType: ScalerType, data: Union[ndarray, DataFrame], loadScaler: bool = False,
                 pathToScaler: str = "./AppData/scaler.save", saveScaler: bool = True):
        if loadScaler is False:
            self._scaler = ScalerHelper.getScaler(scalerType)
            self.fitScaler(data)
            if saveScaler:
                ScalerHelper.saveScaler(self._scaler, pathToScaler)
        else:
            self._scaler = ScalerHelper.loadScaler(pathToScaler)

    def fitScaler(self, data: Union[ndarray, DataFrame]):
        ScalerHelper.fit(self._scaler, data)

    def transform(self, data: Union[ndarray, DataFrame]):
        if isinstance(data, ndarray):
            return ScalerHelper.scaleArray(self._scaler, data)
        else:
            return ScalerHelper.scaleDataset(self._scaler, data)