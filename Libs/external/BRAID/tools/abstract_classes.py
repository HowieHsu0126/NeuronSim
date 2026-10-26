
"""Abstract classes used to standardize predictor models"""

from abc import ABC, abstractmethod


class PredictorModel(ABC):
    @abstractmethod
    def predict(self, Y, U=None):
        pass
