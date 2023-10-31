from abc import ABC, abstractclassmethod
from typing import List


class BaseLikelihood(ABC):
    @abstractclassmethod
    def fout(self, y : List[float], w : List, V : List[float]) -> List[float]:
        pass

    @abstractclassmethod
    def dwfout(self, y : List[float], w : List, V : List[float]) -> List[float]:
        pass

    @abstractclassmethod
    def channel(self, y : List[float], w : List, V : List[float]):
        pass
    