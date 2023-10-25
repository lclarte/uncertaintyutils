import numpy as np
from scipy.optimize import minimize_scalar
import gcmpyo3

from . import base_likelihood

def proximal_operator(func : callable, x : float, tau : float) -> float:
    to_minimize = lambda z : ((z - x)**2) / (2 * tau) + func(z)
    res = minimize_scalar(to_minimize, method='Golden')
    if res['x'] > 1e10:
        print(res['x'])
    return res['x']

class ERMLogitLikelihood(base_likelihood.BaseLikelihood):
    def __init__(self) -> None:
        super().__init__()
        self.likelihood = gcmpyo3.ERMLogistic()

    def fout(self, y, w, V):
        return self.likelihood.call_f0(y, w, V)

    def dwfout(self, y, w, V):
        return self.likelihood.call_df0(y, w, V)

    def channel(self, y, w, V):
        return [self.fout(y_, w_, v_) for y_, w_, v_ in zip(y, w, V)], [self.dwfout(y_, w_, v_) for y_, w_, v_ in zip(y, w, V)]

    # EQUATIONS FOR STATE EVOLUTION 