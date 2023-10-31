from . import base_likelihood

class GaussianLogLikelihood(base_likelihood.BaseLikelihood):
    """
    Corresponds to the loss function 
    l(y, z) = 1/2 * (y - z)^2 / noise_variance
    """
    def __init__(self, noise_variance) -> None:
        super().__init__()
        self.noise_variance = noise_variance

    def fout(self, y, w, V):
        """
        D'apres mes calculs, 
        argmin_z ( (z - w)^2 / (2 * v) + (y - z)^2 / (2 * noise_variance) ) = (noise_variance * w + v * y) / (noise_variance + v)
        et f_out(y, w, V) = 1/v * (prox(y, w, V) - w)
        """
        return (y - w) / (self.noise_variance + V)

    def dwfout(self, y, w, V):
        return - 1.0 / (self.noise_variance + V)

    def channel(self, y, w, V):
        return self.fout(y, w, V), self.dwfout(y, w, V)