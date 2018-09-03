import numpy as np
from scipy.spatial.distance import cdist

class Distance(object):
    def compare(self, x, y):
        raise NotImplementedError()

    def batch_compare(self, x, ys):
        return [self.compare(x, y) for y in ys]

class LambdaMetric(Distance):
    def __init__(self, f):
        self.f = f

    def compare(self, x, y):
        return self.f(x, y)

class Euclidean(Distance):
    def compare(self, x, y):
        if isinstance(x, np.ndarray):
            #return cdist(x, y)
            return ((x - y) ** 2).sum()

        return (x - y).power(2).sum()

class Cosine(Distance):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def norm(self, x):
        if not isinstance(x, np.ndarray):
            x = x.data
        
        return np.linalg.norm(x)

    def compare(self, x, y):
        denom = self.norm(x) * self.norm(y) if self.normalize else 1.
        return 2 - x.dot(y.T).sum() / denom

