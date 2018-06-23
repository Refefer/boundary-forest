from collections import defaultdict

import numpy as np

class Estimator(object):
    def scorer(self, scores):
        raise NotImplementedError()
            
class ShepardClassifier(Estimator):
    def score(self, scores):
        classes = defaultdict(float)
        denom = 0.0
        for dist, cls in scores:
            weight = 1. / (dist + 1e-6)
            classes[cls] += weight
            denom += weight

        score, cls = max((v / denom, k) for k, v in classes.iteritems())
        return cls, score

class ShepardRegressor(Estimator):
    def score(self, scores):
        score = 0.0
        denom = 0.0
        for dist, cls in scores:
            weight = 1. / (dist + 1e-6)
            score += cls * weight
            denom += weight

        return score / denom, 1

class UniformRegressor(Estimator):
    def score(self, scores):
        return sum(cls for _, cls in scores) / float(len(scores)), 1

