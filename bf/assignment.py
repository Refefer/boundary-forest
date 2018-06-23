import numpy as np

class Assignment(object):
    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)

    def assignments(self, n_tress):
        raise NotImplementedError()

class All(Assignment):
    def assignments(self, n_trees):
        return [True] * n_trees

class RoundRobin(Assignment):
    def __init__(self, *args, **kwargs):
        super(RoundRobin, self).__init__(*args, **kwargs)
        self.i = 0

    def assignments(self, n_trees):
        self.i = (self.i + 1) % n_trees
        arr = [False] * n_trees
        arr[self.i] = True
        return arr

class Probability(Assignment):
    def __init__(self, prob, seed):
        super(Probability, self).__init__(seed)
        self.p = prob

    def assignments(self, n_trees):
        return self.rs.rand(n_trees) < self.p

class Subset(Assignment):
    def __init__(self, k_trees, seed):
        super(Subset, self).__init__(seed)
        self.k_trees = k_trees

    def assignments(self, n_trees):
        r = range(n_trees)
        self.rs.shuffle(r)
        arr = [False] * n_trees
        for i in range(self.k_trees):
            arr[r[i]] = True

        return arr


