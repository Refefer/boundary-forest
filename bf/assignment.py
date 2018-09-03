import numpy as np

class Assignment(object):
    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)

    def set_trees(self, n_trees):
        self.n_trees = n_trees

    def setup(self):
        pass

    def assignments(self):
        raise NotImplementedError()

class All(Assignment):
    def setup(self):
        self.arr = [True] * self.n_trees

    def assignments(self):
        self.arr

class RoundRobin(Assignment):
    def __init__(self, *args, **kwargs):
        super(RoundRobin, self).__init__(*args, **kwargs)
        self.i = 0

    def setup(self):
        self.arr = [False] * self.n_trees
        self.arr[0] = True
    
    def assignments(self):
        prev = self.i
        self.i = (self.i + 1) % self.n_trees
        self.arr[prev] = False
        self.arr[self.i] = True
        return self.arr

class Probability(Assignment):
    def __init__(self, prob, seed):
        super(Probability, self).__init__(seed)
        self.p = prob

    def assignments(self):
        return self.rs.rand(self.n_trees) < self.p

class Subset(Assignment):
    def __init__(self, k_trees, seed):
        super(Subset, self).__init__(seed)
        self.k_trees = k_trees

    def setup(self):
        self.arr = [False] * self.n_trees
        for i in range(self.k_trees):
            self.arr[i] = True

    def assignments(self, n_trees):
        self.rs.shuffle(self.arr)
        return self.arr


