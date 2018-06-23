class LabelDistance(object):
    def compare(self, cx, cy):
        raise NotImplementedError()

class Equal(LabelDistance):
    def compare(self, cx, cy):
        return cx == cy

class Index(LabelDistance):
    def compare(self, cx, cy):
        return False

class ThresholdDist(LabelDistance):
    def __init__(self, threshold):
        self.threshold = threshold

    def compare(self, cx, cy):
        return abs(cx - cy) < self.threshold

