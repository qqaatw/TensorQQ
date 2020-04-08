import numpy as np

class QQInitializer(object):
    def __init__(self, clip_min=None, clip_max=None):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self):
        raise NotImplementedError

class QQNormal(QQInitializer):
    def __init__(self, median=0, stddev=1, clip_min=None, clip_max=None):
        super(QQNormal, self).__init__(clip_min, clip_max)
        self.median = median
        self.stddev = stddev

    def __call__(self, shape):
        init = np.random.normal(self.median, self.stddev, shape)
        if self.clip_min is not None or self.clip_max is not None:
            init = np.clip(init, a_min=self.clip_min, a_max=self.clip_max)
        return init
