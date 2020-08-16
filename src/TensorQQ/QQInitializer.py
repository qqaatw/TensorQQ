import numpy as np

class QQInitializer:
    """Abstract initializer class.

    Parameters
    ----------
    clip_min : int or float
        If output value is smaller than clip_min, the value will be clipped as clip_min, by default None.
    clip_max : int or float
        If output value is higher than clip_max, the value will be clipped as clip_max, by default None.
    """

    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def __repr__(self):
        text = {
            'name': self.__class__.__name__,
            'attributes': self.__dict__
        }
        return str(text)

    def __call__(self):
        raise NotImplementedError

    def clip(self, x):
        if self.clip_min is not None or self.clip_max is not None:
            return np.clip(x, a_min=self.clip_min, a_max=self.clip_max)
        else:
            return x

class QQNormal(QQInitializer):
    def __init__(self, median=0.0, stddev=1.0, clip_min=None, clip_max=None):
        super().__init__(clip_min, clip_max)
        self.median = median
        self.stddev = stddev

    def __call__(self, shape=None):
        init = np.random.normal(self.median, self.stddev, shape)
        return self.clip(init)

class QQUniform(QQInitializer):
    def __init__(self, scale=0.1, clip_min=None, clip_max=None):
        super().__init__(clip_min=clip_min, clip_max=clip_max)
        self.scale = scale
    
    def __call__(self, shape=None):
        init = np.random.uniform(-self.scale, self.scale, shape)
        return self.clip(init)

class QQZero(QQInitializer):
    def __init__(self, clip_min=None, clip_max=None):
        super().__init__(clip_min=clip_min, clip_max=clip_max)

    def __call__(self, shape=None):
        init = np.zeros(shape)
        return self.clip(init)
