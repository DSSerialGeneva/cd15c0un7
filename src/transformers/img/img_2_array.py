import numpy as np


class Img2Array:

    @staticmethod
    def transform(img):
        # * 1. to have float instead of int64...
        return np.array(img) * 1.
