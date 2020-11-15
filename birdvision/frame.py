import cv2
import numpy as np


class Frame:
    """
    Represents a single frame of FFT.

    There are many potential processing steps that happen when examining a frame, and this class lazily computes them
    when called upon.
    """

    def __init__(self, color_image):
        self.color = color_image
        self._gray = None
        self._gray_min = None
        self._gray_max = None

    @property
    def gray(self):
        if self._gray is not None:
            return self._gray
        self._gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def gray_min(self):
        if self._gray_min is not None:
            return self._gray_min
        self._gray_min = np.min(self.color, axis=2)
        return self._gray_min

    @property
    def gray_max(self):
        if self._gray_max is not None:
            return self._gray_max
        self._gray_max = np.max(self.color, axis=2)
        return self._gray_max
