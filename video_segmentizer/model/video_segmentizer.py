from .pixel_process import PixelProcess
import time
import numpy as np


class VideoSegmentizer():

    _image_model = None

    def __init__(self, width, height):

        self._image_model = [[PixelProcess(3)] * width] * height

    def fit(self, image):

        start = time.time()
        for row_idx, row in enumerate(self._image_model):
            for col_idx, pixel_process in enumerate(row):
                pixel_process.fit(image[row_idx][col_idx])

        stop = time.time()
        print(stop-start)

    def classify_image(self, image):
        pass
