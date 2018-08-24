from .model import RGBPixelProcess
from multiprocessing import Pool


class Segmentizer():

    _image_model = None
    _last_orig_image = None
    _width = None
    _height = None

    def __init__(self, width, height):

        self._image_model = [[RGBPixelProcess(n_clusters=3)] * width] * height
        self._background = [[0] * width] * height
        self._width = width
        self._height = height

    def _process_row(self, x):
        row_idx, row = x
        return list(map(lambda y: y[1].fit(self._last_orig_image[row_idx][y[0]]), enumerate(row)))

    def fit(self, image):
        #self._last_orig_image = image

        #pool = Pool(10)
        #self._image_model = list(pool.map(self._process_row, enumerate(self._image_model)))
        #pool.close()

        for i in range(self._height):
            for j in range(self._width):
                self._image_model[i][j].fit(image[i][j])


    def classify_image(self, image):

        background = []

        for i in range(self._height):
            row = []
            for j in range(self._width):
                row.append(1.0 if self._image_model[i][j].is_background_pixel(image[i][j]) else 0)

            background.append(row)

        return background
