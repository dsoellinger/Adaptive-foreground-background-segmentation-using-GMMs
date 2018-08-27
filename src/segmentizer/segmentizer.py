from .model import RGBPixelProcess


class Segmentizer:

    def __init__(self, width, height):

        self._image_model = [[RGBPixelProcess(3) for _ in range(width)] for _ in range(height)]
        self._width = width
        self._height = height

    def fit(self, image, init_weight=0.03, init_variance=36.0, lr=0.005):
        for i in range(self._height):
            for j in range(self._width):
                self._image_model[i][j].fit(image[i,j], init_weight, init_variance, lr)
        return image

    def fit_and_predict(self, image, init_weight=0.03, init_variance=36.0, lr=0.005):

        background = []

        for i in range(self._height):
            row = []
            for j in range(self._width):
                x = image[i,j]
                self._image_model[i][j].fit(x, init_weight, init_variance, lr)
                row.append(self._image_model[i][j].is_background_pixel(x))

            background.append(row)

        return background


