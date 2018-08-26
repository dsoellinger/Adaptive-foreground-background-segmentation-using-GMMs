from .model import RGBPixelProcess

class Segmentizer:

    def __init__(self, width, height):

        self._image_model = []
        for h in range(height):
            row = []
            for r in range(width):
                row.append(RGBPixelProcess(3))

            self._image_model.append(row)

        self._width = width
        self._height = height

    def fit_and_predict(self, image):

        background = []

        for i in range(self._height):
            row = []
            for j in range(self._width):
                x = image[i][j].astype('float64')
                self._image_model[i][j].fit(x)
                row.append(0 if self._image_model[i][j].is_background_pixel(x) else 1)

            background.append(row)

        return background


    def classify_image(self, image):

        background = []

        for i in range(self._height):
            row = []
            for j in range(self._width):
                row.append(0 if self._image_model[i][j].is_background_pixel(image[i][j].astype('float64')) else 1)

            background.append(row)

        return background
