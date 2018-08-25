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

    def fit(self, image):

        for i in range(self._height):
            for j in range(self._width):
                self._image_model[i][j].fit(image[i][j].tolist())


    def classify_image(self, image):

        background = []

        for i in range(self._height):
            row = []
            for j in range(self._width):
                row.append(0 if self._image_model[i][j].is_background_pixel(image[i][j].tolist()) else 1)

            background.append(row)

        return background
