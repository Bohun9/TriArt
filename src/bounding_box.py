import numpy as np


class BoundingBox:
    def __init__(self, c1, c2):
        self.points = np.array([c1, c2], dtype=np.int64)

    def unite(self, box):
        self.points[0] = np.minimum(self.points[0], box.points[0])
        self.points[1] = np.maximum(self.points[1], box.points[1])

    def intersect(self, box):
        self.points[0] = np.maximum(self.points[0], box.points[0])
        self.points[1] = np.minimum(self.points[1], box.points[1])

    def height(self):
        return self.points[1][1] - self.points[0][1]

    def width(self):
        return self.points[1][0] - self.points[0][0]

    def corner1(self):
        return self.points[0]

    def corner2(self):
        return self.points[1]

    def get_region_of_image(self, image):
        return image[
            self.points[0][1] : self.points[1][1],
            self.points[0][0] : self.points[1][0],
            :,
        ]

    def __isub__(self, other):
        self.points[0] -= other
        self.points[1] -= other
        return self

    def is_empty(self):
        return self.height() <= 0 or self.width() <= 0

    def area(self):
        return self.height() * self.width()
