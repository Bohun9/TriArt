import numpy as np
from . import target_image
from .thread_local_data import thread_local
from .bounding_box import BoundingBox

TRIANGLE_SAMPLE_SIZE = 50
PROBABILITY_AVERAGE_SAMPLE = 0.7
X_PADDING = 100
Y_PADDING = 100

def project_into_range(x, a, b):
    assert a <= b
    if x < a:
        return a
    if x > b:
        return b
    return x

class Triangle:
    def __init__(self, points, color):
        self.points = points
        self.color = color

    @classmethod
    def random_triangle(cls, focus_mode, target_pixels):
        points = np.empty((3, 2), dtype=np.int64)
        if target_pixels is not None:
            index = thread_local.rng.integers(0, target_pixels.shape[0])
            points[0] = target_pixels[index]
            # assert focus_mode
        else:
            points[0][0] = thread_local.rng.integers(0, target_image.width)
            points[0][1] = thread_local.rng.integers(0, target_image.height)
        d = thread_local.rng.choice([10, 50], p=[0.5, 0.5]) if focus_mode else 50
        points[1] = points[0] + (thread_local.rng.integers(-d, d), thread_local.rng.integers(-d, d))
        points[2] = points[0] + (thread_local.rng.integers(-d, d), thread_local.rng.integers(-d, d))
        color = Triangle.generate_color(points, focus_mode, target_pixels)
        triangle = cls(points, color)
        triangle.fix_orientation()
        return triangle 

    def __repr__(self):
        points = ', '.join(f'({x}, {y})' for x, y in self.points)
        return f"Triangle(points={points}, color={self.color})"

    def fix_orientation(self):
        u = self.points[1] - self.points[0]
        v = self.points[2] - self.points[0]
        if u[0] * v[1] - u[1] * v[0] < 0:
            self.points[0], self.points[1] = self.points[1], self.points[0]

    @staticmethod
    def project_into_canvas(points, height, width):
        x_low, x_high = 0 - X_PADDING, width + X_PADDING
        y_low, y_high = 0 - Y_PADDING, height + Y_PADDING
        points[0] = project_into_range(points[0], x_low, x_high)
        points[1] = project_into_range(points[1], y_low, y_high)

    def adjust_vertices(self, focus_mode, target_pixels):
        d = thread_local.rng.choice([10, 50], p=[0.5, 0.5]) if focus_mode else thread_local.rng.choice([50, 500], p=[0.5, 0.5])
        index = thread_local.rng.integers(0, 3)
        dx = thread_local.rng.integers(-d, d + 1)
        dy = thread_local.rng.integers(-d, d + 1)
        self.points[index] += (dx, dy)
        Triangle.project_into_canvas(self.points[index], target_image.height, target_image.width)
        self.fix_orientation()

    def vertices(self):
        return self.points.copy()

    @staticmethod
    def random_point_inside_triangle(points):
        a = thread_local.rng.random()
        b = thread_local.rng.random()
        if a + b > 1:
            a = 1 - a
            b = 1 - b
        c = 1 - a - b
        x = a * points[0][0] + b * points[1][0] + c * points[2][0]
        y = a * points[0][1] + b * points[1][1] + c * points[2][1]
        return (np.int32(x), np.int32(y))

    @staticmethod
    def random_points_inside_triangle(points, n):
        return np.array([Triangle.random_point_inside_triangle(points) for _ in range(n)])

    @staticmethod
    def average_color_of_points(points):
        n = points.shape[0]
        colors = np.zeros((n, 3))
        for i in range(n):
            colors[i] = target_image.image[points[i][1]][points[i][0]]
        average = np.average(colors, axis=0)
        assert (average.shape == (3,))
        return average

    def average_color_of_sample(self):
        return Triangle.compute_average_color_of_sample(self.points)

    @staticmethod
    def compute_average_color_of_sample(points):
        rnd_points = Triangle.random_points_inside_triangle(points, TRIANGLE_SAMPLE_SIZE)
        for i in range(TRIANGLE_SAMPLE_SIZE):
            rnd_points[i][0] = project_into_range(rnd_points[i][0], 0, target_image.width - 1)
            rnd_points[i][1] = project_into_range(rnd_points[i][1], 0, target_image.height - 1)
        return Triangle.average_color_of_points(rnd_points)

    @staticmethod
    def generate_color(points, focus_mode, target_pixels):
        return Triangle.color_perturbation(Triangle.compute_average_color_of_sample(points), focus_mode, target_pixels)

    @staticmethod
    def color_perturbation(color, focus_mode, target_pixels):
        d = 10 if focus_mode else 30
        for i in range(3):
            if thread_local.rng.random() < 0.5:
                continue
            color[i] += thread_local.rng.integers(-d, d + 1)
            color[i] = project_into_range(color[i], 0, 255)
        return color

    def change_color(self, focus_mode, target_pixels):
        if thread_local.rng.random() < PROBABILITY_AVERAGE_SAMPLE:
            self.color = Triangle.generate_color(self.points, focus_mode, target_pixels)
        else:
            Triangle.color_perturbation(self.color, focus_mode, target_pixels)

    def bounding_box(self):
        return BoundingBox((self.points[:, 0].min(), self.points[:, 1].min()), (self.points[:, 0].max(), self.points[:, 1].max()))

