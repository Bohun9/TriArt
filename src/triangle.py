import numpy as np
from thread_local_data import thread_local
from bounding_box import BoundingBox
import target_image

TRIANGLE_SAMPLE_SIZE = 50
PROBABILITY_AVERAGE_SAMPLE = 0.5
COLOR_PERTURBATION = 30
HEIGHT_START_PERTURBATION = 80
HEIGHT_FINAL_PERTURBATION = 40
WIDTH_START_PERTURBATION = 80
WIDTH_FINAL_PERTURBATION = 40
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
    def random_triangle(cls):
        points = np.empty((3, 2), dtype=np.int64)
        points[0][0] = thread_local.rng.integers(0, target_image.width)
        points[0][1] = thread_local.rng.integers(0, target_image.height)
        points[1] = points[0] + (thread_local.rng.integers(-50, 50), thread_local.rng.integers(-50, 50))
        points[2] = points[0] + (thread_local.rng.integers(-50, 50), thread_local.rng.integers(-50, 50))
        color = Triangle.generate_color(points)
        return cls(points, color)

    def __repr__(self):
        return f"Triangle(points={self.points}, color={self.color})"

    @staticmethod
    def compute_current_perturbation(evolution_point, a, b):
        if thread_local.rng.random() < 0.5:
            return 400
        else:
            return 50
        # assert b <= a
        # assert (0.0 <= evolution_point and evolution_point <= 1.0)
        # res = int(a * (b / a) ** evolution_point)
        # assert (b <= res and res <= a)
        # return res

    @staticmethod
    def project_into_canvas(points, height, width):
        x_low, x_high = 0 - X_PADDING, width + X_PADDING
        y_low, y_high = 0 - Y_PADDING, height + Y_PADDING
        points[0] = project_into_range(points[0], x_low, x_high)
        points[1] = project_into_range(points[1], y_low, y_high)

    def adjust_vertices(self, evolution_point):
        height_perturbation = Triangle.compute_current_perturbation(evolution_point, HEIGHT_START_PERTURBATION, HEIGHT_FINAL_PERTURBATION)
        width_perturbation = Triangle.compute_current_perturbation(evolution_point, WIDTH_START_PERTURBATION, WIDTH_FINAL_PERTURBATION)
        height_delta = thread_local.rng.integers(0, height_perturbation)
        width_delta = thread_local.rng.integers(0, width_perturbation)
        index = thread_local.rng.integers(0, 3)
        dx = thread_local.rng.integers(-width_delta, width_delta + 1)
        dy = thread_local.rng.integers(-height_delta, height_delta + 1)
        self.points[index] += (dx, dy)
        Triangle.project_into_canvas(self.points[index], target_image.height, target_image.width)

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

    @staticmethod
    def average_color_of_sample(points):
        rnd_points = Triangle.random_points_inside_triangle(points, TRIANGLE_SAMPLE_SIZE)
        for i in range(TRIANGLE_SAMPLE_SIZE):
            rnd_points[i][0] = project_into_range(rnd_points[i][0], 0, target_image.width - 1)
            rnd_points[i][1] = project_into_range(rnd_points[i][1], 0, target_image.height - 1)
        return Triangle.average_color_of_points(rnd_points)

    @staticmethod
    def generate_color(points):
        return Triangle.color_perturbation(Triangle.average_color_of_sample(points))

    @staticmethod
    def color_perturbation(color):
        delta = thread_local.rng.integers(0, COLOR_PERTURBATION)
        for i in range(3):
            if thread_local.rng.random() < 0.5:
                continue
            color[i] += thread_local.rng.integers(-delta, delta + 1)
            color[i] = project_into_range(color[i], 0, 255)
        return color

    def change_color(self):
        if thread_local.rng.random() < PROBABILITY_AVERAGE_SAMPLE:
            self.color = Triangle.generate_color(self.points)
        else:
            Triangle.color_perturbation(self.color)

    def bounding_box(self):
        return BoundingBox((self.points[:, 0].min(), self.points[:, 1].min()), (self.points[:, 0].max(), self.points[:, 1].max()))

