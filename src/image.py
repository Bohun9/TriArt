import cv2 as cv
import numpy as np
from numba import njit
from . import target_image
from .bounding_box import BoundingBox


@njit
def cross_product(u, v):
    return u[0] * v[1] - u[1] * v[0]


@njit
def orientation(a, b, c):
    return cross_product(b - a, c - a)


@njit
def rasterise_triangle(image, points, color, c1, c2, alpha):
    assert orientation(points[0], points[1], points[2]) >= 0
    for y in range(c1[1], c2[1]):
        for x in range(c1[0], c2[0]):
            p = np.array((x, y), dtype=np.int64)
            o1 = orientation(points[0], points[1], p)
            o2 = orientation(points[1], points[2], p)
            o3 = orientation(points[2], points[0], p)
            if o1 >= 0 and o2 >= 0 and o3 >= 0:
                image[y][x] = (1 - alpha) * image[y][x] + alpha * color


@njit
def rasterise_triangle2(image, points, color, c1, c2, alpha):
    a01 = points[0][1] - points[1][1]
    a12 = points[1][1] - points[2][1]
    a20 = points[2][1] - points[0][1]

    b01 = points[1][0] - points[0][0]
    b12 = points[2][0] - points[1][0]
    b20 = points[0][0] - points[2][0]

    f01_row = orientation(points[0], points[1], c1)
    f12_row = orientation(points[1], points[2], c1)
    f20_row = orientation(points[2], points[0], c1)

    x_min, y_min = c1[0], c1[1]
    x_max, y_max = c2[0], c2[1]

    for y in range(y_min, y_max):
        f01 = f01_row
        f12 = f12_row
        f20 = f20_row

        for x in range(x_min, x_max):
            if f01 | f12 | f20 >= 0:
                image[y][x] = (1 - alpha) * image[y][x] + alpha * color

            f01 += a01
            f12 += a12
            f20 += a20

        f01_row += b01
        f12_row += b12
        f20_row += b20


def compute_squared_error(image1, image2):
    # return ((image1.astype(np.int64) - image2.astype(np.int64)) ** 2).sum()
    a = image1 - image2
    b = 254 * (image1 < image2).astype(np.uint8) + 1
    c = a * b
    return (c.astype(np.uint16) ** 2).sum()


def absolute_difference(image1, image2):
    a = image1 - image2
    b = 254 * (image1 < image2).astype(np.uint8) + 1
    return a * b


@njit
def compute_target_pixels(image, target_image):
    height, width, _ = image.shape
    diff = image.astype(np.int32) - target_image.astype(np.int32)
    diff_sum = np.zeros((height + 1, width + 1, 3), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            diff_sum[y + 1][x + 1] = (
                diff[y][x] + diff_sum[y + 1][x] + diff_sum[y][x + 1] - diff_sum[y][x]
            )

    eps = 5
    d = 4
    step = 3

    new_height = height // step
    new_width = width // step

    pixel_score = np.zeros((new_height * new_width), dtype=np.int32)

    for yi in range(new_height):
        for xi in range(new_width):
            id = yi * new_width + xi
            x = step * xi
            y = step * yi
            x0, x1 = max(0, x - d), min(width, x + d)
            y0, y1 = max(0, y - d), min(height, y + d)

            sum = (
                diff_sum[y1][x1]
                - diff_sum[y0][x1]
                - diff_sum[y1][x0]
                + diff_sum[y0][x0]
            )

            r_score = (
                eps**2 * (x1 - x0) * (y1 - y0)
                - 2 * np.sign(diff[y][x][2]) * eps * sum[2]
            )
            g_score = (
                eps**2 * (x1 - x0) * (y1 - y0)
                - 2 * np.sign(diff[y][x][1]) * eps * sum[1]
            )
            b_score = (
                eps**2 * (x1 - x0) * (y1 - y0)
                - 2 * np.sign(diff[y][x][0]) * eps * sum[0]
            )

            pixel_score[id] = min(0, r_score) + min(0, g_score) + min(0, b_score)

    sorted_indices = np.argsort(pixel_score)
    res = np.empty((sorted_indices.size, 2), dtype=np.int32)
    for i, id in enumerate(sorted_indices):
        yi = id / new_width
        xi = id % new_width
        res[i][0] = step * xi
        res[i][1] = step * yi

    return res


def paint_points(pixels, background):
    image = background.copy()
    for (x, y) in pixels:
        cv.circle(image, (x, y), 2, (0, 0, 255), -1)
    return image


def paint_edges_of_triangles(triangles, image, color):
    for triangle in triangles:
        cv.line(image, triangle.points[0], triangle.points[1], color, 1)
        cv.line(image, triangle.points[0], triangle.points[2], color, 1)
        cv.line(image, triangle.points[1], triangle.points[2], color, 1)
