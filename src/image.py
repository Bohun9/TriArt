import cv2 as cv
import numpy as np
from numba import njit
from . import target_image
from .bounding_box import BoundingBox

ALPHA = 0.3

def extract_triangle_area(image, p1, p2):
    return image[p1[1]:p2[1], p1[0]:p2[0], :].copy()

def alpha_blend(image, overlay, p1, p2):
    image[p1[1]:p2[1], p1[0]:p2[0], :] = (1 - target_image.alpha) * image[p1[1]:p2[1], p1[0]:p2[0], :] + target_image.alpha * overlay

def paint_triangle(image, triangle, image_origin):
    height, width, _ = image.shape
    bounding_box = triangle.bounding_box()
    bounding_box -= image_origin
    bounding_box.intersect(BoundingBox((0, 0), (width, height)))
    if bounding_box.is_empty():
        return image
    vertices = triangle.vertices()
    vertices -= bounding_box.corner1() + image_origin
    triangle_image = extract_triangle_area(image, bounding_box.corner1(), bounding_box.corner2())
    cv.fillPoly(triangle_image, [vertices], triangle.color)
    alpha_blend(image, triangle_image, bounding_box.corner1(), bounding_box.corner2())

def paint_triangles(triangles, height, width, image_origin=(0, 0), background=None):
    image = np.zeros((height, width, 3), dtype=np.uint8) if background is None else background.copy()
    for triangle in triangles:
        paint_triangle(image, triangle, image_origin)
    return image

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

def compute_target_pixels(image, target_image):
    height, width, _ = image.shape
    diff = image.astype(np.int32) - target_image.astype(np.int32)
    pixel_score = np.zeros((height, width), dtype=np.int32)
    eps = 5
    d = 4
    step = 2

    def square_coordinates(x, y):
        x0, x1 = max(0, x - d), min(width, x + d)
        y0, y1 = max(0, y - d), min(height, y + d)
        return ((x0, y0), (x1, y1))

    for y in range(0, height, step):
        for x in range(0, width, step):
            x0, x1 = max(0, x - d), min(width, x + d)
            y0, y1 = max(0, y - d), min(height, y + d)
            square = diff[y0:y1, x0:x1, :]
            r_square = square[:, :, 2]
            g_square = square[:, :, 1]
            b_square = square[:, :, 0]
            r_score = min(((r_square + eps) ** 2).sum(), ((r_square - eps) ** 2).sum()) - ((r_square) ** 2).sum()
            g_score = min(((g_square + eps) ** 2).sum(), ((g_square - eps) ** 2).sum()) - ((g_square) ** 2).sum()
            b_score = min(((b_square + eps) ** 2).sum(), ((b_square - eps) ** 2).sum()) - ((b_square) ** 2).sum()
            pixel_score[y][x] = min(0, r_score) + min(0, g_score) + min(0, b_score)

    sorted_indices = np.argsort(pixel_score.ravel())
    y, x = np.unravel_index(sorted_indices, pixel_score.shape)
    res = np.column_stack((x, y))

    # res_yx = np.column_stack((y, x))
    # (x0, y0), (x1, y1) = square_coordinates(*res[0])
    # print(res_yx[:10])
    # print(pixel_score[res_yx[:10]])
    # for i in range(100):
    #     y, x = res_yx[i]
    #     print(pixel_score[y][x], end=" ")
    # print("")
    # print("best")
    # print(res[0])
    # print(diff[y0:y1, x0:x1, :])

    return res

def compute_target_pixels2(image, target_image):
    height, width, _ = image.shape
    diff = image.astype(np.int32) - target_image.astype(np.int32)
    sum = np.cumsum(np.cumsum(diff, axis=0), axis=1)
    diff_sum = np.zeros((sum.shape[0] + 1, sum.shape[1] + 1, 3), dtype=sum.dtype)
    diff_sum[1:, 1:, :] = sum
    pixel_score = np.zeros((height, width), dtype=np.int32)

    def subsquare_sum(x0, y0, x1, y1):
        return diff_sum[y1][x1] - diff_sum[y0][x1] - diff_sum[y1][x0] + diff_sum[y0][x0]

    eps = 5
    d = 4
    step = 3
    area = (d ** 2) * (eps ** 2)

    for y in range(0, height, step):
        for x in range(0, width, step):
            x0, x1 = max(0, x - d), min(width, x + d)
            y0, y1 = max(0, y - d), min(height, y + d)
            # square = diff[y0:y1, x0:x1, :]
            # r_square = square[:, :, 2]
            # g_square = square[:, :, 1]
            # b_square = square[:, :, 0]

            sum = subsquare_sum(x0, y0, x1, y1)

            # assert (sum[2] == r_square.sum())
            # assert (sum[1] == g_square.sum())
            # assert (sum[0] == b_square.sum())

            r_score = area - 2 * np.sign(diff[y][x][2]) * eps * sum[2]
            g_score = area - 2 * np.sign(diff[y][x][1]) * eps * sum[1]
            b_score = area - 2 * np.sign(diff[y][x][0]) * eps * sum[0]

            # r_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][2]) * eps * r_square.sum()
            # g_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][1]) * eps * g_square.sum()
            # b_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][0]) * eps * b_square.sum()

            # r_score = ((r_square + (np.sign(diff[y][x][2]) * eps)) ** 2).sum() - ((r_square) ** 2).sum()
            # g_score = ((g_square + (np.sign(diff[y][x][1]) * eps)) ** 2).sum() - ((g_square) ** 2).sum()
            # b_score = ((b_square + (np.sign(diff[y][x][0]) * eps)) ** 2).sum() - ((b_square) ** 2).sum()
            pixel_score[y][x] = min(0, r_score) + min(0, g_score) + min(0, b_score)

    sorted_indices = np.argsort(pixel_score.ravel())
    y, x = np.unravel_index(sorted_indices, pixel_score.shape)
    res = np.column_stack((x, y))

    return res

def compute_target_pixels3(image, target_image):
    height, width, _ = image.shape
    diff = image.astype(np.int32) - target_image.astype(np.int32)
    sum = np.cumsum(np.cumsum(diff, axis=0), axis=1)
    diff_sum = np.zeros((sum.shape[0] + 1, sum.shape[1] + 1, 3), dtype=sum.dtype)
    diff_sum[1:, 1:, :] = sum

    eps = 5
    d = 4
    step = 3
    const = (d ** 2) * (eps ** 2)

    pixel_score = np.zeros((height // step, width // step), dtype=np.int32)

    for yi in range(0, height // step):
        for xi in range(0, width // step):
            x = step * xi
            y = step * yi
            x0, x1 = max(0, x - d), min(width, x + d)
            y0, y1 = max(0, y - d), min(height, y + d)

            sum = diff_sum[y1][x1] - diff_sum[y0][x1] - diff_sum[y1][x0] + diff_sum[y0][x0]

            r_score = const - 2 * np.sign(diff[y][x][2]) * eps * sum[2]
            g_score = const - 2 * np.sign(diff[y][x][1]) * eps * sum[1]
            b_score = const - 2 * np.sign(diff[y][x][0]) * eps * sum[0]

            # r_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][2]) * eps * r_square.sum()
            # g_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][1]) * eps * g_square.sum()
            # b_score = (d ** 2) * (eps ** 2) - 2 * np.sign(diff[y][x][0]) * eps * b_square.sum()

            # r_score = ((r_square + (np.sign(diff[y][x][2]) * eps)) ** 2).sum() - ((r_square) ** 2).sum()
            # g_score = ((g_square + (np.sign(diff[y][x][1]) * eps)) ** 2).sum() - ((g_square) ** 2).sum()
            # b_score = ((b_square + (np.sign(diff[y][x][0]) * eps)) ** 2).sum() - ((b_square) ** 2).sum()
            pixel_score[yi][xi] = min(0, r_score) + min(0, g_score) + min(0, b_score)

    sorted_indices = np.argsort(pixel_score.ravel())
    y, x = np.unravel_index(sorted_indices, pixel_score.shape)
    res = np.column_stack((step * x, step * y))

    return res

@njit
def compute_target_pixels4(image, target_image):
    height, width, _ = image.shape
    diff = image.astype(np.int32) - target_image.astype(np.int32)
    diff_sum = np.zeros((height + 1, width + 1, 3), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            diff_sum[y + 1][x + 1] = diff[y][x] + diff_sum[y + 1][x] + diff_sum[y][x + 1] - diff_sum[y][x]

    eps = 5
    d = 4
    step = 3
    const = (d ** 2) * (eps ** 2)

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

            sum = diff_sum[y1][x1] - diff_sum[y0][x1] - diff_sum[y1][x0] + diff_sum[y0][x0]

            r_score = const - 2 * np.sign(diff[y][x][2]) * eps * sum[2]
            g_score = const - 2 * np.sign(diff[y][x][1]) * eps * sum[1]
            b_score = const - 2 * np.sign(diff[y][x][0]) * eps * sum[0]

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
        cv.circle(image, (x, y), 3, (0,0,255), -1)
        # image[y][x] = (0, 0, 255)
    return image

def paint_edges_of_triangles(triangles, image, color):
    for triangle in triangles:
        cv.line(image, triangle.points[0], triangle.points[1], color, 1)
        cv.line(image, triangle.points[0], triangle.points[2], color, 1)
        cv.line(image, triangle.points[1], triangle.points[2], color, 1)
