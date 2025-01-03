import cv2 as cv
import numpy as np
# from numba import njit, prange
from .bounding_box import BoundingBox

ALPHA = 0.6

def extract_triangle_area(image, p1, p2):
    return image[p1[1]:p2[1], p1[0]:p2[0], :].copy()

# @njit(parallel=True, fastmath=True)
# def extract_triangle_area(image, p1, p2):
#     height = p2[1] - p1[1]
#     width = p2[0] - p1[0]
#     area = np.empty((height, width, 3), dtype=np.uint8)
#     for y in prange(height):
#         for x in range(width):
#             area[y, x] = image[p1[1] + y, p1[0] + x]
#     return area

# @njit(parallel=True, fastmath=True)
def alpha_blend(image, overlay, p1, p2):
    image[p1[1]:p2[1], p1[0]:p2[0], :] = (1 - ALPHA) * image[p1[1]:p2[1], p1[0]:p2[0], :] + ALPHA * overlay

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

# numba breaks this code
# @njit(parallel=True) 
def compute_squared_error(image1, image2):
    # return ((image1.astype(np.int64) - image2.astype(np.int64)) ** 2).sum()
    a = image1 - image2
    b = 254 * (image1 < image2).astype(np.uint8) + 1
    c = a * b
    return (c.astype(np.uint16) ** 2).sum()
