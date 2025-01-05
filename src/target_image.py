import numpy as np

image = np.empty((1, 1, 3), dtype=np.uint8)
height = 0
width = 0
alpha = 0.0

def set_target_image(target_image, target_alpha):
    global image, height, width, alpha
    image = target_image
    height, width, _ = target_image.shape
    alpha = target_alpha
