import sys
import cv2 as cv
import os
import evolution

if len(sys.argv) < 2:
    sys.exit(f"usage: python3 {sys.argv[0]} <image_path> [<population_path>]")

image_path = sys.argv[1]
image_name = os.path.splitext(os.path.basename(image_path))[0]
population_path = sys.argv[2] if len(sys.argv) == 3 else None

target_image = cv.imread(image_path)
if target_image is None:
    sys.exit("could not read the image")

evolution.evolve(target_image, image_name, population_path)
