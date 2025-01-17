import sys
import cv2 as cv
import os
import argparse
from . import evolution

parser = argparse.ArgumentParser(description="TriArt: Generate artwork using evolutionary algorithms.")

parser.add_argument("target_image", type=str, help="Path to the target image.")
parser.add_argument("--save_name", type=str, default=None, help="Prefix of the population file name.")
parser.add_argument("--population", type=int, default=None, help="ID of the population.")
parser.add_argument("--alpha", type=float, default=0.6, help="Alpha parameter.")
parser.add_argument("--display_mode", type=str, default="image", help="Specify the display mode during evolution: 'none' for no image, 'image' to show the painted image, or 'details' to show the image with extra information.")

args = parser.parse_args()

if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.target_image))[0]

target_image = cv.imread(args.target_image)
if target_image is None:
    sys.exit("could not read the image")

evolution.evolve(target_image, args.save_name, display_mode=args.display_mode, generation_index=args.population, alpha=args.alpha)
