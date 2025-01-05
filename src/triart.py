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
parser.add_argument("-debug", default=False, help="Shows targeted pixels and recently added triangles.", action="store_true")
parser.add_argument("-record", default=False, help="Record frames.", action="store_true")
parser.add_argument("-save_all", default=False, help="Saves all generations.", action="store_true")

args = parser.parse_args()

if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.target_image))[0]

target_image = cv.imread(args.target_image)
if target_image is None:
    sys.exit("could not read the image")

evolution.evolve(target_image, args.save_name, generation_index=args.population, record=args.record, save_all=args.save_all, debug=args.debug, alpha=args.alpha)
