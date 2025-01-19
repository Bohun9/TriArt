import sys
import cv2 as cv
import os
import argparse
from . import evolution

parser = argparse.ArgumentParser(
    description="TriArt: Generate artwork using evolutionary algorithms."
)

parser.add_argument("target_image", type=str, help="Path to the target image.")
parser.add_argument("--save_name", type=str, default=None, help="Evolution file name.")
parser.add_argument(
    "--population", type=int, default=None, help="ID of the population."
)
parser.add_argument("--alpha", type=float, default=0.6, help="Alpha parameter.")
parser.add_argument(
    "--display_mode",
    type=str,
    default="image",
    help="Specify the display mode during evolution: 'none' for no image, 'image' to show the painted image, or 'details' to show the image with extra information.",
)
parser.add_argument("--seed", type=int, default=0, help="Seed for initializing rng.")
parser.add_argument(
    "--max_num_shapes", type=int, default=500, help="Maximum number of shapes."
)
parser.add_argument(
    "--gen_per_shape", type=int, default=15, help="Generations per shape."
)
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
parser.add_argument("--population_size", type=int, default=20, help="Population size.")
parser.add_argument("--offspring_size", type=int, default=60, help="Offspring size.")

args = parser.parse_args()

if args.save_name is None:
    args.save_name = os.path.splitext(os.path.basename(args.target_image))[0]

target_image = cv.imread(args.target_image)
if target_image is None:
    sys.exit("could not read the image")

evolution.evolve(
    target_image,
    args.save_name,
    display_mode=args.display_mode,
    generation_index=args.population,
    population_size=args.population_size,
    offspring_size=args.offspring_size,
    alpha=args.alpha,
    seed=args.seed,
    max_number_of_shapes=args.max_num_shapes,
    generations_per_shape=args.gen_per_shape,
    num_workers=args.num_workers,
)
