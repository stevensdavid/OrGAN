import json
import logging
import os
import re
import shutil
import sys
from argparse import ArgumentParser
from os.path import join

from PIL import Image
from tqdm import trange
from util.pytorch_utils import pad_to_square

logger = logging.getLogger("UnsplashDownloader")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--image_dir", type=str)
parser.add_argument("--exif_dir", type=str)
parser.add_argument("--target_dir", type=str)
parser.add_argument("--n_threads", type=int, default=32)
args = parser.parse_args()

if os.path.exists(args.target_dir):
    print(f"This will delete all previous content from {args.target_dir}. Ok? [y/n]")
    if input("> ") != "y":
        sys.exit(0)
    shutil.rmtree(args.target_dir)
os.makedirs(join(args.target_dir, "images"), exist_ok=True)
f_stops = {}
# Parse apertures of type f/x.y
# Some apertures are listed as fractions, but we afford to skip these
skipped = 0
aperture_regex = re.compile(r"^-Aperture(?:\n|\r\n)f/(\d+(?:\.\d+)?)$", re.MULTILINE)


def process_image(image_idx) -> str:
    image_subdir = str(int(image_idx // 1e4))
    image_filename = f"{image_idx}.jpg"
    exif_filename = f"{image_idx}.txt"
    with open(
        join(args.exif_dir, image_subdir, exif_filename), "r", encoding="utf8"
    ) as exif_file:
        try:
            match = aperture_regex.search(exif_file.read())
        except UnicodeDecodeError:
            return None
        if match is None:
            return None
        aperture = match.group(1)
    try:
        image = Image.open(join(args.image_dir, image_subdir, image_filename))
        image = pad_to_square(image)
        image.save(join(args.target_dir, "images", image_filename))
    except OSError:
        return None
    return aperture


for image in trange(1_000_000, desc="Processing images"):
    aperture = process_image(image)
    if aperture is not None:
        f_stops[image] = aperture

with open(join(args.target_dir, "f_stops.json"), "w") as f:
    json.dump(f_stops, f)
