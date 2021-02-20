from argparse import ArgumentParser
from logging import getLogger
from PIL import Image
import requests
import os
from io import BytesIO
from tqdm import tqdm
import glob
from os.path import join
import pandas as pd
from math import isfinite
from tempfile import TemporaryDirectory, mkdtemp
import json
import shutil



logger = getLogger("UnsplashDownloader")
parser = ArgumentParser()
parser.add_argument(
    "--unsplash_dir", type=str, required=True, help="Directory of Unsplash metadata"
)
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument(
    "--max_res", type=int, help="Resolution of longest side", default=500
)
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--y", action="store_true")
args = parser.parse_args()

logger.info(
    f"Downloading images from directory {args.unsplash_dir} to {args.target_dir}" + \
    f" and scaling to {args.max_res}px. This will take a while."
)

if not args.y and os.path.exists(args.target_dir):
    print(f"This will delete existing files in {args.target_dir}. Ok? [y/n]")
    response = input("> ")
    if response == "y":
        logger.info(f"Clearing {args.target_dir}.")
        shutil.rmtree(args.target_dir)

def pad_to_square(pil_image):
    """Adapted from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/"""
    w, h = pil_image.size
    side = max(w,h)
    if w == h: 
        return pil_image
    # pad with black
    result = Image.new(pil_image.mode, (side, side), (0,0,0))
    if w > h:
        result.paste(pil_image, (0, (w - h) //2))
    else:
        result.paste(pil_image, ((h - w) //2, 0))
    return result


def download_image(image: pd.Series):
    url = image["photo_image_url"]
    width = int(image["photo_width"])
    height = int(image["photo_height"])
    # set target side to max 500, otherwise scale to maintain aspect ratio
    target_width = int(args.max_res * min(width / height, 1))
    target_height = int(args.max_res * min(height / width, 1))
    response = requests.get(url, params={"w": target_width, "h": target_height})
    img = Image.open(BytesIO(response.content))
    padded = pad_to_square(img)
    return padded

metadata_files = glob.glob(join(args.unsplash_dir, "photos.tsv*"))
photos = pd.concat([pd.read_csv(file, sep="\t", header=0) for file in metadata_files], axis=0, ignore_index=True)

labels = {}
temp_dir = mkdtemp(dir=args.temp_dir)
try:
    os.makedirs(join(temp_dir, "images"))

    for _, photo in tqdm(photos.iterrows(), desc=f"Downloading images to temp dir {temp_dir}", total=len(photos)):
        try:
            f_stop = float(photo["exif_aperture_value"])
        except ValueError:
            continue
        if not isfinite(f_stop):
            continue
        photo_id = photo["photo_id"]
        try:
            image = download_image(photo)
        except Exception as e:
            logger.error(f"Could not download photo {photo_id}. Error: {e}")
            continue
        labels[photo_id] = f_stop
        try:
            image.save(join(temp_dir, "images", f"{photo_id}.jpg"))
        except OSError as e:
            logger.error(f"Could not save downloaded photo {photo_id}. Error: {e}")
    with open(join(temp_dir, "f_stops.json"), "w") as f:
        json.dump(labels, f)
    shutil.move(temp_dir, args.target_dir)
finally:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
