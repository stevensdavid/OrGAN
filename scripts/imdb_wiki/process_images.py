"""Adapted from https://github.com/imdeepmind/processed-imdb-wiki-dataset/blob/master/age.py"""
# Importing dependencies
import json
import os
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import UUID

import numpy as np
import pandas as pd
from PIL import Image
from scripts.util import pad_to_square
from tqdm import tqdm

# Setup deterministic identifiers
rd = random.Random()
rd.seed(0)
reproducible_uuid = lambda: UUID(int=rd.getrandbits(128))

parser = ArgumentParser()
parser.add_argument("--root_dir", type=str, help="IMDB-WIKI directory", required=True)
parser.add_argument("--target_dim", type=int, default=128)
parser.add_argument("--n_workers", type=int, default=12)
args = parser.parse_args()

# Loading dataset
meta = pd.read_csv(os.path.join(args.root_dir, "meta.csv"))

# Dropping gender column
meta = meta.drop(["gender"], axis=1)

# Filtaring dataset
meta = meta[meta["age"] >= 0]
meta = meta[meta["age"] <= 101]

os.makedirs(os.path.join(args.root_dir, "processed_images"), exist_ok=True)
ages = {}


def process_image(image_metadata):
    image_path = os.path.join(args.root_dir, image_metadata["path"])
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = pad_to_square(image)
    image = image.resize((args.target_dim, args.target_dim))

    age = image_metadata["age"]
    filename = f"{str(age).zfill(3)}_{reproducible_uuid()}.jpg"
    destination_path = os.path.join(args.root_dir, "processed_images", filename)
    ages[filename] = age
    image.save(destination_path)


with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
    jobs = [
        executor.submit(process_image, image_metadata)
        for _, image_metadata in tqdm(
            meta.iterrows(), desc="Queuing images", total=len(meta)
        )
    ]
    for _ in tqdm(as_completed(jobs), desc="Processing images", total=len(meta)):
        continue

with open(os.path.join(args.root_dir, "ages.json"), "w") as f:
    json.dump(ages, f)
