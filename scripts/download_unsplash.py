from argparse import ArgumentParser
import logging
from PIL import Image
from aiohttp import ClientSession
import os
from io import BytesIO
import tqdm.asyncio
import tqdm
import glob
from os.path import join
import pandas as pd
from math import isfinite
from tempfile import mkdtemp
import json
import shutil
import asyncio
import sys
from util import pad_to_square

logger = logging.getLogger("UnsplashDownloader")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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
parser.add_argument("--max_conns", type=int, default=50)
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

async def fetch_img(url, params, session):
    async with session.get(url, params=params) as response:
        return await response.read()

async def download_image(image: pd.Series, session: ClientSession):
    url = image["photo_image_url"]
    width = int(image["photo_width"])
    height = int(image["photo_height"])
    # set target side to max 500, otherwise scale to maintain aspect ratio
    target_width = int(args.max_res * min(width / height, 1))
    target_height = int(args.max_res * min(height / width, 1))
    response = await fetch_img(url, params={"w": target_width, "h": target_height}, session=session)
    img = Image.open(BytesIO(response))
    if img.mode in ("RGBA", "P"): 
        img = img.convert("RGB")
    padded = pad_to_square(img)
    return padded

logger.info("Loading metadata")
metadata_files = glob.glob(join(args.unsplash_dir, "photos.tsv*"))
photos = pd.concat([pd.read_csv(file, sep="\t", header=0) for file in metadata_files], axis=0, ignore_index=True)
photos = photos[["exif_aperture_value", "photo_id", "photo_image_url", "photo_width", "photo_height"]]

labels = {}
temp_dir = mkdtemp(dir=args.temp_dir)

async def process_photo(photo: pd.Series, session: ClientSession):
    try:
        f_stop = float(photo["exif_aperture_value"])
    except ValueError:
        return
    if not isfinite(f_stop):
        return
    photo_id = photo["photo_id"]
    try:
        image = await download_image(photo, session)
    except Exception as e:
        logger.error(f"Could not download photo {photo_id}. Error: {e}")
        return
    labels[photo_id] = f_stop
    try:
        image.save(join(temp_dir, "images", f"{photo_id}.jpg"))
    except OSError as e:
        logger.error(f"Could not save downloaded photo {photo_id}. Error: {e}")

async def download_worker(photo_queue: asyncio.Queue, progbar: tqdm.tqdm):
    async with ClientSession() as session:
        while True:
            photo = await photo_queue.get()
            try:
                if photo is None:
                    # all work is done
                    return
                await process_photo(photo, session)
                progbar.update()
            finally:
                photo_queue.task_done()

async def download_all(temp_dir):
    logger.info("Downloading all photos.")
    photo_queue = asyncio.Queue()#maxsize=args.max_conns*4)
    worker_tasks = []
    with tqdm.tqdm(total=len(photos)) as progbar:
        for _ in range(args.max_conns):
            worker_tasks.append(asyncio.create_task(download_worker(photo_queue, progbar)))
        for _, photo in tqdm.tqdm(photos.iterrows(), desc="Queuing downloads", total=len(photos)):
            await photo_queue.put(photo)
        for _ in range(args.max_conns):
            await photo_queue.put(None)
        await photo_queue.join()
        # progress_bar = tqdm.tqdm(total=len(photos))
        # for task in tqdm.asyncio.tqdm.as_completed(photo_queue, total=len(photos)):
        #     await task
        await asyncio.gather(*worker_tasks)

try:
    os.makedirs(join(temp_dir, "images"))
    asyncio.run(download_all(temp_dir))
    with open(join(temp_dir, "f_stops.json"), "w") as f:
        json.dump(labels, f)
    shutil.move(temp_dir, args.target_dir)
finally:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
