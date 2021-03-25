import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os.path import join
import json
import numpy as np

parser = ArgumentParser()
parser.add_argument("--unsplash_dir", type=str, required=True)
parser.add_argument("--mirflickr_dir", type=str, required=True)
args = parser.parse_args()

print("Loading unsplash")
with open(join(args.unsplash_dir, "f_stops.json"), "r") as f:
    unsplash_labels: dict = json.load(f)

print("Loading MIRFLICKR")
with open(join(args.mirflickr_dir, "f_stops.json"), "r") as f:
    mirflickr_labels: dict = json.load(f)

unsplash_apertures = [x for x in unsplash_labels.values() if 0.5 <= x <= 32]
mirflickr_apertures = [x for x in mirflickr_labels.values() if 0.5 <= x <= 32]

from_ = min(unsplash_apertures + mirflickr_apertures)
to = max(unsplash_apertures + mirflickr_apertures)
bins = np.logspace(np.log10(from_), np.log10(to), num=20,)
print("Processing unsplash")
plt.title("f-stop distribution")
plt.subplot(311).set_title("Unsplash")
plt.hist(unsplash_apertures, log=False, rwidth=0.95, bins=bins)
plt.xlabel("f-stop")
plt.xscale("log")
print("Processing MIRFLICKR")
plt.subplot(312).set_title("MIRFLICKR")
plt.hist(mirflickr_apertures, log=False, rwidth=0.95, bins=bins)
plt.xlabel("f-stop")
plt.xscale("log")
print("Processing combined data")
plt.subplot(313).set_title("Combined")
b = plt.hist(
    mirflickr_apertures + unsplash_apertures, log=False, rwidth=0.95, bins=bins
)
plt.xlabel("f-stop")
plt.xscale("log")

plt.tight_layout()
plt.show()
