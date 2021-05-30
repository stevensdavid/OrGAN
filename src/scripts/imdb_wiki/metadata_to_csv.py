"""
Adapted from 
https://github.com/imdeepmind/processed-imdb-wiki-dataset/blob/master/mat.py
"""

import datetime as date
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.io import loadmat
from tqdm import tqdm, trange

parser = ArgumentParser()
parser.add_argument("--root_dir", help="IMDB-WIKI directory", required=True)
args = parser.parse_args()

cols = ["age", "gender", "path", "face_score1", "face_score2"]

imdb_mat = os.path.join(args.root_dir, "imdb_crop", "imdb.mat")
wiki_mat = os.path.join(args.root_dir, "wiki_crop", "wiki.mat")

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

imdb = imdb_data["imdb"]
wiki = wiki_data["wiki"]

imdb_photo_taken = imdb[0][0][1][0]
imdb_full_path = imdb[0][0][2][0]
imdb_gender = imdb[0][0][3][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_photo_taken = wiki[0][0][1][0]
wiki_full_path = wiki[0][0][2][0]
wiki_gender = wiki[0][0][3][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append("imdb_crop/" + path[0])

for path in wiki_full_path:
    wiki_path.append("wiki_crop/" + path[0])

imdb_genders = []
wiki_genders = []

for n in range(len(imdb_gender)):
    if imdb_gender[n] == 1:
        imdb_genders.append("male")
    else:
        imdb_genders.append("female")

for n in range(len(wiki_gender)):
    if wiki_gender[n] == 1:
        wiki_genders.append("male")
    else:
        wiki_genders.append("female")

imdb_dob = []
wiki_dob = []

for file in tqdm(imdb_path, desc="Parsing IMDB dates of birth"):
    temp = file.split("_")[3]
    temp = temp.split("-")
    if len(temp[1]) == 1:
        temp[1] = "0" + temp[1]
    if len(temp[2]) == 1:
        temp[2] = "0" + temp[2]

    if temp[1] == "00":
        temp[1] = "01"
    if temp[2] == "00":
        temp[2] = "01"

    imdb_dob.append("-".join(temp))

for file in tqdm(wiki_path, desc="Parsing Wikipedia dates of birth"):
    wiki_dob.append(file.split("_")[2])


imdb_age = []
wiki_age = []

invalid = 0
for i in trange(len(imdb_dob), desc="Calculating age for IMDB"):
    try:
        d1 = date.datetime.strptime(imdb_dob[i][0:10], "%Y-%m-%d")
        d2 = date.datetime.strptime(str(imdb_photo_taken[i]), "%Y")
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except ValueError:
        invalid += 1
        diff = -1
    imdb_age.append(diff)
print(f"Found {invalid} invalid IMDB ages. These are represented as -1.")

invalid = 0
for i in trange(len(wiki_dob), desc="Calculating age for Wikipedia"):
    try:
        d1 = date.datetime.strptime(wiki_dob[i][0:10], "%Y-%m-%d")
        d2 = date.datetime.strptime(str(wiki_photo_taken[i]), "%Y")
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except ValueError:
        invalid += 1
        diff = -1
    wiki_age.append(diff)
print(f"Found {invalid} invalid Wikipedia ages. These are represented as -1.")


final_imdb = np.vstack(
    (imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)
).T
final_wiki = np.vstack(
    (wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)
).T

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

meta = pd.concat((final_imdb_df, final_wiki_df))

meta = meta[meta["face_score1"] != "-inf"]
meta = meta[meta["face_score2"] == "nan"]

meta = meta.drop(["face_score1", "face_score2"], axis=1)

meta = meta.sample(frac=1)

meta.to_csv(os.path.join(args.root_dir, "meta.csv"), index=False)
