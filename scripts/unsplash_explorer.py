import numpy as np
import pandas as pd
import glob
from os.path import join
from io import BytesIO
from PIL import Image
import requests
import tkinter
import tkinter.filedialog
from PIL.ImageTk import PhotoImage


def get_image_data(root_dir):
    metadata_files = glob.glob(join(root_dir, "photos.tsv*"))
    metadata = pd.concat([pd.read_csv(file, sep="\t", header=0) for file in metadata_files], axis=0, ignore_index=True)

    images = metadata[metadata["exif_aperture_value"].notna()][["exif_aperture_value", "photo_image_url", "photo_url"]]
    images = metadata[metadata["exif_aperture_value"] != "undef"]
    images = images.sort_values("exif_aperture_value")
    return images

def download_image(image_url):
    response = requests.get(image_url + "?w=500")
    img = Image.open(BytesIO(response.content))
    return img.resize((500,500))


def main():
    root = tkinter.Tk()
    # root.withdraw()
    root_dir = tkinter.filedialog.askdirectory(title="Unsplash data directory")
    images = get_image_data(root_dir)
    # pick one image for each aperture
    images.drop_duplicates(subset="exif_aperture_value")
    current_image = None
    def update_image(image_url):
        nonlocal current_image
        current_image = PhotoImage(download_image(image_url))

    update_image(images.iloc[0]["photo_image_url"])

    canvas = tkinter.Canvas(root, width=500, height=500)
    canvas.pack(side=tkinter.TOP)
    tk_image = canvas.create_image(0, 0, image=current_image, anchor=tkinter.NW)

    selector = tkinter.Scale(root, from_=0, to=100, orient=tkinter.HORIZONTAL)
    f_label = tkinter.Label(root)
    def f_stop_cb(_):
        percentile = int(selector.get())
        n_images = len(images)
        image_idx = int((percentile / 100) * (n_images -1))
        selected_image = images.iloc[image_idx]
        update_image(selected_image["photo_image_url"])
        canvas.itemconfig(tk_image, image=current_image)
        f_label["text"] = f"f={selected_image['exif_aperture_value']}"

    selector.bind("<ButtonRelease-1>", f_stop_cb)
    f_label.pack(side=tkinter.BOTTOM)
    selector.pack(side=tkinter.BOTTOM)
    f_stop_cb(...)
    root.mainloop()



if __name__ is "__main__":
    main()
