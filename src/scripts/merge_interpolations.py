import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.image import imread, imsave

rc("font", **{"family": "sans-serif", "sans-serif": ["TeX Gyre Heros"]})
rc("text", usetex=True)

parser = ArgumentParser()
parser.add_argument("--project_root")
parser.add_argument("--res", type=int, default=32)
args = parser.parse_args()
res = args.res

sweeps = os.listdir(args.project_root)
paths = [
    "ralsgan",
    "stargan",
    "ccgan_hard",
    "ccgan_soft",
    "fpgan",
    "lsgan",
    "wgan_gp",
    "no_noise",
    "plain_discriminator",
    "plain_generator",
]
paths = [f"{args.project_root}/{path}/interpolations.png" for path in paths]
paths = [path for path in paths if os.path.exists(path)]
# paths = sorted(
#     filter(
#         os.path.exists,
#         map(lambda s: os.path.join(args.project_root, s, "interpolations.png"), sweeps),
#     )
# )
interpolations = np.stack([imread(path) for path in paths])

display_names = {
    "ralsgan": "OrGAN",
    "stargan": "StarGAN",
    "ccgan_hard": "+ HVDL",
    "ccgan_soft": "+ SVDL",
    "fpgan": r"+ $\mathcal{L}_{id}$",
    "lsgan": r"$\sim$ LSGAN",
    "wgan_gp": r"$\sim$ WGAN-GP",
    "no_noise": "- Noise",
    "plain_discriminator": "- $D$ embedding",
    "plain_generator": "- $G$ embedding",
}


def add_image_to_plot(image, name, n_rows, idx, n_cols=1):
    ax = plt.subplot(n_rows, n_cols, idx)
    ax.imshow(image)
    if n_cols == 1 or idx % n_cols == 1:
        ax.set_title(name, x=-0.01, y=0, loc="right")
    elif idx % n_cols == 0:
        ax.set_title(name, x=1.01, y=0, loc="left")
    ax.axis("off")


n_models = interpolations.shape[0]
n_samples = interpolations.shape[1] // (2*res)
for image_idx in range(n_samples):
    plt.figure(image_idx)
    input_image = interpolations[0, image_idx * 2*res : (image_idx * 2*res) + res, :res]
    add_image_to_plot(input_image, "Input", n_models + 2, 1)
    ground_truth = interpolations[0, image_idx * 2*res + res : (image_idx + 1) * 2*res, res:]
    outputs = interpolations[:, image_idx * 2*res : (image_idx * 2*res) + res, res:]
    n_models, h, w, c = outputs.shape
    for model in range(n_models):
        model_name = os.path.split(os.path.dirname(paths[model]))[1]
        add_image_to_plot(
            outputs[model], display_names[model_name], n_models + 2, model + 2
        )
    add_image_to_plot(ground_truth, "Ground truth", n_models + 2, n_models + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.project_root, f"interp_{image_idx}.png"), dpi=600)
plt.figure(image_idx + 1, figsize=[6.4 * n_samples / 2, 4.8])
for model_idx in range(n_models):
    model_name = os.path.split(os.path.dirname(paths[model_idx]))[1]
    for image_idx in range(n_samples):
        output = interpolations[model_idx, image_idx * 2*res : (image_idx * 2*res) + res, res:]
        add_image_to_plot(
            output,
            display_names[model_name],
            n_models + 1,
            model_idx * n_samples + image_idx + 1,
            n_cols=n_samples,
        )
for image_idx in range(n_samples):
    ground_truth = interpolations[0, image_idx * 2*res + res : (image_idx + 1) * 2*res, res:]
    add_image_to_plot(
        ground_truth,
        "Ground truth",
        n_models + 1,
        n_models * n_samples + image_idx + 1,
        n_cols=n_samples,
    )
# plt.subplots_adjust(wspace=0.01, hspace=0)
plt.tight_layout()
plt.savefig(os.path.join(args.project_root, "interp_all.png"), dpi=300)
