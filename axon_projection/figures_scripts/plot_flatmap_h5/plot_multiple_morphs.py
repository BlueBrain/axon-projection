"""Module to plot flatmaps of multiple bio or synth morphologies in parallel."""
import os
import sys
from multiprocessing import Manager
from multiprocessing import Pool

import cv2

# import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from axon_synthesis.utils import get_morphology_paths

# from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from PIL import Image

from axon_projection.query_atlas import without_hemisphere

# from scipy.ndimage import gaussian_filter


def plot_all_morphs(morphs_list, axon_color, batch_size, n_morphs_to_plot):
    """Plots the flatmaps of the given morphs."""
    args_list = []
    for n_morphs, morph_path in enumerate(morphs_list):
        args = (morph_path, axons_color)
        args_list.append(args)
        if n_morphs == (n_morphs_to_plot - 1):
            break

    for i in range(0, len(args_list), batch_size):
        print(f"{i}/{len(args_list)}")
        batch = args_list[i : i + batch_size]
        with Manager() as _:
            with Pool(batch_size) as pool:
                pool.starmap(plot_flat_morph, batch)
    return args_list


def plot_flat_morph(morph_path, axon_color):
    """Plots the flatmap of the given morph."""
    morph_name = os.path.basename(morph_path).split(".")[0]
    os.system(
        "python flatplot.py "
        f"--dots --h5morpho axon --color '{axons_color}' -p 512 "
        f"--dual flatmap_both.nrrd {morph_path} {morph_name}"
    )
    os.system(f"convert temp.png {morph_name}.png -composite temp.png")

    # os.system(f"convert background_black_.png
    # {morph_name}.png -composite {morph_name}_black.png")
    # os.system(f"rm {morph_name}.png")


def overlay_morph(morph_path):
    """Overlays the flatmap of the given morph."""
    morph_name = os.path.basename(morph_path).split(".")[0]
    print(morph_name)
    os.system(f"convert temp.png {morph_name}.png -composite temp.png")
    os.system(f"rm {morph_name}.png")


def delete_temp_files(morphs_list):
    """Deletes the temporary files."""
    for morph_path in morphs_list:
        morph_name = os.path.basename(morph_path).split(".")[0]
        os.system(f"rm {morph_name}.png")
        os.system(f"rm {morph_name}_black.png")
    os.system("rm temp.png")


def plot_gaussian_smoothing(morph_paths, is_bio):
    """Plots the gaussian smoothing of the given morphs."""
    for m, morph_path in enumerate(morph_paths):
        # load the morph png image as a numpy array
        morph_name = os.path.basename(morph_path).split(".")[0]
        img_path = morph_name + "_black.png"
        morph_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # replace the pixels > 0 with 1 (so it's normalized)
        morph_array[morph_array > 0] = 1.0
        if m == 0:
            all_morph_array = morph_array
        else:
            # sum the morph_array with the all_morph_array
            all_morph_array = all_morph_array + morph_array
    # get rid of all spines and ticks
    plt.axis("off")
    # load the background image
    background_img = np.array(Image.open("background.png"))

    # normalize the data
    smoothed_data = all_morph_array / np.max(all_morph_array)
    # smooth the data
    # sigma = 1.0  # Standard deviation for Gaussian kernel
    # smoothed_data = gaussian_filter(all_morph_array, sigma=sigma)
    # smoothed_data = smoothed_data / np.max(smoothed_data)
    # Choose a sequential colormap
    if is_bio:
        # we want the inverted Blues cmap
        base_cmap = plt.get_cmap("Purples_r")
        # base_cmap = mcolors.LinearSegmentedColormap.
        # from_list('BlackBlue', ['black', 'blue'])
    else:
        base_cmap = plt.get_cmap("Reds_r")
        # base_cmap = mcolors.LinearSegmentedColormap.
        # from_list('BlackBlue', ['black', 'red'])

    colors = base_cmap(np.arange(base_cmap.N))
    colors[0] = [1, 1, 1, 0]  # Set the color for the 0 value to transparent (white with 0 alpha)
    cmap = ListedColormap(colors)
    # Define the boundaries for the discrete color map
    # bounds = np.arange(-0.5, np.max(smoothed_data) + 0.5, 1)
    # norm = BoundaryNorm(bounds, cmap.N)
    # plot the background image
    plt.imshow(background_img, cmap=None, interpolation=None)
    # plot the smoothed data on the background image
    plt.imshow(smoothed_data, cmap=cmap, interpolation=None)
    if is_bio:
        label_ = "Biological"
    else:
        label_ = "Synthesized"
    plt.colorbar(label=label_ + " axons density")  # ticks=np.arange(0, np.max(smoothed_data)+1),
    plt.savefig("axons_density.pdf")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage <bio|synth> <n_max_morphs_to_plot>")
        exit(1)

    bio_morph_dir = (
        "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
        "axon_projection/out_a_p_final/axon_lengths_12.csv"
    )
    synth_morph_dir = (
        "/gpfs/bbp.cscs.ch/project/proj135/home/petkantc/axon/axonal-projection/"
        "axon_projection/validation/circuit-build/lite_MOp5_final/a_s_out/Morphologies/non_centered"
    )
    # morph_dir = sys.argv[1]
    is_bio = sys.argv[1] == "bio"
    is_synth = sys.argv[1] == "synth"
    is_both = sys.argv[1] == "both"
    n_morphs_to_plot = int(sys.argv[2])
    source = "MOp5"

    batch_size = 8

    bio_color = "#5ec3ed"
    synth_color = "#fa3a3a"

    # os.system("cp background_black_.png temp.png")
    os.system("cp background_old.png temp.png")

    args_list_tot = []

    if is_bio or is_both:
        axons_color = bio_color
        # filter bio_df from path to have morphs list from MOp5
        # in that case, morph_dir is the path to axonal_lengths
        # (because already filtered for source)
        axons_df = pd.read_csv(bio_morph_dir, index_col=0)
        axons_df = axons_df[axons_df["source"].apply(without_hemisphere) == source]
        list_morphs = axons_df["morph_path"].unique().tolist()
        print(f"INFO: {len(list_morphs)} bio morphs for {source}.")
        args_list_tot += plot_all_morphs(list_morphs, axons_color, batch_size, n_morphs_to_plot)

    if is_synth or is_both:
        axons_color = synth_color
        # get list of morphologies at morph_dir location
        list_morphs = get_morphology_paths(synth_morph_dir)["morph_path"].values.tolist()
        # shuffle randomly the list
        rng = np.random.default_rng()
        rng.shuffle(list_morphs)
        args_list_tot += plot_all_morphs(list_morphs, axons_color, batch_size, n_morphs_to_plot)

    # the actual list_morphs is the first value of the args_list tuples
    list_morphs = [x[0] for x in args_list_tot]
    # plot_gaussian_smoothing(list_morphs, is_bio)
    # os.system(f"mv axons_density.pdf
    # axons_density_{sys.argv[2]}_n{n_morphs_to_plot}_hemisphere.pdf")

    for morph_path, _ in args_list_tot:
        overlay_morph(morph_path)

    os.system(f"mv temp.png axons_{sys.argv[1]}_n{n_morphs_to_plot}_final.png")

    delete_temp_files(list_morphs)
