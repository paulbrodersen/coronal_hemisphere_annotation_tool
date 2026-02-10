"""
Create an output PDF that displays each segmented slice,
the corresponding Allen Brain Atlas annnotation,
and the isolated samples including their barcode.

Example
-------
python code/make_figures.py test/segmentation/ test/annotation/ test/sample_data.csv test/figures/

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from matplotlib.backends import backend_pdf
from matplotlib.colors import to_hex
from skimage.measure import label, find_contours
from shapely.geometry import Polygon, LineString
from shapely.ops import polylabel
from itertools import combinations
from scipy.optimize import minimize, NonlinearConstraint


def get_well_spaced_angles(angles, minimum_delta_angle=np.pi/9):

    def get_angle_difference(a, b):
        return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

    def cost_function(new, old):
        return np.sum(get_angle_difference(new, old))

    def constraint_function(angles):
        a, b = np.array(list(combinations(angles, 2))).T
        return get_angle_difference(a, b)

    nonlinear_constraint = NonlinearConstraint(constraint_function, lb=minimum_delta_angle, ub=np.pi + 1e-3)
    res = minimize(lambda x: cost_function(x, angles), angles, constraints=[nonlinear_constraint])
    return res.x


def get_well_spaced_vectors(vectors, minimum_delta_angle=2*np.pi/36):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = get_well_spaced_angles(angles, minimum_delta_angle)
    return np.c_[np.cos(angles), np.sin(angles)]


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser = ArgumentParser()
    parser.add_argument("segmentation",     help="/path/to/segmentation/directory/", type=str)
    parser.add_argument("annotation",       help="/path/to/annotation/directory/",   type=str)
    parser.add_argument("sample_data",      help="/path/to/sample_data.csv",         type=str)
    parser.add_argument("output_directory", help="/path/to/ouput/directory/",        type=str)
    parser.add_argument("--highlight",      help="List of barcodes",                 type=str, nargs="+", default=[])
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    # load segmentation
    segmentation_directory = Path(args.segmentation)
    data = np.load(segmentation_directory / "segmentation_results.npz")
    slice_images  = data["slice_images"]
    slice_masks   = data["slice_masks"]
    sample_masks  = data["sample_masks"]
    slice_numbers = data["slice_numbers"]

    df = pd.read_csv(args.sample_data)

    # load annotation
    data = np.load(Path(args.annotation) / "annotation_results.npz", allow_pickle=True)
    annotations     = data["annotations"]
    color_map_array = data["color_map"]
    name_map_array  = data["name_map"]

    color_map = dict(zip(color_map_array[:, 0], color_map_array[:, 1:]))
    name_map = dict(name_map_array)

    # determine image center
    total_slices, total_rows, total_cols = slice_images.shape
    xc = total_cols / 2
    yc = total_rows / 2

    # --------------------------------------------------------------------------------

    print("Plotting slices & annotations...")
    date, = df["date"].unique()
    date = date.replace("-", "_")

    with backend_pdf.PdfPages(output_directory / f"individual_slices_{date}.pdf") as pdf:

        for ii in range(total_slices):
            print(f"{ii+1} / {total_slices}")
            slice_number = slice_numbers[ii]
            slice_image  = slice_images[ii]
            slice_mask   = slice_masks[ii]
            annotation   = annotations[ii]
            sample_mask  = sample_masks[ii]

            fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 12))
            for ax in axes:
                ax.axis("off")
                ax.set_aspect("equal")
            axes[0].imshow(slice_image, cmap="gray")
            axes[2].imshow(slice_image, cmap="gray")

            # plot slice contour
            contours = find_contours(slice_mask, 0.5)
            for contour in contours:
                x = contour[:, 1]
                y = contour[:, 0]
                axes[1].plot(x, y, "#677e8c", linewidth=1.0)
                axes[2].plot(x, y, "#677e8c", linewidth=1.0)

            # get the contour around the outside of the slice
            contour = sorted(contours, key = lambda x : len(x))[-1]
            slice_contour = Polygon(np.c_[contour[:, 1], contour[:, 0]])

            # plot region contours
            for annotation_id in np.unique(annotation):
                if annotation_id > 0: # 0 is background
                    region_mask = annotation == annotation_id
                    for contour in find_contours(region_mask, 0.5):
                        x = contour[:, 1]
                        y = contour[:, 0]
                        axes[1].plot(x, y, color=to_hex(color_map[annotation_id]/255), linewidth=0.25)
                        axes[2].plot(x, y, color=to_hex(color_map[annotation_id]/255), linewidth=0.25)

            subset = df[df["slice_number"] == slice_number]

            if len(subset) > 0:

                # compute slice radius, i.e. maximum distance from center
                slice_radius = 0
                center = np.array([yc, xc])
                for contour in contours:
                    delta = contour - center[np.newaxis, :]
                    distance = np.linalg.norm(delta, axis=1)
                    slice_radius = max(slice_radius, np.max(distance))

                # plot sample locations
                for _, row in subset.iterrows():
                    x = row["segmentation_col"]
                    y = row["segmentation_row"]
                    if row["barcode"] in args.highlight:
                        markersize = 2
                    else:
                        markersize = 1
                    axes[1].plot(x, y, linestyle="", marker='o', markersize=markersize, color=row["subclass_color"])
                    axes[2].plot(x, y, linestyle="", marker='o', markersize=markersize, color=row["subclass_color"])

                # annotate samples; prevent labels from overlapping
                labels = subset["barcode"]
                indices = subset.index.values
                center = np.array([xc, yc])
                coordinates = subset[["segmentation_col", "segmentation_row"]].values
                vectors = coordinates - center[np.newaxis, :]
                vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
                if len(vectors) > 1:
                    vectors = get_well_spaced_vectors(vectors)
                for xy, vector, label, idx in zip(coordinates, vectors, labels, indices):
                    line = LineString([center, center + 1.1 * slice_radius * vector])
                    intersection = np.array(slice_contour.intersection(line).coords[:][-1])
                    if label in args.highlight:
                         fontsize = 7
                         fontweight = "normal"
                    else:
                         fontsize = 5
                         fontweight="normal"

                    axes[1].annotate(
                        f"{label} ({idx})",
                        xy,
                        center + 1.1 * (intersection - center),
                        ha="right", va="bottom",
                        fontsize=fontsize,
                        fontweight=fontweight,
                        color="#677e8c",
                        arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                        wrap=True,
                    )
                    axes[2].annotate(
                        f"{label} ({idx})",
                        xy,
                        center + 1.1 * (intersection - center),
                        ha="right", va="bottom",
                        fontsize=fontsize,
                        fontweight=fontweight,
                        color="#677e8c",
                        arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                        wrap=True,
                    )

                # plot regions that have a sample in them
                region_coordinates = []
                region_labels = []
                region_colors = []
                for annotation_id in np.unique(subset["annotation_id"]):
                    region_mask = annotation == annotation_id
                    region_mask[:, :int(xc) + 1] = False
                    contours = find_contours(region_mask, 0.5)
                    # The contour finding algorithm sometimes identifies
                    # spurious contours around the corners of the region.
                    # We hence only plot the largest contour.
                    contour = sorted(contours, key = lambda x : len(x))[-1]
                    patch = plt.Polygon(np.c_[contour[:, 1], contour[:, 0]], color=to_hex(color_map[annotation_id]/255))
                    axes[1].add_patch(patch)
                    patch = plt.Polygon(np.c_[contour[:, 1], contour[:, 0]], color=to_hex(color_map[annotation_id]/255))
                    axes[2].add_patch(patch)
                    polygon = Polygon(np.c_[contour[:, 1], contour[:, 0]])
                    poi = polylabel(polygon, tolerance=0.1)
                    x, y = poi.x, poi.y
                    region_coordinates.append((x, y))
                    region_labels.append(name_map[int(annotation_id)])

                # annotate regions
                region_coordinates = np.array(region_coordinates)
                vectors = region_coordinates - center[np.newaxis, :]
                vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
                if len(vectors) > 1:
                    vectors = get_well_spaced_vectors(vectors)
                for xy, vector, label in zip(region_coordinates, vectors, region_labels):
                    line = LineString([center, center + 1.1 * slice_radius * vector])
                    intersection = np.array(slice_contour.intersection(line).coords[:][-1])
                    axes[1].annotate(
                        label,
                        xy,
                        center + 1.1 * (intersection - center),
                        ha="left", va="bottom",
                        fontsize=5,
                        color="#677e8c",
                        arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                        wrap=True,
                    )
                    axes[2].annotate(
                        label,
                        xy,
                        center + 1.1 * (intersection - center),
                        ha="left", va="bottom",
                        fontsize=5,
                        color="#677e8c",
                        arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                        wrap=True,
                    )

            fig.savefig(output_directory / f"slice_{slice_number:03d}.svg")
            pdf.savefig(fig)
            plt.close(fig)

    # --------------------------------------------------------------------------------

    print("Plotting slices aligned in 3D...")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis("off")

    for ii in range(total_slices):
        print(f"{ii+1} / {total_slices}")
        slice_number = slice_numbers[ii]
        slice_image  = slice_images[ii]
        slice_mask   = slice_masks[ii]
        annotation   = annotations[ii]
        sample_mask  = sample_masks[ii]

        # plot slice contour
        contours = find_contours(slice_mask, 0.5)
        for contour in contours:
            x = contour[:, 1]
            y = contour[:, 0]
            z = -slice_number * np.ones_like(x)
            ax.plot(z, x, -y, "#677e8c", linewidth=1.0)

        # compute slice radius
        slice_radius = 0
        center = np.array([yc, xc])
        for contour in contours:
            delta = contour - center[np.newaxis, :]
            distance = np.linalg.norm(delta, axis=1)
            slice_radius = max(slice_radius, np.max(distance))

        # plot and label samples
        for _, row in df[df["slice_number"] == slice_number].iterrows():
            x = row["segmentation_col"]
            y = row["segmentation_row"]
            xyz = np.array([-slice_number, x, -y])
            ax.plot(*xyz, linestyle="", marker='o', color=row["subclass_color"])

    # label clones
    if len(args.highlight) == 0: # label all clones
        for barcode in np.unique(df["barcode"]):
            clone = df[df["barcode"] == barcode]
            if len(clone) > 1:
                Z = -clone["slice_number"]
                X = clone["segmentation_col"]
                Y = clone["segmentation_row"]
                zt = Z.mean()
                xt = xc + 2 * (X.mean() - xc)
                yt = yc + 2 * (Y.mean() - yc)
                ax.text(zt, xt, -yt, barcode + " ", ha="right")
                for zz, xx, yy in zip(Z, X, Y):
                    ax.plot([zt, zz], [xt, xx], [-yt, -yy], linewidth=0.5, color="#677e8c")
    else: # only label clones to be highlighted
        for barcode in args.highlight:
            clone = df[df["barcode"] == barcode]
            if len(clone) > 1:
                Z = -clone["slice_number"]
                X = clone["segmentation_col"]
                Y = clone["segmentation_row"]
                zt = Z.mean()
                xt = xc + 2 * (X.mean() - xc)
                yt = yc + 2 * (Y.mean() - yc)
                ax.text(zt, xt, -yt, barcode + " ", ha="right", fontsize="x-large")
                for zz, xx, yy in zip(Z, X, Y):
                    ax.plot([zt, zz], [xt, xx], [-yt, -yy], linewidth=0.5, color="#677e8c")


    print("Select the desired view. The figure will be saved on closing.")
    plt.show()
    fig.savefig(output_directory / "reconstruction_in_3d.pdf")
    fig.savefig(output_directory / "reconstruction_in_3d.svg")
