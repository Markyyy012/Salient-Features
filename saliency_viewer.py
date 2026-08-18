import argparse
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from saliency_utils import compute_static, fine_threshold, extract_rois, draw_rois

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to image file, or directory to scan for images")
ap.add_argument("-d", "--debug", action="store_true",
                help="show all maps side-by-side (default: original + ROIs)")
args = vars(ap.parse_args())

if os.path.isdir(args["images"]):
    candidates = sorted(glob.glob(os.path.join(args["images"], "*")))
    paths = [p for p in candidates
             if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
else:
    paths = [args["images"]]

if not paths:
    raise SystemExit(f"No images found in {args['images']}")

results = []
for path in paths:
    image = cv2.imread(path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (spectralMap, fineMap) = compute_static(image)
    threshMap = fine_threshold(fineMap)
    minArea = 0.005 * image.shape[0] * image.shape[1]
    rois = extract_rois(threshMap, min_area=minArea)
    roiImg = draw_rois(image, rois)
    results.append((os.path.basename(path), image, spectralMap, fineMap,
                    threshMap, roiImg, rois))

if not results:
    raise SystemExit("No readable images")

idx = 0


def render(i):
    (name, image, spectralMap, fineMap, threshMap, roiImg, rois) = results[i]
    if args["debug"]:
        panels = [(image, "Image"), (spectralMap, "Spectral"),
                  (fineMap, "Fine Grained"), (threshMap, "Threshold"),
                  (roiImg, f"{len(rois)} ROI(s)")]
        n = len(panels)
        rows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        axes = np.atleast_1d(axes.ravel())
        for ax in axes:
            ax.axis("off")
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title)
        for ax in axes[n:]:
            ax.set_visible(False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(roiImg)
        axes[1].set_title(f"{len(rois)} ROI(s) (saliency contours)")
        axes[1].axis("off")

    fig.suptitle(f"{name}  [{i + 1}/{len(results)}]  "
                 f"(right/left = next/prev, q = quit)")
    return fig


fig = render(idx)


def on_key(event):
    global idx
    if event.key == "q":
        plt.close(fig)
        return
    if event.key in ("right", "n", " "):
        idx = (idx + 1) % len(results)
    elif event.key in ("left", "p"):
        idx = (idx - 1) % len(results)
    else:
        return
    plt.close(fig)
    show()


def show():
    global fig
    fig = render(idx)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


show()