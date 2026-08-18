import argparse
import cv2
import matplotlib.pyplot as plt

from saliency_utils import (compute_static, fine_threshold, extract_rois,
                            draw_rois, resolve_image)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=None,
                help="path to input image (default: newest file in images/)")
args = vars(ap.parse_args())

image = cv2.imread(resolve_image(args["image"], "images"))
if image is None:
    raise SystemExit(f"Could not read image: {args['image']}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(spectralMap, fineMap) = compute_static(image)
threshMap = fine_threshold(fineMap)

minArea = 0.005 * image.shape[0] * image.shape[1]
rois = extract_rois(threshMap, min_area=minArea)
roiImg = draw_rois(image, rois)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(image)
axes[0, 0].set_title("Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(spectralMap, cmap="gray")
axes[0, 1].set_title("Spectral Residual Saliency")
axes[0, 1].axis("off")

axes[0, 2].imshow(fineMap, cmap="gray")
axes[0, 2].set_title("Fine Grained Saliency")
axes[0, 2].axis("off")

axes[1, 0].imshow(threshMap, cmap="gray")
axes[1, 0].set_title("Fine Grained (Otsu Threshold)")
axes[1, 0].axis("off")

axes[1, 1].imshow(roiImg)
axes[1, 1].set_title(f"{len(rois)} ROI(s) from contours")
axes[1, 1].axis("off")

axes[1, 2].axis("off")

fig.tight_layout()
plt.show()