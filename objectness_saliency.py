import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from saliency_utils import resolve_image

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to BING objectness saliency model")
ap.add_argument("-i", "--image", default=None,
                help="path to input image (default: newest file in images/)")
ap.add_argument("-n", "--max-detections", type=int, default=10,
                help="maximum # of detections to examine")
args = vars(ap.parse_args())

image = cv2.imread(resolve_image(args["image"], "images"))
if image is None:
    raise SystemExit(f"Could not read image: {args['image']}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath(args["model"])

(success, saliencyMap) = saliency.computeSaliency(image)
if not success:
    raise SystemExit("Objectness saliency computation failed")

numDetections = saliencyMap.shape[0] if len(saliencyMap.shape) > 1 else 1
numDetections = min(numDetections, args["max_detections"])

cols = 2
rows = int(np.ceil(numDetections / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = np.atleast_1d(axes).ravel()

for i in range(numDetections):
    det = saliencyMap[i].flatten() if numDetections > 1 else saliencyMap.flatten()
    (startX, startY, endX, endY) = det

    output = image.copy()
    color = (np.random.randint(0, 255), np.random.randint(0, 255),
             np.random.randint(0, 255))
    output = cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    axes[i].imshow(output)
    axes[i].set_title(f"Proposal #{i + 1}")
    axes[i].axis("off")

for blank in range(numDetections, len(axes)):
    axes[blank].axis("off")

fig.suptitle(f"{numDetections} objectness proposals")
fig.tight_layout()
plt.show()