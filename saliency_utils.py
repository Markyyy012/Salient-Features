import glob
import os

import cv2


def resolve_image(path=None, folder="images"):
    if path is None or path == "":
        candidates = [p for p in glob.glob(os.path.join(folder, "*"))
                      if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        if not candidates:
            raise SystemExit(f"No images found in {folder}/ — pass --image <path>")
        path = max(candidates, key=os.path.getmtime)
        print(f"No --image given; using newest in {folder}/: {os.path.basename(path)}")
    return path


def normalize_map(saliencyMap):
    if saliencyMap.dtype != "uint8":
        saliencyMap = cv2.normalize(saliencyMap, None, 0, 255,
                                    cv2.NORM_MINMAX).astype("uint8")
    return saliencyMap


def compute_static(image):
    spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
    (_, spectralMap) = spectral.computeSaliency(image)
    spectralMap = normalize_map(spectralMap)

    fine = cv2.saliency.StaticSaliencyFineGrained_create()
    (_, fineMap) = fine.computeSaliency(image)
    fineMap = normalize_map(fineMap)

    return spectralMap, fineMap


def fine_threshold(fineMap):
    (_, threshMap) = cv2.threshold(fineMap, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshMap = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel, iterations=2)
    threshMap = cv2.morphologyEx(threshMap, cv2.MORPH_OPEN, kernel, iterations=1)
    return threshMap


def extract_rois(threshMap, min_area=500):
    contours = cv2.findContours(threshMap, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    rois = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        rois.append((x, y, w, h))
    return rois


def draw_rois(image, rois, color=(255, 0, 0)):
    output = image.copy()
    for (x, y, w, h) in rois:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    return output