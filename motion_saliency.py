import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default=None,
                help="optional path to a video file; if omitted, use the webcam")
ap.add_argument("-u", "--url", type=str, default=None,
                help="streaming URL (e.g. DroidCam: http://<phone-ip>:4747/video)")
ap.add_argument("--synthetic", action="store_true",
                help="generate a synthetic moving-object video (no camera/file)")
ap.add_argument("--width", type=int, default=500,
                help="resize frames to this width to speed up processing")
args = vars(ap.parse_args())


def synthetic_frames(tot=150, width=500, height=375):
    i = 0
    while i < tot:
        frame = np.full((height, width, 3), 40, dtype="uint8")
        phases = [0, 90, 180]
        for idx, phase in enumerate(phases):
            cx = int(width * 0.25 + width * 0.5 * ((i + phase) % tot) / tot)
            cy = height // 2 + idx * 30
            cv2.circle(frame, (cx, cy), 25, (230, 180, 20), -1)
        i += 1
        yield frame


def frame_source():
    if args["synthetic"]:
        for f in synthetic_frames(width=args["width"]):
            yield f
        return

    if args["url"] is not None:
        cap = cv2.VideoCapture(args["url"])
    elif args["input"] is not None:
        cap = cv2.VideoCapture(args["input"])
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit(f"Could not open stream ({args['url'] or args['input'] or 'webcam 0'})")

    while True:
        (ok, frame) = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


saliency = None
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axFrame, axMap = axes
imFrame = axFrame.imshow(np.zeros((args["width"], args["width"], 3), dtype="uint8"))
imMap = axMap.imshow(np.zeros((args["width"], args["width"]), dtype="uint8"), cmap="gray", vmin=0, vmax=255)
axFrame.set_title("Frame")
axMap.set_title("Motion Saliency Map")
plt.ion()
plt.show()

try:
    for frame in frame_source():
        frame = cv2.resize(frame, (args["width"],
                                   int(frame.shape[0] * args["width"] / frame.shape[1])))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if saliency is None:
            saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
            saliency.setImagesize(frame.shape[1], frame.shape[0])
            saliency.init()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (success, saliencyMap) = saliency.computeSaliency(gray)
        if not success:
            continue
        saliencyMap = (saliencyMap * 255).astype("uint8")

        imFrame.set_data(rgb)
        imMap.set_data(saliencyMap)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.03)

except KeyboardInterrupt:
    pass

plt.ioff()
plt.close()