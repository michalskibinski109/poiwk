import cv2
import os
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np
from miskibin import get_logger

logger = get_logger(lvl=10)


def split_video(video_path):

    vidcap = cv2.VideoCapture(str(video_path.resolve()))
    success, image = vidcap.read()
    count = 0
    bar = tqdm(total=vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    while success:
        # cv2.imwrite(f"frame{count}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        images.append(np.array(image))
        count += 1
        bar.update(1)
    bar.close()
    logger.info(f"Read {count} frames each of shape {images[0].shape}")
    return np.array(images)


def merge_images(images, path="project.mp4", fps=30):
    height, width, layers = images[0].shape
    video = cv2.VideoWriter(path, 0, fps, (width, height))
    bar = tqdm(total=len(images))
    for image in images:
        video.write(image)
        bar.update(1)
    bar.close()
    cv2.destroyAllWindows()
    video.release()


def prepare_images(images, size=(32, 32)):
    # here goes your code. Images must be prepared for model: resize, grayscale, etc.
    for i, image in enumerate(images):
        # image = cv2.resize(image, (32, 32))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = np.expand_dims(image, axis=0)
        images[i] = np.array(image)
    return np.array(images)


def run_model(images: np.ndarray):
    # here goes your code. Images must be processed by model. every image must be replaced with processed image
    model = load_model("model.h5")
    for i, image in enumerate(images):
        image = cv2.resize(image, (32, 32))

        # expand dimensions so that it represents a single 'sample'
        image = np.expand_dims(image, axis=0)
        # resize image
        predicted = model.predict(image)
        bbox = predicted[1][0] * 255
        print(bbox[0], bbox[1], bbox[2], bbox[3])
        # draw bounding box on the image
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
        images[i] = image

        logger.debug(bbox)
    return images


def run(video_path):
    images = split_video(video_path)
    images = prepare_images(images)
    images = run_model(images)
    merge_images(images, video_path)


if __name__ == "__main__":
    run(Path("videos") / "f.mp4")
