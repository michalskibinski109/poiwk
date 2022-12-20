import cv2
import os
from pathlib import Path
from tqdm import tqdm


def split_video(video_path):

    vidcap = cv2.VideoCapture(str(video_path.resolve()))
    success, image = vidcap.read()
    count = 0
    bar = tqdm(total=vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    while success:
        # cv2.imwrite(f"frame{count}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        images.append(image)
        count += 1
        bar.update(1)
    bar.close()

    return images


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


def prepare_images(images):
    # here goes your code. Images must be prepared for model: resize, grayscale, etc.
    return images


def run_model(images):
    # here goes your code. Images must be processed by model. every image must be replaced with processed image
    return images


def run(video_path):
    images = split_video(video_path)
    images = prepare_images(images)
    images = run_model(images)
    merge_images(images, video_path)


if __name__ == "__main__":
    run(Path("videos") / "f.mp4")
