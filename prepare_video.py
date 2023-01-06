import cv2
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import load_model
import numpy as np
from miskibin import get_logger


class ModelRunner:
    def __init__(
        self,
        logger=get_logger(lvl=10),
        model_path: Path = Path("model.h5"),
        model_input_shape=(32, 32),
    ):
        self.model = load_model(str(model_path.resolve()))
        self.logger = logger
        self.model_input_shape = model_input_shape

    def run(self, video_path: Path, fps: int = 30) -> str:
        """Run model on video and save result to mp4 file.
        args:
            `video_path`: path to video file. Also used as output file name.
            `fps`: frames per second for output video
        returns:
            `str`: path to output file
        """
        images = self.__split_video(video_path)
        images = self.__run_model(images)
        self.__merge_images(images, str(video_path).replace(".avi", ".mp4"), fps)
        return str(video_path).replace(".avi", ".mp4")

    def __split_video(self, video_path):
        vidcap = cv2.VideoCapture(str(video_path.resolve()))
        success, image = vidcap.read()
        count = 0
        bar = tqdm(total=vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []
        while success:
            # cv2.imwrite(f"frame{count}.jpg", image)  # save frame as JPEG file
            success, image = vidcap.read()
            if image is not None:
                images.append(np.array(image))
                count += 1
            bar.update(1)
        bar.close()
        self.logger.info(f"Read {count} frames each of shape {images[0].shape}")
        arr = np.zeros(
            (count, images[0].shape[0], images[0].shape[1], images[0].shape[2]),
            dtype=np.uint8,
        )
        for i, image in enumerate(images):
            if image is None:
                break
            arr[i] = image
        return arr

    def __prepare_images(
        self,
        old_images,
    ):
        # here goes your code. Images must be prepared for model: resize, grayscale, etc.
        new_images = []
        grey = (100, 100, 100)
        light_grey = (146, 145, 149)
        for i, img in enumerate(old_images):
            mask = cv2.inRange(img, grey, light_grey)
            result = cv2.bitwise_and(img, img, mask=mask)
            # grey scale
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = result / 255
            result = cv2.resize(result, self.model_input_shape)
            new_images.append(np.array(result))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = np.expand_dims(image, axis=0)
        return np.array(new_images)

    def __run_model(self, images: np.ndarray):
        # here goes your code. Images must be processed by model. every image must be replaced with processed image
        new_images = self.__prepare_images(images)
        IMG_SIZE = images[0].shape[0]
        bar = tqdm(total=len(images))
        for i, image in enumerate(new_images):
            predicted = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
            bbox = predicted[1][0]
            x, y, w, h = bbox * IMG_SIZE
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
            try:
                image = cv2.rectangle(images[i], start_point, end_point, (255, 0, 0), 3)
            except ValueError as exc:
                self.logger.debug(f"ValueError {exc}")
            images[i] = image
            bar.update(1)
            self.logger.debug(bbox)
        bar.close()
        return images

    def __merge_images(self, images, path, fps=10):
        height, width = images[0].shape[0], images[0].shape[1]
        self.logger.warning(
            f"Video will be saved to {path} with {fps} fps and {width}x{height} size total {images[0].shape}"
        )
        fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
        # video in grey scale
        video = cv2.VideoWriter(path, fourcc, fps, (width, height), True)
        bar = tqdm(total=len(images))
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
            bar.update(1)
        bar.close()
        cv2.destroyAllWindows()
        video.release()
