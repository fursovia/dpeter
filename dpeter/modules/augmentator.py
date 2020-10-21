import numpy as np
from allennlp.common import Registrable
from torchvision import transforms


class ImageAugmentator(Registrable):

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass


@ImageAugmentator.register("null")
class NullAugmentator(ImageAugmentator):

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


@ImageAugmentator.register("rotation")
class RotationAugmentator(ImageAugmentator):

    def __init__(self, degree: int = 5):
        super().__init__()
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=(-degree, degree), fill=255),
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(self._transform(image))


@ImageAugmentator.register("perspective_rotation")
class PerspectiveRotationAugmentator(ImageAugmentator):

    def __init__(self, degree: int = 4, distortion_scale: float = 0.25, p: float = 0.7, interpolation: int = 2):
        super().__init__()
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomPerspective(
                    distortion_scale=distortion_scale,
                    p=p,
                    interpolation=interpolation,
                    fill=255
                ),
                transforms.RandomRotation(degrees=(-degree, degree), fill=255),
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(self._transform(image))
