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
