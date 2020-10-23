import numpy as np
from allennlp.common import Registrable
from torchvision import transforms

from dpeter.constants import WHITE_CONSTANT, VALUES_ABOVE_THIS_ARE_ALSO_WHITE, HEIGHT, WIDTH, NUM_CHANNELS


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
                transforms.RandomRotation(degrees=(-degree, degree), fill=WHITE_CONSTANT),
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
                    fill=WHITE_CONSTANT
                ),
                transforms.RandomRotation(degrees=(-degree, degree), fill=WHITE_CONSTANT),
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(self._transform(image))


@ImageAugmentator.register("google")
class GoogleAugmentator(ImageAugmentator):

    def __init__(self):
        super().__init__()

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomChoice(
                    [
                        transforms.Resize((int(HEIGHT * 1.04), int(WIDTH * 1.04)), interpolation=i) for i in range(6)
                    ]
                ),
                transforms.RandomCrop((HEIGHT, WIDTH)),
                transforms.RandomRotation(degrees=(-3, 2), fill=WHITE_CONSTANT),
                transforms.RandomChoice(
                    [
                        transforms.ColorJitter(contrast=0.1),
                        transforms.ColorJitter(brightness=0.05),
                        transforms.ColorJitter(saturation=0.1)
                    ]
                ),
                transforms.RandomChoice(
                    [
                        transforms.RandomPerspective(distortion_scale=0.1, p=1.0, interpolation=i, fill=WHITE_CONSTANT)
                        for i in [0, 2, 3]
                    ]
                )
            ]
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = np.array(self._transform(image))
        img[img >= VALUES_ABOVE_THIS_ARE_ALSO_WHITE] = WHITE_CONSTANT

        num_first_zeros = ((img[:, :, 0] != WHITE_CONSTANT).sum(axis=0) != 0).argmax()
        if num_first_zeros > 0:
            add_zeros = np.full((HEIGHT, num_first_zeros, NUM_CHANNELS), WHITE_CONSTANT).astype(np.uint8)
            img = np.concatenate((img[:, num_first_zeros:], add_zeros), axis=1)

        return img
