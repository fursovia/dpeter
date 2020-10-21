from torchvision import models
import torch


def get_inception_encoder():
    # for (128, 1024) image size the output is (288, 13, 125)
    # 13 * 125 = 1625
    inception = models.inception_v3()
    # Mixed_5d
    encoder = torch.nn.Sequential(*list(inception.children())[:10])
    return encoder
