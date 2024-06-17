import cv2
import numpy as np
from .functions import add_noise_to_image, HSV_image


class ImageAugmentationBuilder:
    def __init__(self):
        self.config = {
            'HSV': None,
            'noise_level': None,
            'rotate_angle': None,
            'resize_scale': None,
            'sharpen_factor': None
        }

    def HSV(self, intensity_factor):
        self.config['HSV'] = intensity_factor
        return self

    def noise(self, noise_level):
        if 0 > noise_level or noise_level >= 1:
            raise ValueError("noise level must be in range by 0 to 1 ")
        self.config['noise_level'] = noise_level
        return self

    def rotate(self, angle):
        self.config['rotate_angle'] = angle
        return self

    def resize(self, scale):
        self.config['resize_scale'] = scale
        return self

    def sharpen(self, factor):
        self.config['sharpen_factor'] = factor
        return self

    def build(self):
        return self.config


def augment_image(image, label, augmentation_config):
    augmented_image = image.copy()
    augmented_label = label

    if augmentation_config['HSV']:
        intensity_factor = augmentation_config['HSV']
        augmented_image, label = HSV_image(augmented_image, label, intensity_factor)

    if augmentation_config['noise_level']:
        noise_level = augmentation_config['noise_level']
        augmented_image, augmented_label = add_noise_to_image(augmented_image, label, noise_level)
    # Проходимся по каждому аугменту и применяем его, если он указан в списке augmentations
    # for augmentation in augmentation_config:
    # if augmentation == 'flip_horizontal':
    #     augmented_image = cv2.flip(augmented_image, 1)
    # elif augmentation == 'flip_vertical':
    #     augmented_image = cv2.flip(augmented_image, 0)
    if augmentation_config['rotate_angle']:
        angle = augmentation_config['rotate_angle']
        height, width = augmented_image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        augmented_image = cv2.warpAffine(augmented_image, rotation_matrix, (width, height))
        # elif augmentation.startswith('resize_'):
        #     scale = float(augmentation.split('_')[1])
        #     augmented_image = cv2.resize(augmented_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # elif augmentation.startswith('brightness_'):
        #     brightness = float(augmentation.split('_')[1])
        # augmented_image = np.clip(augmented_image + brightness, 0, 255).astype(np.uint8)

    return augmented_image, augmented_label


def augment_with_yolo_txt(image_path: str, txt_path: str, augmentation_config) -> (
        bytes, str):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"img by path {image_path} not found")

    augmented_image, augmented_label = augment_image(image, "someThing", augmentation_config)

    cv2.imshow('Original Image', image)
    cv2.imshow('Augmented Image', augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# config = ImageAugmentationBuilder().noise(0.0).HSV(4).build()
#
# augment_with_yolo_txt(r'../tests/images/archive/train/images/image1_0_png.rf.c864ea1d0d7a6ba347b3bd6fa51bf4c9.jpg',
#                       r'../tests/images/archive/train/labels/image1_0_png.rf.c864ea1d0d7a6ba347b3bd6fa51bf4c9.txt',
#                       config)
