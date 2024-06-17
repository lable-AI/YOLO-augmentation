import cv2
import numpy as np

def HSV_image(image, label, intensity_factor=4):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    increased_value_channel = np.clip(value_channel * intensity_factor, 0, 255).astype(np.uint8)
    hsv_image[:, :, 2] = increased_value_channel
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image, label

def add_noise_to_image(image, label, noise_level):
    """
    Добавляет гауссов шум к изображению.

    Parameters:
    - image: входное изображение (numpy array).
    - noise_level: коэффициент шума от 0 до 1.

    Returns:
    - Изображение с добавленным шумом.
    """

    # Размер изображения
    h, w, c = image.shape

    mean = 0
    sigma = noise_level * 255
    gauss = np.random.normal(mean, sigma, (h, w, c)).astype('uint8')

    noisy_image = cv2.addWeighted(image, 1 - noise_level, gauss, noise_level, 0)

    return noisy_image, label
