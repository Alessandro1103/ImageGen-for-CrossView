import cv2
import numpy as np


# Lo usiamo solo per le ground images

def apply_canny(image):
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image

    img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
    if edges.max() > 1.0:
        edges = edges / 255.0
    return edges


def concatenate_images(image, edgeImg):
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image

    if not isinstance(edgeImg, np.ndarray):
        edgeImg_np = np.array(edgeImg)
    else:
        edgeImg_np = edgeImg

    edgeImg_np = np.expand_dims(edgeImg_np, axis=-1)
    concatenated = np.concatenate((image_np, edgeImg_np), axis=-1)
    return concatenated
