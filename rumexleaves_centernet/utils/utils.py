import numpy as np


def img_tensor_to_numpy(img):
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    img = img.cpu().numpy() * 255
    img = img.astype(np.uint8)
    return img
