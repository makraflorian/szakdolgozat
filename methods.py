import cv2
import numpy as np

def invtry(image):
    im = image
    print(im.shape)

    # Inverz készítés keresőtáblával
    lut = np.arange(0, 256, 1, np.uint8)
    lut = 255 - lut
    im_inv = cv2.LUT(im, lut)



    return im_inv
