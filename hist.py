import math
import cv2
import numpy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from metrics import get_CEF, get_AVG_Contrast


def histogramEqualization(image):
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    print(image.shape)
    # global r,g,b
    # r, g, b = cv2.split(img)
    wh = width*height

    img = np.float64(image)
    intensity = np.sum(img, axis=2) / 768
    img_out = np.zeros_like(img)
    for y in range(height):
        for x in range(width):
            som = something(intensity[y, x])
            alpha = som/intensity[y, x]
            # print(alpha)
            if 1 >= alpha >= 0:
                img[y, x, 0] = alpha * img[y, x, 0]
                img[y, x, 1] = alpha * img[y, x, 1]
                img[y, x, 2] = alpha * img[y, x, 2]
            elif alpha > 1:
                print("CMY")
                alpha2 = (3-som) / (3-intensity[y, x])
                C = 255 - img[y, x, 0]
                M = 255 - img[y, x, 1]
                Y = 255 - img[y, x, 2]
                C = alpha2 * C
                M = alpha2 * M
                Y = alpha2 * Y
                img[y, x, 0] = 255 - C
                img[y, x, 1] = 255 - M
                img[y, x, 2] = 255 - Y

    img_out = np.uint8(img)

    print("Done")

    print("Metrics:")
    get_CEF(image, img_out)
    get_AVG_Contrast(image, img_out)

    # img_out = cv2.merge((b, g, r))
    return img_out


def something(intensity):
    delta1 = 0
    delta2 = 1
    n = 2
    m = 0.7 #0.5
    alphaval = 0

    if delta1 <= intensity <= m:
        alphaval = delta1 + ((m - delta1)*(((intensity-delta1)/(m-delta1)) ** n))
    elif m <= intensity <= delta2:
        alphaval = delta1 - ((delta2 - m) * (((delta2 - intensity) / (delta2 - m)) ** n))

    return alphaval

