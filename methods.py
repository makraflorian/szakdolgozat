import math

import cv2
import numpy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def invtry(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape
    print(img.shape)
    # global r,g,b
    r, g, b = cv2.split(img)
    wh = width*height

    # pil_image = Image.fromarray(image)
    # pil_channels = pil_image.split()
    #
    # normalizedRedChannel = pil_channels[0].point(normalizeRed)
    # normalizedGreenChannel = pil_channels[1].point(normalizeGreen)
    # normalizedBlueChannel = pil_channels[2].point(normalizeBlue)
    #
    # normalizedImage = Image.merge("RGB", (normalizedRedChannel, normalizedGreenChannel, normalizedBlueChannel))
    # im_np = np.asarray(normalizedImage) #return

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            intensity = (int(r[i][j]) + int(g[i][j]) + int(b[i][j])) / 768
            som = something(intensity)
            alpha = som/intensity
            print(alpha)
            if alpha <= 1 and alpha >= 0:
                r[i][j] = math.floor(r[i][j] * alpha)
                g[i][j] = math.floor(g[i][j] * alpha)
                b[i][j] = math.floor(b[i][j] * alpha)


    newRGBImage = cv2.merge((b, g, r))
    return newRGBImage


def something(intensity):
    delta1 = 0
    delta2 = 1
    n = 2
    m = 0.5
    alphaval = 0

    if delta1 <= intensity <= m:
        alphaval = delta1 + ((m - delta1)*(((intensity-delta1)/(m-delta1)) ** n))
    elif m <= intensity <= delta2:
        alphaval = delta1 - ((delta2 - m) * (((delta2 - intensity) / (delta2 - m)) ** n))

    return alphaval


def normalizeRed(intensity):

    minInput = r.min() * 2
    maxInput = r.max()
    minOutput = 0
    maxOutput = 255

    normalized = (intensity - minInput) * (((maxOutput - minOutput) / (maxInput - minInput)) + minOutput)

    return normalized


def normalizeGreen(intensity):

    minInput = g.min() * 2
    maxInput = g.max()
    minOutput = 0
    maxOutput = 255

    normalized = (intensity - minInput) * (((maxOutput - minOutput) / (maxInput - minInput)) + minOutput)

    return normalized


def normalizeBlue(intensity):

    minInput = b.min() * 2
    maxInput = b.max()
    minOutput = 0
    maxOutput = 255

    normalized = (intensity - minInput) * (((maxOutput - minOutput) / (maxInput - minInput)) + minOutput)

    return normalized


# def counter(img):
#
#     cdf_table = []
#
#
#     # Iterate through all values.
#     for x in range(0, 255):
#         for y in range(0, 255):
#             cdf_table.append(np.count_nonzero(img == [x, y]))
#
#     for x in cdf_table:
#         cdf_table.append(np.count_nonzero(img == x))
#
#     return cdf_table
#
#
# def cdf(x, allPixel, cdf_table):
#
#     a = cdf_table[x]
#     prob = a / allPixel
#     return prob














