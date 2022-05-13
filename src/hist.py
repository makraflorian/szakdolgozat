import numpy as np
from metrics import get_CEF, get_AVG_Contrast


def histogramEqualization(image):

    print("Hue-Preserving Histogram Equalization, image shape: " + str(image.shape))

    height, width, channels = image.shape
    img = np.float64(image)
    intensity = np.sum(img, axis=2) / 768

    for y in range(height):
        for x in range(width):
            som = fFunction(intensity[y, x])
            alpha = som/intensity[y, x]

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

    return img_out


def fFunction(intensity):
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



