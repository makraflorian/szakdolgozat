import cv2
import numpy as np


def invtry(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width, channels = img.shape
    print(img.shape)
    wh = width*height

    h, s, v = cv2.split(img)

    s_copy = s.copy()
    v_copy = v.copy()
    cdf_s_table = counter(s)
    cdf_v_table = counter(v)

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            cdf_s = cdf(s[i][j], wh, cdf_s_table)
            s_copy[i][j] = round((cdf_s * 255))
            cdf_v = (cdf(v[i][j], wh, cdf_v_table) / cdf_s)
            v_copy[i][j] = round((cdf_v * 255))

    s2 = cv2.equalizeHist(v)
    hsv_image = cv2.merge([h, s_copy, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


def counter(img):

    cdf_table = []
    for i in range(0, 256, 1):
        if i == 0:
            cdf_table.append(np.count_nonzero(img == i))
        else:
            cdf_table.append(np.count_nonzero(img == i) + cdf_table[i-1])

    return cdf_table


def cdf(x, allPixel, cdf_table):

    a = cdf_table[x]
    prob = a / allPixel
    return prob

