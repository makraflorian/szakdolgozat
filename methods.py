import cv2
import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt

# ezt hívjuk meg gombnyomásra a képpel
def retinex(image):
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width, channels = img.shape
    # print(img.shape)
    print(image.shape)

    # Multiscale Retinex eljárás hívása a képre
    newRGBImage = MSRCP(image, [15, 80, 250], 0.01, 0.99)

    print("Done!")

    return newRGBImage


def MSRCP(img, sigma_list, s1, s2):
    height, width, channels = img.shape

    # lebegőpontos számokkal tudjunk számolni, és +1 hogy elkerüljük a crash-eket
    img = np.float64(img) + 1.0

    # a három channel átlaga = intensity
    intensity = np.sum(img, axis=2) / 3

    msr = multiscaleRetinex(intensity, sigma_list)

    intensity1 = simplestColorBalance(msr, s1, s2)
    # rescale
    intensity1 = (intensity1 - np.min(intensity1)) / (np.max(intensity1) - np.min(intensity1)) * 255.0 + 1.0

    # ugyanolyan shape-el és tipussal csak 0-kat tartalmazó matrix
    img_msrcp = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            B = np.max(img[y, x])
            A = np.minimum(255.0 / B, intensity1[y, x] / intensity[y, x])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    # visszaalakítjuk 8bites egészeket tartalmazó mátrixra
    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


def gaussianConvolution(intensity, sigma):

    # intenzitás - Gauss konvolucio a sigma parameterrel (nem vagyok benne biztos hogy log10 vagy sima log, kicsit eltérő eredményt ad)
    diff = np.log10(intensity) - np.log10(cv2.GaussianBlur(src=intensity, ksize=(0, 0), sigmaX=sigma))

    return diff


def multiscaleRetinex(intensity, sigmaList):

    msr = np.zeros_like(intensity)
    # konvolució mindegyik sigma paraméterrel és ezen eredmények átlagolása
    for sigma in sigmaList:
        msr += gaussianConvolution(intensity, sigma)

    msr = msr / 3
    return msr


def simplestColorBalance(img, s1, s2):

    height, width = img.shape
    pixelcount = width * height

    unique = np.unique(img[:, :])
    low_val = 0
    high_val = 255

    current = 0
    for u in unique:
        if float(current) / pixelcount < s1:
            low_val = u
        if float(current) / pixelcount < s2:
            high_val = u
        current += 1

    img[:, :] = np.maximum(np.minimum(img[:, :], high_val), low_val)

    return img

