import cv2
import numpy as np
from metrics import get_CEF, get_AVG_Contrast


def retinexOnIntensity(image):

    print("MSRetinex with Chromaticity Preservation, image shape: " + str(image.shape))

    # Multiscale Retinex
    newRGBImage = MSRCP(image, [15, 80, 250], 0.02, 0.99)

    print("Done!")

    print("Metrics:")
    get_CEF(image, newRGBImage)
    get_AVG_Contrast(image, newRGBImage)

    return newRGBImage


def retinexOnChannels(image):

    print("MSRetinex with Color Restoration, image shape: " + str(image.shape))

    # Multiscale Retinex
    newRGBImage = MSRCR(image, [15, 80, 250], 192.0, -30.0, 125.0, 46.0, 0.01, 0.99)

    print("Done!")

    print("Metrics:")
    get_CEF(image, newRGBImage)
    get_AVG_Contrast(image, newRGBImage)

    return newRGBImage


# =======================================================================================================
def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0

    img_retinex = multiscaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * 255

    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))

    return img_msrcr


# =======================================================================================================
def MSRCP(img, sigma_list, s1, s2):
    height, width, channels = img.shape

    # prevent crash
    img = np.float64(img) + 1.0

    # intensity
    intensity = np.sum(img, axis=2) / 3

    msr = multiscaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    msr = np.expand_dims(msr, 2)

    intensity1 = simplestColorBalance(msr, s1, s2)
    # rescale
    intensity1 = (intensity1 - np.min(intensity1)) / (np.max(intensity1) - np.min(intensity1)) * 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            B = np.max(img[y, x])
            A = np.minimum(255.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


def gaussianConvolution(intensity, sigma):

    # Single Scale Retinex
    diff = np.log10(intensity) - np.log10(cv2.GaussianBlur(src=intensity, ksize=(0, 0), sigmaX=sigma))

    return diff


def multiscaleRetinex(intensity, sigmaList):

    msr = np.zeros_like(intensity)

    for sigma in sigmaList:
        msr += gaussianConvolution(intensity, sigma)

    msr = msr / 3
    return msr


def colorRestoration(img, alpha, beta):

    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, s1, s2):

    height, width, channels = img.shape
    pixelcount = width * height

    low_val = 0
    high_val = 255

    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / pixelcount < s1:
                low_val = u
            if float(current) / pixelcount < s2:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img

