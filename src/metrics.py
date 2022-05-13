import math
import cv2
import numpy as np

def metrics(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)

    gamma = r - g
    beta = 1 / 2 * (r + g) - b

    ro_gamma = gamma.std()
    ro_beta = beta.std()

    mu_gamma = np.mean(gamma)
    mu_beta = np.mean(beta)

    cm = math.sqrt(ro_gamma ** 2 + ro_beta ** 2) + math.sqrt(mu_gamma ** 2 + mu_beta ** 2)
    return cm

def get_CEF(original, enhanced):

    enhanced_metrics = metrics(enhanced)
    original_metrics = metrics(original)

    print("CEF:" + str(enhanced_metrics / original_metrics))


def metrics2(img):

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(lab)

    # get min and max
    kernel = np.ones((5, 5), np.uint8)
    min = cv2.erode(L, kernel, iterations=1)
    max = cv2.dilate(L, kernel, iterations=1)

    # convert to float
    min1 = cv2.normalize(min, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    max1 = cv2.normalize(max, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

    max1 = max1 + 1
    # get local contrast
    contrast = cv2.divide((cv2.subtract(max1, min1)),cv2.add(max1, min1))

    # get average global
    average_contrast = 100 * np.mean(np.abs(contrast))

    return average_contrast

def get_AVG_Contrast(original, enhanced):

    enhanced_metrics = metrics2(enhanced)
    original_metrics = metrics2(original)

    print("AVG_Contrast:")
    print("Original: " + "%.2f" % original_metrics + "%")
    print("Enhanced: " + "%.2f" % enhanced_metrics + "%")