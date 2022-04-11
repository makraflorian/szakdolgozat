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

    print(enhanced_metrics / original_metrics)
