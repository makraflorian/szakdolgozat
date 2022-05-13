import cv2
import numpy as np
from metrics import get_CEF, get_AVG_Contrast


def multiscaleMorphology(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print("Multiscale Morphology, image shape: " + str(image.shape))

    # openings
    op1 = open(img, 3)
    op2 = open(img, 5)

    # closings
    clos1 = close(img, 3)
    clos2 = close(img, 5)

    # White-Top-Hat
    wth1 = cv2.subtract(img, op1)
    wth2 = cv2.subtract(img, op2)

    # Black-Top-Hat
    bth1 = cv2.subtract(clos1, img)
    bth2 = cv2.subtract(clos2, img)

    # wth diff - WTHV img
    wthv1 = cv2.subtract(wth2, wth1)


    # bth diff - BTHV img
    bthv1 = cv2.subtract(bth2, bth1)

    # WTH_MAX
    wthmax1 = vectorMax(wth1, wth2)


    # BTH_MAX
    bthmax1 = vectorMax(bth1, bth2)

    # last add and sub tract
    white = cv2.add(wthmax1, wthv1)
    black = cv2.add(bthmax1, bthv1)
    diff = cv2.subtract(white, black)
    out = cv2.add(img, diff)

    h2, s2, v2 = cv2.split(out)
    h2 = cv2.normalize(h2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    s2 = cv2.normalize(s2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    v2 = cv2.normalize(v2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # morphology for hue noise
    kernel = np.ones((3, 3), 'uint8')
    h2 = cv2.erode(h2, kernel, iterations=1)
    h2 = cv2.dilate(h2, kernel, iterations=1)

    out = cv2.merge((h2, s2, v2))

    img_bgr = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    print("Done")

    print("Metrics:")
    get_CEF(image, img_bgr)
    get_AVG_Contrast(image, img_bgr)

    return img_bgr


def dilate(img, k):

    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    # SE = np.ones((k, k), dtype=np.uint8)

    constant = (k - 1) // 2

    dilated = img.copy()

    # dilate without using inbuilt cv2 function for morphology
    for i in range(constant, m - constant):
        for j in range(constant, n - constant):
            temp = img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp
            max_pixel = [255, 255, 255]
            for x in range(k):
                for y in range(k):
                    if product[x,y,2] > product[constant,constant,2]:
                        max_pixel = product[x,y]
                    elif product[x,y,2] == product[constant,constant,2] and product[x,y,1] > product[constant,constant,1]:
                        max_pixel = product[x, y]
                    elif product[x,y,2] == product[constant,constant,2] and product[x,y,1] == product[constant,constant,1] and product[x, y, 0] > product[constant, constant, 0]:
                        max_pixel = product[x, y]
                    else:
                        max_pixel = product[constant, constant]
            dilated[i, j] = max_pixel

    return dilated


def erode(img, k):

    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    # SE = np.ones((k, k), dtype=np.uint8)

    constant = (k - 1) // 2

    imgErode = i= img.copy()

    # Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m - constant):
        for j in range(constant, n - constant):
            temp = img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp
            min_pixel = [255, 255, 255]
            for x in range(k):
                for y in range(k):
                    if product[x,y,2] < product[constant,constant,2]:
                        min_pixel = product[x,y]
                    elif product[x,y,2] == product[constant,constant,2] and product[x,y,1] < product[constant,constant,1]:
                        min_pixel = product[x, y]
                    elif product[x,y,2] == product[constant,constant,2] and product[x,y,1] == product[constant,constant,1] and product[x, y, 0] < product[constant, constant, 0]:
                        min_pixel = product[x, y]
                    else:
                        min_pixel = product[constant, constant]
            imgErode[i, j] = min_pixel

    return imgErode


def vectorMax(img1, img2):
    m, n, channels = img1.shape

    for x in range(m):
        for y in range(n):
            if img1[x, y, 2] < img2[x, y, 2]:
                img1[x, y] = img2[x, y]
            elif img1[x, y, 2] == img2[x, y, 2] and img1[x, y, 1] < img2[x, y, 1]:
                img1[x, y] = img2[x, y]
            elif img1[x, y, 2] == img2[x, y, 2] and img1[x, y, 1] == img2[x, y, 1] and img2[x, y, 0] < img2[x, y, 0]:
                img1[x, y] = img2[x, y]
            else:
                img1[x, y] = img1[x, y]

    return img1


def open(img, k):
    img_erosion = erode(img, k)
    img_dilation = dilate(img_erosion, k)

    return img_dilation


def close(img, k):
    img_dilation = dilate(img, k)
    img_erosion = erode(img_dilation, k)

    return img_erosion
