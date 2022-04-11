import math

import cv2
import numpy
import numpy as np
from math import pi
from PIL import Image
from matplotlib import pyplot as plt
from metrics import get_CEF

def multiscaleMorphology(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    height, width, channels = image.shape
    print(image.shape)
    # global r,g,b
    # r, g, b = cv2.split(img)

    # Acquire size of the image
    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    k = 3
    SE = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2
    # Define new image


    op1 = open(img, 3)
    op2 = open(img, 5)
    op3 = open(img, 7)
    op4 = open(img, 9)
    op5 = open(img, 11)
    op6 = open(img, 13)
    op7 = open(img, 15)

    clos1 = close(img, 3)
    clos2 = close(img, 5)
    clos3 = close(img, 7)
    clos4 = close(img, 9)
    clos5 = close(img, 11)
    clos6 = close(img, 13)
    clos7 = close(img, 15)

    wth1 = cv2.subtract(img, op1)
    wth2 = cv2.subtract(img, op2)
    wth3 = cv2.subtract(img, op3)
    wth4 = cv2.subtract(img, op4)
    wth5 = cv2.subtract(img, op5)
    wth6 = cv2.subtract(img, op6)
    wth7 = cv2.subtract(img, op7)

    bth1 = cv2.subtract(clos1, img)
    bth2 = cv2.subtract(clos2, img)
    bth3 = cv2.subtract(clos3, img)
    bth4 = cv2.subtract(clos4, img)
    bth5 = cv2.subtract(clos5, img)
    bth6 = cv2.subtract(clos6, img)
    bth7 = cv2.subtract(clos7, img)

    wthv1 = cv2.subtract(wth2, wth1)
    wthv2 = cv2.subtract(wth3, wthv1)
    wthv3 = cv2.subtract(wth4, wthv2)
    wthv4 = cv2.subtract(wth5, wthv3)
    wthv5 = cv2.subtract(wth6, wthv4)
    wthv6 = cv2.subtract(wth7, wthv5)

    bthv1 = cv2.subtract(bth2, bth1)
    bthv2 = cv2.subtract(bth3, bthv1)
    bthv3 = cv2.subtract(bth4, bthv2)
    bthv4 = cv2.subtract(bth5, bthv3)
    bthv5 = cv2.subtract(bth6, bthv4)
    bthv6 = cv2.subtract(bth7, bthv5)

    wthmax1 = vectorMax(wth1, wth2)
    wthmax2 = vectorMax(wth3, wth4)
    wthmax2_5 = vectorMax(bth6, bth7)
    asd = vectorMax(wth5, wthmax2_5)
    wthmax3 = vectorMax(wthmax1, wthmax2)
    wthmaxLast = vectorMax(wthmax3, asd)

    bthmax1 = vectorMax(bth1, bth2)
    bthmax2 = vectorMax(bth3, bth4)
    bthmax2_5 = vectorMax(bth6, bth7)
    asd = vectorMax(bth5, bthmax2_5)
    bthmax3 = vectorMax(bthmax1, bthmax2)
    bthmaxLast = vectorMax(bthmax3, asd)

    wthvmax1 = vectorMax(wthv1, wthv2)
    wthvmax2 = vectorMax(wthv3, wthv4)
    wthvmax3 = vectorMax(wthv5, wthv6)
    wthvmaxLast1 = vectorMax(wthvmax1, wthvmax2)
    wthvmaxLast = vectorMax(wthvmaxLast1, wthvmax3)

    bthvmax1 = vectorMax(bthv1, bthv2)
    bthvmax2 = vectorMax(bthv3, bthv4)
    bthvmax3 = vectorMax(bthv5, bthv6)
    bthvmaxLast1 = vectorMax(bthvmax1, bthvmax2)
    bthvmaxLast = vectorMax(bthvmaxLast1, bthvmax3)

    elso = cv2.add(wthmaxLast, wthvmaxLast)
    masodik = cv2.add(bthmaxLast, bthvmaxLast)
    harmadik = cv2.subtract(elso, masodik)
    out = cv2.add(img, harmadik)


    img_bgr = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    print("Done")

    get_CEF(img, out)
    # img_bgr = cv2.cvtColor(back, cv2.COLOR_HLS2BGR)
    # img_bgr = HSI2RGB(back)
    # img_bgr = np.uint8(img_bgr)

    return img_bgr


def dilate(img, k):

    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element

    # SE = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2

    dilated = np.zeros_like(img, dtype=np.uint8)
    for i in range(constant, m - constant):
        for j in range(constant, n - constant):
            temp = img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp
            asd = [255, 255, 255]
            for x in range(k):
                for y in range(k):
                    if product[x,y,2] > product[constant,constant,2]:
                        asd = product[x,y]
                        break
                    elif product[x,y,1] > product[constant,constant,1]:
                        asd = product[x, y]
                        break
                    elif product[x, y, 0] > product[constant, constant, 0]:
                        asd = product[x, y]
                        break
                    else:
                        asd = product[constant, constant]
            dilated[i, j] = asd

    return dilated


def erode(img, k):

    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    # SE = np.ones((k, k), dtype=np.uint8)
    constant = (k - 1) // 2
    # Define new image
    imgErode = np.zeros_like(img, dtype=np.uint8)
    # Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m - constant):
        for j in range(constant, n - constant):
            temp = img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            product = temp
            asd = [255, 255, 255]
            for x in range(k):
                for y in range(k):
                    if product[x,y,2] < product[constant,constant,2]:
                        asd = product[x,y]
                        break
                    elif product[x,y,1] < product[constant,constant,1]:
                        asd = product[x, y]
                        break
                    elif product[x, y, 0] < product[constant, constant, 0]:
                        asd = product[x, y]
                        break
                    else:
                        asd = product[1, 1]
            imgErode[i, j] = asd

    return imgErode


def vectorMax(img1, img2):
    m, n, channels = img1.shape

    for x in range(m):
        for y in range(n):
            if img1[x, y, 2] < img2[x, y, 2]:
                img1[x, y] = img2[x, y]
                break
            elif img1[x, y, 2] > img2[x, y, 2]:
                break
            elif img1[x, y, 1] < img1[x, y, 1]:
                img1[x, y] = img2[x, y]
                break
            elif img1[x, y, 1] > img1[x, y, 1]:
                break
            elif img1[x, y, 0] < img1[x, y, 0]:
                img1[x, y] = img2[x, y]
                break
            elif img1[x, y, 0] > img1[x, y, 0]:
                break
            else:
                img1[x, y] = img1[x, y]

    return img1



def open(img, k):
    img_erosion = erode(img, k)
    img_dilation = dilate(img_erosion, k)

    return img_dilation

def open5x5(img):
    img_erosion = erode(img, 5)
    img_dilation = dilate(img_erosion, 5)

    return img_dilation

def open7x7(img, kernel_addr):
    img_erosion = cv2.erode(img, kernel_addr, iterations=3)
    img_dilation = cv2.dilate(img_erosion, kernel_addr, iterations=3)

    return img_dilation

def open9x9(img, kernel_addr):
    img_erosion = cv2.erode(img, kernel_addr, iterations=4)
    img_dilation = cv2.dilate(img_erosion, kernel_addr, iterations=4)

    return img_dilation

def open11x11(img, kernel_addr):
    img_erosion = cv2.erode(img, kernel_addr, iterations=5)
    img_dilation = cv2.dilate(img_erosion, kernel_addr, iterations=5)

    return img_dilation

def open13x13(img, kernel_addr):
    img_erosion = cv2.erode(img, kernel_addr, iterations=6)
    img_dilation = cv2.dilate(img_erosion, kernel_addr, iterations=6)

    return img_dilation

def open15x15(img, kernel_addr):
    img_erosion = cv2.erode(img, kernel_addr, iterations=7)
    img_dilation = cv2.dilate(img_erosion, kernel_addr, iterations=7)

    return img_dilation

def close(img, k):
    img_dilation = dilate(img, k)
    img_erosion = erode(img_dilation, k)

    return img_erosion

def close5x5(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=2)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=2)

    return img_erosion

def close7x7(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=3)

    return img_erosion

def close9x9(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=4)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=4)

    return img_erosion

def close11x11(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=5)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=5)

    return img_erosion

def close13x13(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=6)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=6)

    return img_erosion

def close15x15(img, kernel_addr):
    img_dilation = cv2.dilate(img, kernel_addr, iterations=7)
    img_erosion = cv2.erode(img_dilation, kernel_addr, iterations=7)

    return img_erosion

def convertTo256(img):
    height, width, channels = img.shape

    temp = np.zeros((height, width))
    out = np.float64(temp)

    for y in range(height):
        for x in range(width):
            out[y, x] = ((img[y, x, 2] * 1) + (img[y, x, 1] * 256) + (img[y, x, 0] * pow(256, 2))) / 100

    return out

def convertFrom256(img):
    height, width = img.shape

    out = np.uint8(np.zeros((height, width, 3)))
    # dog = np.uint8(img)

    for y in range(height):
        for x in range(width):
            asd = szamrendszerShit(img[y, x])
            # print(asd)
            out[y, x, 2] = asd[2]
            out[y, x, 1] = asd[1]
            out[y, x, 0] = asd[0] - 1

    out = np.uint8(out)
    return out

def szamrendszerShit(asd):

    asd = asd * 100
    negyzet = asd // pow(256, 2)
    asd = asd % pow(256, 2)
    elso = asd // 256
    asd = asd % 256
    nulladik = asd // 1

    return [negyzet, elso, nulladik]

def RGB2HSI(rgb_img):
    """
    This is the function to convert RGB color image to HSI image
    :param rgm_img: RGB color image
    :return: HSI image
    """

    # Save the number of rows and columns of the original image
    row = np.shape(rgb_img)[ 0 ]
    col = np.shape(rgb_img)[1]
    # Copy the original image
    hsi_img = rgb_img.copy()
    # Channel splitting the image
    B, G, R = cv2.split(rgb_img)
    # Normalize the channel to [0,1]
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]
    H = np.zeros((row, col))  # define     H channel
    I = (R + G + B) / 3.0
    # Calculate I channel
    S = np.zeros((row,col)) #define S channel
    for i in range(row):
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
        # Calculate the included angle
        h = np.zeros(col)                #define temporary array# den > 0 and G >= Belementh is assignedtothetha
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        # den>0 and the element h of G<=B is assigned to thetha
        h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
        # den<0 The element h is assigned to 0
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)  # Assignment to the H channel afterradianization  # Calculate the S channel
    for i in range(row):
        min = []
        # Find the minimum value of each group of RGB values
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        # Calculate the S channel
        S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
        # I is a value of 0 directly assigned to 0
        S[i][R[i] + B[i] + G[i] == 0] = 0
        # Expand to 255 for easy display. Generally, the H component is between [0,2pi], and S and I are between [0,1]
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


def HSI2RGB(hsi_img):
    """
    This is the function to convert HSI image to RGB image
    :param hsi_img: HSI color image
    :return: RGB image
    """
    # The number of rows and columns to save the original image
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    # Copy the original image
    rgb_img = hsi_img.copy()
    # Channel splitting the image
    H, S, I = cv2.split(hsi_img)
    # Normalize the channel to [0,1]
    [H, S, I] = [i / 255.0 for i in ([H, S, I])]
    R, G, B = H, S, I
    for i in range(row):
        h = H[i] * 2 * np.pi
        # H is greater than or equal to 0 and less than 120 degrees when
        a1 = h >= 0
        a2 = h < 2 * np.pi / 3
        a = a1 & a2  # The flower of the first case Style index
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i] * (1 + S[i] * np.cos(h) / tmp)
        g = 3 * I[i] - r - b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        # H is greater than or equal to 120 degrees and less than 240 degrees
        a1 = h >= 2 * np.pi / 3
        a2 = h < 4 * np.pi / 3
        a = a1 & a2
        # index
        # of
        # the
        # second
        # case
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i] * (1 + S[i] * np.cos(h - 2 * np.pi / 3) / tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        # H is greater than or equal to 240 degrees and less than 360 degrees
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2
        # The fancy index of the third case tmp = np.cos( 5 * np.pi/3 -h)
        g = I[i] * (1 - S[i])
        b = I[i] * (1 + S[i] * np.cos(h - 4 * np.pi / 3) / tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:, :, 0] = B * 255
    rgb_img[:, :, 1] = G * 255
    rgb_img[:, :, 2] = R * 255
    return rgb_img