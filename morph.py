import cv2
import numpy as np
from metrics import get_CEF, get_AVG_Contrast


def multiscaleMorphology(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    height, width, channels = img.shape
    print("Multiscale Morphology, image shape: " + str(image.shape))

    # img = np.float32(img)
    # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # openings
    op1 = open(img, 3)
    op2 = open(img, 5)
    op3 = open(img, 7)
    op4 = open(img, 9)
    op5 = open(img, 11)
    op6 = open(img, 13)
    op7 = open(img, 15)

    # closings
    clos1 = close(img, 3)
    clos2 = close(img, 5)
    clos3 = close(img, 7)
    clos4 = close(img, 9)
    clos5 = close(img, 11)
    clos6 = close(img, 13)
    clos7 = close(img, 15)

    # White-Top-Hat
    wth1 = cv2.subtract(img, op1)
    wth2 = cv2.subtract(img, op2)
    wth3 = cv2.subtract(img, op3)
    wth4 = cv2.subtract(img, op4)
    wth5 = cv2.subtract(img, op5)
    wth6 = cv2.subtract(img, op6)
    wth7 = cv2.subtract(img, op7)

    # Black-Top-Hat
    bth1 = cv2.subtract(clos1, img)
    bth2 = cv2.subtract(clos2, img)
    bth3 = cv2.subtract(clos3, img)
    bth4 = cv2.subtract(clos4, img)
    bth5 = cv2.subtract(clos5, img)
    bth6 = cv2.subtract(clos6, img)
    bth7 = cv2.subtract(clos7, img)

    # wth-k különbségei - WTHV képek
    wthv1 = cv2.subtract(wth2, wth1)
    wthv2 = cv2.subtract(wth3, wthv1)
    wthv3 = cv2.subtract(wth4, wthv2)
    wthv4 = cv2.subtract(wth5, wthv3)
    wthv5 = cv2.subtract(wth6, wthv4)
    wthv6 = cv2.subtract(wth7, wthv5)

    # bth-k különbségei - BTHV képek
    bthv1 = cv2.subtract(bth2, bth1)
    bthv2 = cv2.subtract(bth3, bthv1)
    bthv3 = cv2.subtract(bth4, bthv2)
    bthv4 = cv2.subtract(bth5, bthv3)
    bthv5 = cv2.subtract(bth6, bthv4)
    bthv6 = cv2.subtract(bth7, bthv5)

    # WTH_MAX megkapása
    wthmax1 = vectorMax(wth1, wth2)
    wthmax2 = vectorMax(wth3, wth4)
    wthmax3 = vectorMax(bth6, bth7)
    wthmax4 = vectorMax(wth5, wthmax3)
    wthmax5 = vectorMax(wthmax1, wthmax2)
    wthmaxLast = vectorMax(wthmax5, wthmax4)

    # BTH_MAX megkapása
    bthmax1 = vectorMax(bth1, bth2)
    bthmax2 = vectorMax(bth3, bth4)
    bthmax3 = vectorMax(bth6, bth7)
    bthmax4 = vectorMax(bth5, bthmax3)
    bthmax5 = vectorMax(bthmax1, bthmax2)
    bthmaxLast = vectorMax(bthmax5, bthmax4)

    # WTHV_MAX megkapása
    wthvmax1 = vectorMax(wthv1, wthv2)
    wthvmax2 = vectorMax(wthv3, wthv4)
    wthvmax3 = vectorMax(wthv5, wthv6)
    wthvmaxLast1 = vectorMax(wthvmax1, wthvmax2)
    wthvmaxLast = vectorMax(wthvmaxLast1, wthvmax3)

    # BTHV_MAX megkapása
    bthvmax1 = vectorMax(bthv1, bthv2)
    bthvmax2 = vectorMax(bthv3, bthv4)
    bthvmax3 = vectorMax(bthv5, bthv6)
    bthvmaxLast1 = vectorMax(bthvmax1, bthvmax2)
    bthvmaxLast = vectorMax(bthvmaxLast1, bthvmax3)

    # a végső kivonás és összeadások
    elso = cv2.add(wthmaxLast, wthvmaxLast)
    masodik = cv2.add(bthmaxLast, bthvmaxLast)
    harmadik = cv2.subtract(elso, masodik)
    out = cv2.add(img, harmadik)

    # out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # out = np.uint8(out)

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
    # nem használom mert felesleges vele szorozni mivel ugyanugy az 1-eseket tartalmazo szerkesztoelemmel dolgozok
    # SE = np.ones((k, k), dtype=np.uint8)

    constant = (k - 1) // 2

    dilated = np.zeros_like(img, dtype=np.uint8)
    # dilated = np.zeros_like(img, dtype=np.float32)

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
                    elif product[x,y,1] > product[constant,constant,1]:
                        max_pixel = product[x, y]
                    elif product[x, y, 0] > product[constant, constant, 0]:
                        max_pixel = product[x, y]
                    else:
                        max_pixel = product[constant, constant]
            dilated[i, j] = max_pixel

    return dilated


def erode(img, k):

    m, n, channels = img.shape

    # Define the structuring element
    # k= 11,15,45 -Different sizes of the structuring element
    # nem használom mert felesleges vele szorozni mivel ugyanugy az 1-eseket tartalmazo szerkesztoelemmel dolgozok
    # SE = np.ones((k, k), dtype=np.uint8)

    constant = (k - 1) // 2

    imgErode = np.zeros_like(img, dtype=np.uint8)
    # imgErode = np.zeros_like(img, dtype=np.float32)

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
                    elif product[x,y,1] < product[constant,constant,1]:
                        min_pixel = product[x, y]
                    elif product[x, y, 0] < product[constant, constant, 0]:
                        min_pixel = product[x, y]
                    else:
                        min_pixel = product[1, 1]
            imgErode[i, j] = min_pixel

    return imgErode


def vectorMax(img1, img2):
    m, n, channels = img1.shape

    for x in range(m):
        for y in range(n):
            if img1[x, y, 2] < img2[x, y, 2]:
                img1[x, y] = img2[x, y]
            elif img1[x, y, 1] < img2[x, y, 1]:
                img1[x, y] = img2[x, y]
            elif img1[x, y, 0] < img2[x, y, 0]:
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
