import os
import cv2

from PyQt5.QtWidgets import *


def getFileName():
    file_filter = 'Image File (*.png *.jpg)'
    response = QFileDialog.getOpenFileName(
        caption='Select an image file',
        directory=os.getcwd(),
        filter=file_filter,
    )
    print(response[0])
    return response[0]

