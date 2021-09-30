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
    fsrc = response[0]
    if len(fsrc) == 0:
        return ""
    else:
        return fsrc

