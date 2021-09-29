import os
import cv2
from PyQt5 import *
from PyQt5.QtWidgets import *
import sys

from fileOpener import *

from PyQt5.uic import loadUi



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("MainWindow.ui", self)
        self.fileDialog.clicked.connect(getFileName)

        # img = cv2.imread(imgsrc)
        # print(img.shape)


# class Screen2(QDialog):
#     def __init__(self):
#         super(Screen2, self).__init__()
#         loadUi("MainWindow.ui", self)



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


main()