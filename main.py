import os
import cv2
from PyQt5 import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
import sys

from fileOpener import *
from methods import *

from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("MainWindow.ui", self)
        self.screen2Btn.clicked.connect(self.gotoScreen2)

    def gotoScreen2(self):
        screen2 = Screen2()
        widget.addWidget(screen2)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class Screen2(QDialog):
    def __init__(self):
        super(Screen2, self).__init__()
        self._kep = None
        self._tempKep = None
        loadUi("screen2.ui", self)
        self.mainWindowBtn.clicked.connect(self.gotoMainWindow)
        self.mainWindowBtn.move(100, 10)
        self.newpic.clicked.connect(self.imageThings)
        self.newpic.setText("Másik kép")
        self.imageThings()

    @property
    def kep(self):  # get property
        return self._kep

    @kep.setter
    def kep(self, ertek):  # set property
        self._kep = ertek

    @property
    def tempKep(self):  # get property
        return self._tempKep

    @tempKep.setter
    def tempKep(self, ertek):  # set property
        self._tempKep = ertek

    def imageThings(self):
        img = getFileName()
        if len(img) == 0:
            print("Cancelled ", len(img))
        else:
            self.kep = cv2.imread(img, cv2.IMREAD_COLOR)
            self.tempKep = self.kep.copy()
            # print(self.kep.shape)
            pixmap = QPixmap(self.image_cv2qt(self.tempKep))
            self.picLabel.setPixmap(pixmap)
            self.picLabel.resize(pixmap.width(), pixmap.height())

            self.MSRCP = QPushButton(self)
            self.MSRCP.setText("MSRCP")
            self.MSRCP.move(300, 15)
            self.MSRCP.clicked.connect(self.msrcpCall)
            self.MSRCP.setStyleSheet("QPushButton"
                                     "{"
                                     "background-color : lightblue;"
                                     "}"
                                     "QPushButton::pressed"
                                     "{"
                                     "background-color : #4acfff;"
                                     "}"
                                     )
            self.MSRCR = QPushButton(self)
            self.MSRCR.setText("MSRCR")
            self.MSRCR.move(400, 15)
            self.MSRCR.clicked.connect(self.msrcrCall)
            self.MSRCR.setStyleSheet("QPushButton"
                                     "{"
                                     "background-color : lightblue;"
                                     "}"
                                     "QPushButton::pressed"
                                     "{"
                                     "background-color : #4acfff;"
                                     "}"
                                     )


    def msrcpCall(self):
        self.tempKep = retinexOnIntensity(self.kep)
        # print(self.kep.shape)
        # cv2.imshow('Inverz', self.kep)
        # cv2.waitKey(0)
        pixmap = QPixmap(self.image_cv2qt(self.tempKep))
        self.picLabel.setPixmap(pixmap)
        cv2.imwrite("output.png", self.tempKep)


    def msrcrCall(self):
        self.tempKep = retinexOnChannels(self.kep)
        pixmap = QPixmap(self.image_cv2qt(self.tempKep))
        self.picLabel.setPixmap(pixmap)
        cv2.imwrite("output.png", self.tempKep)


    def image_cv2qt(self, img):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = imgrgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(imgrgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        return qImg


    def gotoMainWindow(self):
        window = MainWindow()
        widget.addWidget(window)
        widget.setCurrentIndex(widget.currentIndex() + 1)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    widget = QStackedWidget()
    window = MainWindow()
    widget.addWidget(window)

    widget.setFixedHeight(600)
    widget.setFixedWidth(800)


    widget.show()

    sys.exit(app.exec_())
