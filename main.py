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
        loadUi("screen2.ui", self)
        self.mainWindowBtn.clicked.connect(self.gotoMainWindow)
        self.newpic.clicked.connect(self.imageThings)
        self.imageThings()


    def imageThings(self):
        imgsrc = getFileName()
        if len(imgsrc) == 0:
            print("Cancelled ", len(imgsrc))
        else:
            img = cv2.imread(imgsrc, cv2.IMREAD_GRAYSCALE)
            print(img.shape)
            pixmap = QPixmap(self.image_cv2qt(img))
            self.picLabel.setPixmap(pixmap)
            self.picLabel.resize(pixmap.width(), pixmap.height())
            #150 260
            self.magic = QPushButton(self)
            self.magic.setText("magic")
            self.magic.move(300, 300)



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
