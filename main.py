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
        imgsrc = getFileName()
        img = cv2.imread(imgsrc)
        print(img.shape)

    def gotoMainWindow(self):
        window = MainWindow()
        widget.addWidget(window)
        widget.setCurrentIndex(widget.currentIndex() + 1)


app = QApplication(sys.argv)
widget = QStackedWidget()
window = MainWindow()
widget.addWidget(window)

widget.setFixedHeight(600)
widget.setFixedWidth(800)


widget.show()

sys.exit(app.exec_())
