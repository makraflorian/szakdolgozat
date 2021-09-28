from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
import sys

def fileDialog():
    print('asd')

def main():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(400, 300, 500, 500)
    win.setWindowTitle("Kontrasztjavítás")

    fileDialogButton = QPushButton(win)
    fileDialogButton.setText("Choose image")
    fileDialogButton.move(50, 20)
    fileDialogButton.clicked.connect(fileDialog)

    win.show()
    sys.exit(app.exec_())


main()  # make sure to call the function))