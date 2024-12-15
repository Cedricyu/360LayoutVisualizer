import os
import sys
import cv2
import numpy as np
from imageio import imread
import json
import argparse
import shader

from PyQt5 import QtWidgets, QtGui, QtOpenGL
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore

import glm
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from Viewer import Utils
from Viewer import LayoutView

class TopWindow(QMainWindow):
    def __init__(self, img, layout, floor_reverse=False, parent=None):
        super().__init__(parent)
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        [self.h, self.w] = [sizeObject.height(), sizeObject.width()]
        ratio = 0.9
        self.h = int(self.h * ratio)
        self.w = int(self.w * ratio)
        self.setGeometry(20, 60, self.w, self.h)
        self.setWindowTitle("Layout Visualizer")
        self.centeralWidget = QWidget(self)

        self.layout = layout
        self.LayoutViewer = LayoutView.GLWindow(img, main=self, parent=self.centeralWidget)
        wallNum, wallPoints, lines, mesh = Utils.Label2Mesh(Utils.OldFormat2Mine(self.layout), floor_reverse)
        # print("Mesh data:", mesh)
        # mesh[:, 1] *= -1
        self.LayoutViewer.updateLayoutMesh(wallNum, wallPoints, lines, mesh)

        layout = QGridLayout()
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(self.LayoutViewer, 0, 0, 1, 1)
        self.centeralWidget.setLayout(layout)
        self.setCentralWidget(self.centeralWidget)

    def enterEvent(self, event):
        self.setFocus(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='360 Layout Visualizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument('--img', type=str, required=True, help='The panorama path')
    parser.add_argument('--json', type=str, required=True, help='The output json path')
    args = parser.parse_args()

    img = imread(args.img, pilmode='RGB')
    with open(args.json, 'r') as f: layout = json.load(f)

    app = QtWidgets.QApplication(sys.argv)
    window = TopWindow(img, layout=layout)
    window.show()
    sys.exit(app.exec_())
