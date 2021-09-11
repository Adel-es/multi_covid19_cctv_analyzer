import os, sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PIL import Image, ImageQt
        
def getTimeFromFrame(frame, fps):
    sec = frame/int(fps)

    s = int(sec % 60)
    sec /= 60
    m = int(sec % 60)
    h = int(sec / 60)

    # return {'hour': h, 'minute': m, 'second': s}
    return "{}:{}:{}".format(h,m,s)

class HorizontalLine(QWidget):
    '''
        use in TimeLineWidget class (in RootOfConfirmedCaseWindow)
    '''
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet( "background-color: gray;")
        layout.addWidget(line)

        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

class ColorBar(QWidget):
    '''
        use in TimeLineWidget class (in RootOfConfirmedCaseWindow)
    '''
    def __init__(self, in_time, out_time):
        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setFixedHeight(20)
        self.label.setStyleSheet("background-color: orange;")
        layout.addWidget(label)
        self.setLayout(layout)

        '''
            color bar를 클릭하면 start~end 시간이 나타나도록 하려고 했는데
            동작하지 않는다..
            꼭 필요하다면 나중에 구현해보겠다.
        '''
        self.showPositionWindow = QDialog()
        spLabel = QLabel( "{} ~ {}".format(in_time, out_time), self.showPositionWindow)

        self.label.setMouseTracking(True)
        
    def mousePressEvent(self, e):
        print("Press!")
        self.showPositionWindow.setGeometry(e.x(), e.y(), 100, 30)
        self.showPositionWindow.show()
    
    def mouseReleaseEvent(self, e):
        print("Release!")
        self.showPositionWindow.close()


class TimeLineWidget(QWidget):
    '''
        use in RootOfConfirmedCaseWindow class
    '''
    def __init__(self, videoResult):
        super().__init__()
        layout = QHBoxLayout()

        self.frameNo = videoResult.frameNo
        self.fps = videoResult.fps
        self.targetInfoList = videoResult.targetInfoList

        prev_end_time = 0
        for info in self.targetInfoList:
            in_time = info['in']
            out_time = info['out']
            
            lineRelWidth = in_time - prev_end_time + 1
            labelRelWidth = out_time - in_time + 1
            prev_end_time = out_time
            print('***',lineRelWidth, ' ', labelRelWidth)

            line = HorizontalLine()
            label = QLabel()
            label.setFixedHeight(20)
            label.setStyleSheet("background-color: orange;")

            layout.addWidget(line, stretch = lineRelWidth)
            layout.addWidget(label, stretch = labelRelWidth)

        line = HorizontalLine()
        layout.addWidget(line, stretch = self.frameNo-prev_end_time+1)
        layout.setSpacing(0) # widget 간 거리를 0으로 만듦 -> hline과 colorbar간의 거리를 없앰
        self.setLayout(layout)
