import os, sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PIL import Image, ImageQt
from .utils import *
        
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
    def __init__(self, targetInfoListOfEachVideo, video_start_clock, video_end_clock, interval_start_time, interval_mid_time, interval_end_time, interval_total_time):
        super().__init__()
        layout_for_total_video = QHBoxLayout()
        layout = QHBoxLayout()

        
        print('\033[41m min start clock: \n\tis{}\n\tim{}\n\tie{} \033[0m'.format(interval_start_time, interval_mid_time, interval_end_time) )
        
        # self.targetInfoListOfEachVideo = targetInfoListOfEachVideo
        # self.frameNo = videoResult.frameNo
        # self.fps = videoResult.fps
        # self.targetInfoList = videoResult.targetInfoList
        # frameNo = self.targetInfoListOfEachVideo[0]['frame_no']
        
        self.first_in_time = targetInfoListOfEachVideo[0]['in'] + interval_start_time       
        start_frame = targetInfoListOfEachVideo[0]['frame_start']
        end_frame   = targetInfoListOfEachVideo[0]['frame_end']
        fps         = targetInfoListOfEachVideo[0]['fps']
        frameNo     = end_frame - start_frame + 1
        if frameNo < 0 :
            print(" [Error] TimeLineWidget: frame_count < 0 of '{}'".format(targetInfoListOfEachVideo[0]['video_name']))

        prev_end_time_dict = get_video_end_clock(video_start_clock, start_frame, fps)
        prev_end_time = prev_end_time_dict['m'] * 60 + prev_end_time_dict['s']
        print("\tstart time: {}".format(prev_end_time))
        for info in targetInfoListOfEachVideo:
            # sec 단위
            in_time_dict = get_video_end_clock(video_start_clock, info['in'], fps) 
            out_time_dict = get_video_end_clock(video_start_clock, info['out'], fps)
            in_time = in_time_dict['m'] * 60 + in_time_dict['s']
            out_time = out_time_dict['m'] * 60 + out_time_dict['s']
            print("\tin_time: {}, out_time: {}".format(in_time, out_time))
            lineRelWidth = in_time - prev_end_time
            labelRelWidth = out_time - in_time
            prev_end_time = out_time
            
            print(" ** line: {}, label: {}".format(lineRelWidth, labelRelWidth))
            if lineRelWidth != 0:
                # 시간축 가로선 추가
                line = HorizontalLine()
                layout.addWidget(line, stretch=lineRelWidth)

            if labelRelWidth != 0 or in_time > prev_end_time:
                # 컬러 시간 블록 추가
                label = QLabel()
                label.setFixedHeight(20)
                label.setStyleSheet("background-color: orange;")
                layout.addWidget(label, stretch=labelRelWidth)


        line = HorizontalLine()
        end_time_dict = get_video_end_clock(video_start_clock, frameNo, fps)
        end_time = end_time_dict['m'] * 60 + end_time_dict['s']
        print(" ** end: {}".format(end_time - prev_end_time))
        layout.addWidget(line, stretch = end_time - prev_end_time)
        layout.setSpacing(0) # widget 간 거리를 0으로 만듦 -> hline과 colorbar간의 거리를 없앰
        # layout.setFixedWidth(int(widthPerFrame * interval_mid_time))
        
        blank1 = QLabel()
        blank2 = QLabel()
        layout_for_total_video.addWidget(blank1, stretch=interval_start_time)
        layout_for_total_video.addLayout(layout, stretch=interval_mid_time)
        layout_for_total_video.addWidget(blank2, stretch=interval_end_time)
        layout_for_total_video.setSpacing(0)
        self.setLayout(layout_for_total_video)
    
    def getFirstInStartTime(self):
        return self.first_in_time
    
        