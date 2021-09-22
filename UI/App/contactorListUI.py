import os, sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PIL import Image, ImageQt

from .utils import *

class OriginPictureWindow(QDialog):
    '''
        접촉자의 원본(전신) 이미지를 보여주는 Window class

        Args:
            pixmap : 접촉자의 원본 이미지를 담고 있는 Pixmap()
    '''
    def __init__(self, pixmap):
        super().__init__()
        # this will hide the title bar
        self.setWindowFlag(Qt.FramelessWindowHint)
  
        # set the title
        self.setWindowTitle("no title")

        # get contactor's picture
        self.layout = QVBoxLayout()
        self.pix_label = QLabel()
        self.pix_label.setPixmap(pixmap)
        self.layout.addWidget(self.pix_label)
        self.setLayout(self.layout)

class PictureWidget(QWidget):
    '''
        접촉자의 썸네일 이미지와 원본 이미지 창을 띄워주는 역할을 하는 class

        Args:
            path(임시): 접촉자 이미지의 경로 
                        (-> 나중에 bbox와 frame을 받아서 crop하는 로직으로 수정)
    '''
    def __init__(self, path):
        super().__init__()
        self.layout = QVBoxLayout()

        # 이미지 로드
        picture = Image.open(path)
        self.w, self.h = picture.width, picture.height
        
        # 원본 사진 & 썸네일 용 작은 사진
        self.origin_pixmap = QPixmap.fromImage(ImageQt.ImageQt(picture))
        self.origin_window = OriginPictureWindow(self.origin_pixmap)
        self.small_pixmap = QPixmap.fromImage(ImageQt.ImageQt(picture.crop( (0, 0, self.w, self.w) )))

        # 목록에는 썸네일 사진만 보여준다.
        self.picture_label = QLabel()
        self.picture_label.setPixmap( self.small_pixmap )
        self.picture_label.setMaximumWidth(self.w)

        self.layout.addWidget(self.picture_label)
        self.setLayout(self.layout)
        self.setMaximumWidth(self.w)

        # 썸네일 사진을 마우스 왼쪽 키로 누르면 원본 사진 창이 뜨고
        # 마우스 왼쪽 키를 놓으면 창이 꺼진다.
        self.picture_label.setMouseTracking(True)

    def mousePressEvent(self, e):
        self.origin_window.show()
    
    def mouseReleaseEvent(self, e):
        self.origin_window.close()


class ContactorItem(QWidget):
    '''
        접촉자의 사진과 정보를 구성하는 Widget.
        접촉자의 사진은 PictureWidget이라는 Widget을 통해 추가한다.

        Args:
            info: 접촉자의 정보(사진, 영상 이름..)를 담고있는 ContactorInfo()
    '''
    def __init__(self, info, videoName, fps):
        QWidget.__init__(self)
        loadUi("./UI/contactorItem.ui", self)
        
        # 접촉자 사진 추가하기
        thumbnail = PictureWidget(info['image_path'])
        self.thumbnail_layout.addWidget(thumbnail)

        # 접촉자 정보 추가하기 (영상 이름, 나타난 시간, 위험도)
        self.video_name_label.setText(videoName)
        # self.date_label.setText(info.date)
        self.time_label.setText("{} ~ {}".format(
            getTimeFromFrame(info["start_time"], fps), 
            getTimeFromFrame(info["end_time"], fps)
            ))
        if type(info['danger_level']) != str:
            info['danger_level'] = str(info['danger_level'])
        self.dangerous_score_label.setText(info["danger_level"])
