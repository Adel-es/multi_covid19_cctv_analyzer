import os, sys

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot, QSize
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PIL import Image, ImageQt
import cv2

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
        # picture = Image.open(path)
        picture = cv2.imread(path)
        h, w, c = picture.shape
        origin_fixed_h = 500
        small_fixed_sz = 150
        # 원본 사진 & 썸네일 용 작은 사진
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        
        origin_w = int(w * (origin_fixed_h / h))
        origin_picture = cv2.resize(picture, dsize=(origin_w, origin_fixed_h), interpolation=cv2.INTER_LINEAR)
        origin_pixmap = QPixmap.fromImage(QtGui.QImage(origin_picture.data, origin_w, origin_fixed_h, origin_w*c, QtGui.QImage.Format_RGB888))
        self.origin_window = OriginPictureWindow(origin_pixmap)
        
        small_picture = picture[0:w, 0:w]
        small_picture = cv2.resize(small_picture, dsize=(small_fixed_sz, small_fixed_sz), interpolation=cv2.INTER_LINEAR)
        sh, sw, sc = small_picture.shape
        # small_picture = picture.crop( (0, 0, self.w, self.w) ).resize( (small_fixed_sz,small_fixed_sz))
        small_pixmap = QPixmap.fromImage(QtGui.QImage(small_picture.data, sw, sh, sw*sc, QtGui.QImage.Format_RGB888)) # QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(small_picture)))

        
        # 목록에는 썸네일 사진만 보여준다.
        self.picture_label = QLabel()
        self.picture_label.setPixmap( small_pixmap )
        self.picture_label.setFixedWidth(small_fixed_sz)
        self.picture_label.setFixedHeight(small_fixed_sz)

        self.layout.addWidget(self.picture_label)
        self.setLayout(self.layout)
        self.setFixedHeight(small_fixed_sz)

        # self.setMaximumWidth(self.w)

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
    def __init__(self, info, videoName, start_time_str, end_time_str, target_mask, contactor_mask):
        QWidget.__init__(self)
        loadUi("./UI/ui/contactorItem.ui", self)
        
        # 접촉자 사진 추가하기
        thumbnail = PictureWidget(info['image_path'])
        self.thumbnail_layout.addWidget(thumbnail)

        # 접촉자 정보 추가하기 (영상 이름, 나타난 시간, 위험도)
        self.video_name_label.setText(videoName)
        # self.date_label.setText(info.date)
        self.time_label.setText("{} ~ {}".format(
            start_time_str, 
            end_time_str
            ))
        if type(info['danger_level']) != str:
            info['danger_level'] = str(info['danger_level'])
        self.dangerous_score_label.setText("{}  (확진자: 마스크 {}, 접촉자: 마스크 {})".format(
                                        info["danger_level"], 
                                        self.getMaskStatusStatement(info['target_mask']),
                                        self.getMaskStatusStatement(info['contactor_mask'])))

    def getMaskStatusStatement(self, mask):
        if mask == "masked" : 
            return "O"
        elif mask == "notMasked" : 
            return "X"
        elif mask == "faceNotFound" : 
            return "?"
        elif mask == "UnKnown" : 
            return "?"