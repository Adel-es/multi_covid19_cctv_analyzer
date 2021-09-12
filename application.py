import sys
import threading
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.uic import loadUi
from PyQt5 import QtGui
# from qtimeline import QTimeLine
import cv2 # for test

from UI.App import appInfo
from UI.App.contactorListUI import *
from UI.App.confirmedListUI import *
from UI.App.utils import *
import numpy as np

class FirstWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/first.ui", self)
        self.createBtn.clicked.connect(self.createBtnClicked)

    @pyqtSlot()
    def createBtnClicked(self):
        if self.projNameLineEdit.text() != "":
            '''기존 프로젝트 이름과 겹치지 않는지 확인 필요'''
            # DataInputWindow로 전환
            widget.setCurrentIndex(widget.currentIndex()+1)


class DataInputWindow(QDialog):

    def __init__(self):
        super().__init__()
        loadUi("./UI/dataInput.ui", self)
        self.path = 'C:'
        self.video_paths = []
        # for test
        # self.video_paths = [
        #     'G:/내 드라이브/졸업과제/Deep-SORT-YOLOv4 출력 영상/output_yolov4_filtered.avi'
        # ]
        self.addPhotoBtn.clicked.connect(self.addPhotoBtnClicked)
        self.addVideoBtn.clicked.connect(self.addVideoBtnClicked)
        self.startAnalysisBtn.clicked.connect(self.startAnalysisBtnClicked)

    def addPhotoBtnClicked(self):
        '''사진 파일만 받도록'''
        self.filename = QFileDialog.getOpenFileName(self, self.path)
        if self.filename[0] != '':
            self.photoTextBrowser.setText(self.filename[0])
            photo_path = self.filename[0] # for test

    def addVideoBtnClicked(self):
        '''영상 파일만 받도록'''
        '''여러 파일 한 번에 선택할 수 있도록'''
        '''영상이 중복해서 추가되지 않도록'''
        '''영상 주소 받는 부분도 추가'''
        self.filename = QFileDialog.getOpenFileName(self, self.path)
        if self.filename[0] != '':
            self.videoTextBrowser.append(self.filename[0])
            self.video_paths.append(self.filename[0]) # for test

    def startAnalysisBtnClicked(self):
        '''입력 파일에 대한 유효성 검사(비어있지 않은지 등)'''
        # AnalysisWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)
        # 분석 thread 시작
        widget.currentWidget().start(self.video_paths)

class AnalysisWindow(QDialog):

    def __init__(self):
        super().__init__()
        loadUi("./UI/analysis.ui", self)
        self.running = False
        self.labels = [self.videoLabel1, self.videoLabel2, self.videoLabel3, self.videoLabel4]
        self.timer = 0
        self.playTime = 3
        self.showRsltBtn.clicked.connect(self.showRsltBtnClicked)

    def analysis(self, video_path, i):
        cap = cv2.VideoCapture(video_path)
        label = self.labels[i % len(self.labels)]
        group = i // len(self.labels)
        displaying = False
        # get label geometry
        qrect = label.geometry()
        width = qrect.width()
        height = qrect.height()

        while self.running:
            ret, img = cap.read()
            if ret:
                isMyTurnToDisplay = self.timer // self.playTime % self.displaySetNum == group
                if isMyTurnToDisplay:
                    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    h,w,c = img.shape
                    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)
                    label.setPixmap(pixmap)
                    if displaying == False:
                        displaying = True
                elif displaying == True:
                    label.setText("empty")
                    displaying = False
            else:
                break
        cap.release()
        label.setText("empty")
        print(f"({i}) Thread end.")

    def stop(self):
        if self.running != False:
            self.running = False
        print("stoped..")

    def start(self, video_paths):
        self.running = True
        self.displaySetNum = math.ceil(len(video_paths) / 4)
        self.startTimer()

        for i, path in enumerate(video_paths):
            th = threading.Thread(target=self.analysis, args=(path, i))
            th.start()
        print("started..")

    def onExit(self):
        print("exit")
        self.stop()

    def startTimer(self):
        self.timer += 1
        self.testLabel.setText(str(self.timer))
        timerThread = threading.Timer(1, self.startTimer)
        timerThread.start()
        if self.running == False:
            print("Timer stop")
            timerThread.cancel()

    def showRsltBtnClicked(self):
        '''분석 중단일 경우 정리할 것들 정리'''
        # 결과 화면 목록창으로 전환
        self.stop()
        widget.setCurrentIndex(widget.currentIndex()+1)

class ResultListWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/resultList.ui", self)
        self.showRootRsltBtn.clicked.connect(self.showRootRsltBtnClicked)
        self.showContactorLstBtn.clicked.connect(self.showContactorLstBtnClicked)

    def showRootRsltBtnClicked(self):
        # RouteOfConfirmedCaseWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)

    def showContactorLstBtnClicked(self):
        # ContactorListWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+2)


class RouteOfConfirmedCaseWindow(QDialog):
    def __init__(self, targetInfoList):
        super().__init__()
        loadUi("./UI/routeOfConfirmedCase.ui", self)
        # self.videoResultList = videoResultList
        self.targetInfoList = targetInfoList
        self.showResult()
        self.backBtn.clicked.connect(self.backBtnClicked)

    def showResult(self):
        # targetListInfo를 1차원 list로 합치기
        targetInfoFlattenList = np.array(targetInfoList)
        targetInfoFlattenList = targetInfoFlattenList.flatten()

        # 위쪽 tableWidget setting
        self.tableWidget.setRowCount( len(targetInfoFlattenList) )
        self.tableWidget.setColumnCount(4)        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 위쪽 table -> 전체 video 결과에 대해 정렬해야함.
        for row, targetInfo in enumerate(targetInfoFlattenList):
            result = [ targetInfo['video_name'],
                        str(targetInfo['index']),
                        getTimeFromFrame(targetInfo['in'], targetInfo['fps']), 
                        getTimeFromFrame(targetInfo['out'], targetInfo['fps'])]

            for col in range(4):
                self.tableWidget.setItem(row, col, 
                                        QTableWidgetItem(result[col]))

        # 아래쪽 list -> 각 video결과에 대해 timeline을 그려야 함.
        for targetInfoListOfEachVideo in self.targetInfoList:
            print('in showResult')

            # (아래쪽 list) Timeline widget 추가
            timelineWidget = TimeLineWidget(targetInfoListOfEachVideo)
            self.insertWidgetInListWidget( timelineWidget, self.listWidget )

            # (아래쪽 list) 영상 이름 추가
            videoNameWidget = QLabel( targetInfoListOfEachVideo[0]['video_name'] )
            videoNameWidget.setAlignment(Qt.AlignCenter)
            videoNameWidget.setFixedHeight( timelineWidget.height()+4 )
            self.insertWidgetInListWidget( videoNameWidget, self.listWidget_2 )

    def backBtnClicked(self):
        # 결과 화면 목록창으로 전환
        widget.setCurrentIndex(widget.currentIndex()-1)
    
    def insertWidgetInListWidget(self, widget, listWidget):
        '''
            QListWidget에 QWidget 객체를 삽입하는 함수
        '''
        item = QListWidgetItem( listWidget )
        item.setSizeHint( widget.sizeHint() )
        listWidget.setItemWidget( item, widget )
        listWidget.addItem( item )

class ContactorListWindow(QDialog):
    '''
        접촉자 목록을 띄워주는 Window

        Args:
            contactorInfoList: 접촉자들의 정보(사진, 영상 이름..)를 담고있는 ContactorInfo()의 리스트
    '''
    def __init__(self, contactorInfoList):
        super().__init__()
        loadUi("./UI/contactorList.ui", self)
        self.contactorInfoList = contactorInfoList
        self.showContactor()
        self.backBtn.clicked.connect(self.backBtnClicked)
        
    def showContactor(self):
        for contactorInfo in self.contactorInfoList:
            if os.path.exists(contactorInfo['image_path']):
                # custom widget를 listWidgetItem으로 추상화하는 용도.
                custom_widget = ContactorItem(contactorInfo, contactorInfo['video_name'], contactorInfo['fps'])
                item = QListWidgetItem(self.contactorList)

                # listWidgetItem은 custom widget의 크기를 모르므로 알려줘야 한다.
                item.setSizeHint(custom_widget.sizeHint())
                self.contactorList.setItemWidget(item, custom_widget)
                self.contactorList.addItem(item)
                
            else:
                print("Image is not exists: {}")

    def backBtnClicked(self):
        # 결과 화면 목록창으로 전환
        widget.setCurrentIndex(widget.currentIndex()-2)

def center(self):
    qr = self.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)

    #화면 전환용 Widget 설정
    widget = QStackedWidget()
    
    # system 출력 결과 json파일 로드
    targetInfoList, contactorInfoList = loadJson()

    #레이아웃 인스턴스 생성
    firstWindow = FirstWindow()
    dataInputWindow = DataInputWindow()
    analysisWindow = AnalysisWindow()
    resultListWindow = ResultListWindow()
    routeOfConfirmedCaseWindow = RouteOfConfirmedCaseWindow(targetInfoList)
    contactorListWindow = ContactorListWindow(contactorInfoList)

    #Widget 추가
    widget.addWidget(firstWindow)
    widget.addWidget(dataInputWindow)
    widget.addWidget(analysisWindow)
    widget.addWidget(resultListWindow)
    widget.addWidget(routeOfConfirmedCaseWindow)
    widget.addWidget(contactorListWindow)

    #프로그램 화면을 보여주는 코드
    widget.setWindowTitle('CCTV 영상 분석을 통한 코로나 확진자 동선 및 접촉자 판별 시스템')
    widget.resize(1000, 700)
    # 화면을 중앙에 위치시킴
    center(widget)
    widget.show()

    app.aboutToQuit.connect(analysisWindow.onExit)
    
    sys.exit(app.exec_())