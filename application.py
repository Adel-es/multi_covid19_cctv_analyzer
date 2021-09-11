import sys
import threading
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.uic import loadUi
from PyQt5 import QtGui
# from qtimeline import QTimeLine
import json
import cv2 # for test

from App import appInfo
from App.contactorListUI import *
from App.confirmedListUI import *

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
        # RootOfConfirmedCaseWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)
        #  결과 띄우기
        widget.currentWidget().showResult()

    
    def showContactorLstBtnClicked(self):
        # RootOfConfirmedCaseWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+2)


class RootOfConfirmedCaseWindow(QDialog):
    def __init__(self, videoResultList):
        super().__init__()
        loadUi("./UI/rootOfConfirmedCase.ui", self)
        # self.videoName = videoResult.videoName
        # self.fps = videoResult.fps
        # self.targetInfoList = videoResult.targetInfoList
        self.videoResultList = videoResultList
        self.backBtn.clicked.connect(self.backBtnClicked)

    def showResult(self):
        
        table_row_cnt = 0;
        # table_row_cnt = 각 video result에 있는 targetInfoList의 개수
        for videoResult in videoResultList:
            table_row_cnt += len(videoResult.targetInfoList)

        # tableWidget setting
        self.tableWidget.setRowCount(table_row_cnt)
        self.tableWidget.setColumnCount(4)        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        cur_table_row = 0
        for videoResult in videoResultList:
            print('in showResult')
            targetInfoList = videoResult.targetInfoList
            videoName = videoResult.videoName
            fps = videoResult.fps

            result = [ (videoName, str(info['index']), 
                        getTimeFromFrame(info['in'], fps), 
                        getTimeFromFrame(info['out'], fps)) 
                        for info in targetInfoList ]

            # (위쪽 list에 항목 추가)
            for row in range(len(result)):
                for col in range(4):
                    self.tableWidget.setItem(row + cur_table_row, col, QTableWidgetItem(result[row][col]))
            cur_table_row += len(result)

            # Timeline widget 추가
            timelineWidget = TimeLineWidget(videoResult)
            self.insertWidgetInListWidget( timelineWidget, self.listWidget )

            # (아래쪽 list) 영상 이름 추가
            videoNameWidget = QLabel( videoName )
            videoNameWidget.setAlignment(Qt.AlignCenter)
            videoNameWidget.setFixedHeight( timelineWidget.height()+4 )
            self.insertWidgetInListWidget( videoNameWidget, self.listWidget_2 )


    def backBtnClicked(self):
        # 결과 화면 목록창으로 전환
        widget.setCurrentIndex(widget.currentIndex()-1)
    
    def insertWidgetInListWidget(self, widget, listWidget):
        # QListWidget에 QWidget 객체를 삽입하는 함수
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
    def __init__(self, videoResultList):
        super().__init__()
        loadUi("./UI/contactorList.ui", self)
        # self.contactorInfoList = videoResult.contactorInfoList
        # self.videoName = videoResult.videoName
        self.videoResultList = videoResultList
        # self.fps = videoResult.fps

        self.showContactor()
        self.backBtn.clicked.connect(self.backBtnClicked)
        
    def showContactor(self):
        for videoResult in self.videoResultList:
            contactorInfoList = videoResult.contactorInfoList
            videoName = videoResult.videoName
            fps = videoResult.fps
            for contactorInfo in contactorInfoList:
                if os.path.exists(contactorInfo['image_path']):
                    # custom widget를 listWidgetItem으로 추상화하는 용도.
                    custom_widget = ContactorItem(contactorInfo, videoName, fps)
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
        

class VideoResult:
    def __init__(self, videoName, targetInfoList, contactorInfoList, frameNo, fps):
        self.videoName = videoName
        self.targetInfoList = targetInfoList
        self.contactorInfoList = contactorInfoList
        self.frameNo = frameNo
        self.fps = fps

def loadJson():
    '''
        system 출력 결과 json파일 로드
    '''
    videoResultList = []

    # 지정된 dir에서 result json 파일들을 찾는다.
    result_json_dir = appInfo.result_json_dir
    result_json_path = [ "{}/{}".format(result_json_dir, _) for _ in os.listdir(result_json_dir) if _.endswith(".json")]

    for json_path in result_json_path:
        # json 파일 로드
        with open(json_path) as json_file:
            result_json = json.load(json_file)
        targetInfoList, contactorInfoList = result_json['target'], result_json['contactor']

        for idx, info in enumerate(targetInfoList):
            info['index'] = idx

        # danger level 순으로 sorting하기
        contactorInfoList = sorted( contactorInfoList, key=lambda info : info['danger_level'], reverse=True)
        for info in contactorInfoList:
            info['image_path'] = appInfo.contactor_dir + "/fr{}_tid{}.png".format(info['capture_time'], info['tid'])

        # video의 frame no, fps 구하기    
        video_name = result_json['video_name']
        video_capture = cv2.VideoCapture( "{}/{}".format(appInfo.output_video_dir, result_json['video_name']))
        video_frameno = video_capture.get( cv2.CAP_PROP_FRAME_COUNT )
        video_fps = video_capture.get( cv2.CAP_PROP_FPS )

        videoResultList.append(VideoResult(video_name, targetInfoList, contactorInfoList, video_frameno, video_fps))
    return videoResultList

def center(self):
    qr = self.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

def getTimeFromFrame(frame, fps):
    sec = frame/int(fps)

    s = int(sec % 60)
    sec /= 60
    m = int(sec % 60)
    h = int(sec / 60)

    # return {'hour': h, 'minute': m, 'second': s}
    return "{}:{}:{}".format(h,m,s)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    #화면 전환용 Widget 설정
    widget = QStackedWidget()
    
    # system 출력 결과 json파일 로드
    videoResultList = loadJson()

    #레이아웃 인스턴스 생성
    firstWindow = FirstWindow()
    dataInputWindow = DataInputWindow()
    analysisWindow = AnalysisWindow()
    resultListWindow = ResultListWindow()
    rootOfConfirmedCaseWindow = RootOfConfirmedCaseWindow(videoResultList)
    contactorListWindow = ContactorListWindow(videoResultList)

    #Widget 추가
    widget.addWidget(firstWindow)
    widget.addWidget(dataInputWindow)
    widget.addWidget(analysisWindow)
    widget.addWidget(resultListWindow)
    widget.addWidget(rootOfConfirmedCaseWindow)
    widget.addWidget(contactorListWindow)

    #프로그램 화면을 보여주는 코드
    widget.setWindowTitle('CCTV 영상 분석을 통한 코로나 확진자 동선 및 접촉자 판별 시스템')
    widget.resize(1000, 700)
    # 화면을 중앙에 위치시킴
    center(widget)
    widget.show()

    app.aboutToQuit.connect(analysisWindow.onExit)
    
    sys.exit(app.exec_())