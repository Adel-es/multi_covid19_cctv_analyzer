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

if appInfo.only_app_test == False:
    import shutil
    if appInfo.sync_analysis_system == True:
        from multiprocessing import Process, Queue
        import run

class ErrorAlertMessage(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Error")
        self.setIcon(QMessageBox.Critical)
        self.setStandardButtons(QMessageBox.Ok)

    def setCustomText(self, message):
        self.setText(message)
        
class FirstWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/first.ui", self)
        self.proj_parent_dir_path = appInfo.repo_path # default로 현재 속한 repo의 경로를 설정
        self.proj_dir_path = ""
        self.projParentDirLabel.setText( self.proj_parent_dir_path )
        self.findPathBtn.clicked.connect(self.findPathBtnClicked)
        if appInfo.only_app_test == False:
            self.errorMessage = ErrorAlertMessage() # 유효성 검사 에서 사용
            self.createBtn.clicked.connect(self.createBtnClicked)
        else:
            self.createBtn.clicked.connect(self.createBtnClickedNoValid)

    @pyqtSlot()
    def createBtnClickedNoValid(self):
        # DataInputWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)

    @pyqtSlot()
    def createBtnClicked(self):
        '''파일 및 디렉토리 유효성 검사'''
        if (not isinstance( self.proj_parent_dir_path, str )) or \
                    (len(self.proj_parent_dir_path) == 0):
            self.errorMessage.setCustomText("프로젝트 생성 경로를 선택하세요.")
            self.errorMessage.show()
        elif not os.path.exists( self.proj_parent_dir_path ):
            self.errorMessage.setText("{}는 존재하지 않는 디렉토리 입니다.".format(self.proj_parent_dir_path))
            self.errorMessage.show()
        elif self.projNameLineEdit.text() == "":
            self.errorMessage.setCustomText("프로젝트 이름을 입력하세요.")
            self.errorMessage.show()
        else:
            self.proj_dir_path = "{}/{}".format(self.proj_parent_dir_path
                                                , self.projNameLineEdit.text())
            
            # 매번 새로운 프로젝트를 생성하기 귀찮으므로 잠시 생략함.
            # if os.path.exists(self.proj_dir_path) and \
            #     os.path.isdir(self.proj_dir_path):
            #     self.errorMessage.setCustomText("{}는 이미 존재하는 디렉토리 입니다.".format(self.proj_dir_path))
            #     self.errorMessage.show()
            # else:
            #     os.makedirs("{}/{}".format(self.proj_dir_path, "data"))
            #     os.makedirs("{}/{}".format(self.proj_dir_path, "data/input"))
            #     os.makedirs("{}/{}".format(self.proj_dir_path, "data/output"))
            #     os.makedirs("{}/{}".format(self.proj_dir_path, "data/input/query"))
            #     os.makedirs("{}/{}".format(self.proj_dir_path, "data/output/analysis"))
            #     # DataInputWindow로 전환
            #     widget.setCurrentIndex(widget.currentIndex()+1)
            #     # 프로젝트 디렉토리 이름dmf DataInputWindow에 전달
            #     widget.currentWidget().getProjectDirPath( self.proj_dir_path)
            
            ### (임시) 이미 존재하는 디렉토리인 경우, 그냥 그 프로젝트를 사용하기로 수정.
            if os.path.exists(self.proj_dir_path):
                if os.path.isdir(self.proj_dir_path):
                    # 프로젝트에 필요한 디렉토리가 없는 경우 새로 생성.
                    if not os.path.exists("{}/{}".format(self.proj_dir_path, "data")):
                        os.makedirs("{}/{}".format(self.proj_dir_path, "data"))
                    if not os.path.exists("{}/{}".format(self.proj_dir_path, "data/input")):
                        os.makedirs("{}/{}".format(self.proj_dir_path, "data/input"))
                    if not os.path.exists("{}/{}".format(self.proj_dir_path, "data/output")):
                        os.makedirs("{}/{}".format(self.proj_dir_path, "data/output"))
                    if not os.path.exists("{}/{}".format(self.proj_dir_path, "data/input/query")):
                        os.makedirs("{}/{}".format(self.proj_dir_path, "data/input/query"))
                    if not os.path.exists("{}/{}".format(self.proj_dir_path, "data/output/analysis")):
                        os.makedirs("{}/{}".format(self.proj_dir_path, "data/output/analysis"))
                        
                    # 이미 존재하는 프로젝트를 사용할 것인지 물음
                    buttonReply = QMessageBox.question(self, 'Warning', u"이미 존재하는 프로젝트입니다. 계속하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if buttonReply == QMessageBox.Yes:
                        # DataInputWindow로 전환
                        widget.setCurrentIndex(widget.currentIndex()+1)
                        # 프로젝트 디렉토리 이름dmf DataInputWindow에 전달
                        widget.currentWidget().getProjectDirPath( self.proj_dir_path)
                        print('Yes clicked.')
                    else:
                        print('No clicked.')
                # 선택한 파일이 디렉토리가 아니면 프로젝트로 사용할 수 없다. --> 사실 이런 경우가 없긴 함. 하지만 넣어서 나쁠건 없으니
                else:
                    self.errorMessage.setCustomText("{}는 디렉토리가 아닙니다..".format(self.proj_dir_path))
                    self.errorMessage.show()
            ### 새로 생성한 디렉토리인 경우, 필요한 디렉토리들을 새로 생성.
            else:
                # 프로젝트를 새로 생성할 것인지 물음
                buttonReply = QMessageBox.question(self, 'Warning', u"프로젝트를 새로 생성합니다. 계속하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if buttonReply == QMessageBox.Yes:
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/input"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/output"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/input/query"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/output/analysis"))
                    # DataInputWindow로 전환
                    widget.setCurrentIndex(widget.currentIndex()+1)
                    # 프로젝트 디렉토리 이름dmf DataInputWindow에 전달
                    widget.currentWidget().getProjectDirPath( self.proj_dir_path)
                    print('Yes clicked.')
                else:
                    print('No clicked.')
    @pyqtSlot()
    def findPathBtnClicked(self):
        self.proj_parent_dir_path = QFileDialog.getExistingDirectory(self, 'Select a Directory')
        self.projParentDirLabel.setText(self.proj_parent_dir_path)
        print("click btn : {}".format(self.proj_parent_dir_path))
    
class DataInputWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/dataInput.ui", self)
        self.project_dir_path = ""
        self.query_dir_path = ""
        self.video_dir_path = ""
        self.video_paths = []
        self.photo_paths = []

        self.addPhotoBtn.clicked.connect(self.addPhotoBtnClicked)
        self.addVideoBtn.clicked.connect(self.addVideoBtnClicked)
        if appInfo.only_app_test == False:
            self.errorMessage = ErrorAlertMessage() # 유효성 검사 에서 사용
            self.startAnalysisBtn.clicked.connect(self.startAnalysisBtnClicked)
        else:
            self.startAnalysisBtn.clicked.connect(self.startAnalysisBtnClickedNoValid)

    def getProjectDirPath(self, project_dir_path):
        self.project_dir_path = project_dir_path
        self.query_dir_path = "{}/{}".format(self.project_dir_path, "data/input/query")
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/input")
        # 기존에 있던 프로젝트의 경우, 이미 옮겨놓은 파일들을 ListWidget에 띄워준다.
        self.photo_paths = [ self.query_dir_path+'/'+_ for _ in os.listdir(self.query_dir_path)]
        for ppath in self.photo_paths:
            filepath_label = QLabel( ppath )
            filepath_label.setFixedHeight(20)
            self.insertWidgetInListWidget( filepath_label, self.photoListWidget )
            
        # print('existing photos: ', self.photo_paths)
        self.video_paths = [ self.video_dir_path+'/'+_ for _ in os.listdir(self.video_dir_path) if not os.path.isdir( self.video_dir_path+'/'+_ )]
        for vpath in self.video_paths:
            filepath_label = QLabel( vpath )
            filepath_label.setFixedHeight(20)
            self.insertWidgetInListWidget( filepath_label, self.videoListWidget )
        # print('existing videos: ', self.video_paths)
        
        
    def addPhotoBtnClicked(self):
        '''사진 파일만 받도록'''
        image_format = 'All File(*);; PNG File(*.png *.PNG);; JPEG File(*.jpg *.jpeg *.jfif)'
        filepath = QFileDialog.getOpenFileName(self, 'Open Images', '', image_format)
        if filepath[0] != '':
            filepath_label = QLabel( filepath[0] )
            filepath_label.setFixedHeight(20)
            self.insertWidgetInListWidget( filepath_label, self.photoListWidget )
            self.photo_paths.append( filepath[0] )

            if appInfo.only_app_test == False:
                shutil.copy(filepath[0], self.query_dir_path)

    def addVideoBtnClicked(self):
        '''영상 파일만 받도록'''
        '''여러 파일 한 번에 선택할 수 있도록'''
        '''영상이 중복해서 추가되지 않도록'''
        '''영상 주소 받는 부분도 추가'''
        video_format = 'All File(*);; Video File(*.avi *.mp4);; H264 file(*.h264)'
        filepath = QFileDialog.getOpenFileName(self, 'Open Videos', '', video_format)
        if filepath[0] != '':
            filepath_label = QLabel( filepath[0] )
            filepath_label.setFixedHeight(20)
            self.insertWidgetInListWidget( filepath_label, self.videoListWidget )
            self.video_paths.append( filepath[0] )

            if appInfo.only_app_test == False:
                shutil.copy(filepath[0], self.video_dir_path)

    def startAnalysisBtnClicked(self):
        '''입력 파일에 대한 유효성 검사(비어있지 않은지 등)'''
        if len(self.photo_paths) == 0:
            self.errorMessage.setText("확진자 파일을 추가하세요.")
            self.errorMessage.show()
        elif len(self.video_paths) == 0:
            self.errorMessage.setText("분석할 영상 파일을 추가하세요.")
            self.errorMessage.show()
        else:
            # AnalysisWindow로 전환
            widget.setCurrentIndex(widget.currentIndex()+1)
            # 분석 thread 시작
            widget.currentWidget().getProjectDirPath(self.project_dir_path, self.photo_paths, self.video_paths)
            widget.currentWidget().start(self.video_paths)

    def startAnalysisBtnClickedNoValid(self):
        '''입력 파일에 대한 유효성 검사(비어있지 않은지 등)'''
        # AnalysisWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)
        # 분석 thread 시작
        widget.currentWidget().getProjectDirPath(self.project_dir_path, self.photo_paths, self.video_paths)
        widget.currentWidget().start(self.video_paths)

    def insertWidgetInListWidget(self, widget, listWidget):
        '''
            QListWidget에 QWidget 객체를 삽입하는 함수
        '''
        item = QListWidgetItem( listWidget )
        item.setSizeHint( widget.sizeHint() )
        listWidget.setItemWidget( item, widget )
        listWidget.addItem( item )

class AnalysisWindow(QDialog):

    def __init__(self):
        super().__init__()
        loadUi("./UI/analysis.ui", self)
        self.setting_path = "" # 분석 시스템 setting file; runInfo.py
        self.repo_path = ""
        self.project_dir_path = ""
        self.query_dir_path = ""
        self.input_video_dir_path = ""
        self.output_video_dir_path = ""
        self.result_dir_path = ""

        self.photo_paths = []
        self.video_paths = []

        self.running = False
        self.labels = [self.videoLabel1, self.videoLabel2, self.videoLabel3, self.videoLabel4]
        self.timer = 0
        self.playTime = 3
        self.showRsltBtn.clicked.connect(self.showRsltBtnClicked)
        
    def getProjectDirPath(self, project_dir_path, photo_paths, video_paths):
        self.project_dir_path = project_dir_path
        self.query_dir_path = "{}/{}".format(self.project_dir_path, "data/input/query")
        self.input_video_dir_path = "{}/{}".format(self.project_dir_path, "data/input")
        self.output_video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")

        self.photo_paths = photo_paths
        self.video_paths = video_paths

        # project 디렉토리는 항상 시스템 레포 바로 하위에 있어야 함.
        self.repo_path = os.path.dirname(self.project_dir_path)
        print("repo path: {}".format(self.repo_path))
        # runInfo.py의 path
        self.setting_path = "{}/{}".format(self.repo_path, "configs/runInfo.py")
        print("runInfo path: {}".format(self.setting_path))
        
    def writeRunInfoFile(self):
        '''
            runInfo 설정값 변경하기
        '''
        setting_file = open(self.setting_path, "w", encoding="utf8")

        project_name        = self.project_dir_path.split('/')[-1]
        input_video_name    = self.video_paths[0].split('/')[-1]        # ****************************** 일단 video_paths를 하나만 받는 걸로
        output_video_name   = input_video_name.split('.')[0] + ".avi"   # input video name에서 확장자만 avi로 변경
        
        contents = getRunInfoFileContents(  input_video_path        = project_name + '/data/input/' + input_video_name, 
                                            query_image_path        = project_name + '/data/input/query/',
                                            output_json_path        = project_name + '/data/output/analysis/' + input_video_name.split('.')[0] +'.json', 
                                            output_video_path       = project_name + '/data/output/' + output_video_name, 
                                            output_contactors_path  = project_name + '/data/output/analysis/',
                                            )
        setting_file.write(contents)
        setting_file.close()

    def analysis(self, video_path, i):
        if appInfo.only_app_test == False:
            self.writeRunInfoFile()
            if appInfo.sync_analysis_system == True:
                # covid system 시작
                shm_queue = Queue()
                covid_system_process = Process(target=run.main, args=(shm_queue,))
                covid_system_process.start()
            else:
                cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)

        label = self.labels[i % len(self.labels)]
        group = i // len(self.labels)
        displaying = False
        # get label geometry
        qrect = label.geometry()
        width = qrect.width()
        height = qrect.height()

        if appInfo.only_app_test == False:
            if appInfo.sync_analysis_system == True:
                while self.running:
                    if shm_queue.empty():
                        continue
                    img = shm_queue.get()
                    ret = True
                    # ret, img = cap.read()
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
                label.setText("empty")
                covid_system_process.join() # covid system 종료까지 waiting
            else:
                # covid system 사용 X
                # video path에서 읽어진 비디오가 제대로 출력되는지 확인하는 용도
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
        else:
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
        widget.currentWidget().getProjectDirPath(self.project_dir_path)

class ResultListWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/resultList.ui", self)
        self.project_dir_path = ""
        self.video_dir_path = ""
        self.result_dir_path = ""
        self.targetInfoList = []
        self.contactorInfoList = []
        self.showRootRsltBtn.clicked.connect(self.showRouteRsltBtnClicked)
        self.showContactorLstBtn.clicked.connect(self.showContactorLstBtnClicked)
        self.showRootRsltBtnClicked_Once = False
        self.showContactorLstBtnClicked_Once = False
        
    def getProjectDirPath(self, project_dir_path):
        self.project_dir_path = project_dir_path
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")
        self.targetInfoList, self.contactorInfoList = loadJson(contactor_dir= self.result_dir_path, 
                                                               result_json_dir= self.result_dir_path)

    def showRouteRsltBtnClicked(self):
        # RouteOfConfirmedCaseWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+1)
        # setting하는 함수가 한번만 수행되도록 함
        if self.showRootRsltBtnClicked_Once == False:
            widget.currentWidget().getProjectDirPath(self.project_dir_path, self.targetInfoList)
            widget.currentWidget().showResult()
            self.showRootRsltBtnClicked_Once = True

    def showContactorLstBtnClicked(self):
        # ContactorListWindow로 전환
        widget.setCurrentIndex(widget.currentIndex()+2)
        # setting하는 함수가 한번만 수행되도록 함
        if self.showContactorLstBtnClicked_Once == False:
            widget.currentWidget().getProjectDirPath(self.project_dir_path, self.contactorInfoList)
            widget.currentWidget().showContactor()
            self.showContactorLstBtnClicked_Once = True

class RouteOfConfirmedCaseWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("./UI/routeOfConfirmedCase.ui", self)
        self.project_dir_path = ""
        self.video_dir_path = ""
        self.result_dir_path = ""
        self.targetInfoList = []
        # self.showResult()
        self.backBtn.clicked.connect(self.backBtnClicked)

    def getProjectDirPath(self, project_dir_path, targetInfoList):
        self.project_dir_path = project_dir_path
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")
        self.targetInfoList = targetInfoList

    def showResult(self):
        # targetListInfo를 1차원 list로 합치기
        print(self.targetInfoList)
        if len(self.targetInfoList) == 0:
            print("There is no target information -> np.hstack(self.targetInfoList) is error")
        targetInfoFlattenList = np.hstack(self.targetInfoList)

        # 위쪽 tableWidget setting
        self.tableWidget.setRowCount( len(targetInfoFlattenList) )
        self.tableWidget.setColumnCount(4)        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 위쪽 table -> 전체 video 결과에 대해 정렬해야함.
        for row, targetInfo in enumerate(targetInfoFlattenList):
            print(targetInfo)
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
    def __init__(self):
        super().__init__()
        loadUi("./UI/contactorList.ui", self)
        self.project_dir_path = ""
        self.video_dir_path = ""
        self.result_dir_path = ""
        self.contactorInfoList = []
        # self.showContactor()
        self.backBtn.clicked.connect(self.backBtnClicked)

    def getProjectDirPath(self, project_dir_path, contactorInfoList):
        self.project_dir_path = project_dir_path
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")
        self.contactorInfoList = contactorInfoList
        
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
    # targetInfoList, contactorInfoList = loadJson()

    #레이아웃 인스턴스 생성
    firstWindow = FirstWindow()
    dataInputWindow = DataInputWindow()
    analysisWindow = AnalysisWindow()
    resultListWindow = ResultListWindow()
    routeOfConfirmedCaseWindow = RouteOfConfirmedCaseWindow()
    contactorListWindow = ContactorListWindow()

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