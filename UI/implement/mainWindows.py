import os, sys
import threading
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, Qt, pyqtSlot, pyqtSignal, QEventLoop, QTimer
from PyQt5.uic import loadUi
from PyQt5 import QtGui
import cv2 # for test

from configs import appInfo
from .contactorListUI import *
from .confirmedListUI import *
from .utils import *
import numpy as np


if appInfo.only_app_test == False:
    import shutil
    if appInfo.sync_analysis_system == True:
        # from write_video
        from typing import List
        from distance import getCentroid
        from utils.types import MaskToken
        from configs import runInfo
        from utils.resultManager import Contactor, ResultManager

    def draw_bbox_and_tid(frame, person, isConfirmed):
        TEXT_UP_FROM_BBOX = 5
        Width = int((person.bbox.maxX - person.bbox.minX))
        Height = int( (person.bbox.maxY  - person.bbox.minY) )
        bboxLeftUpPoint = (int(person.bbox.minX ), int(person.bbox.minY))
        bboxRightDownPoint = (int(person.bbox.maxX), int(person.bbox.maxY))
        bboxHeadX = int((person.bbox.minX + person.bbox.maxX) / 2 )
        bboxHeadY = int(person.bbox.minY - Height / 10)
        
        triangleSize = Width / 5 * 2 
        pt1 = (bboxHeadX, bboxHeadY)
        pt2 = (int(bboxHeadX - 0.5 * triangleSize), int(bboxHeadY - 0.86 *triangleSize)) 
        pt3 =  (int(bboxHeadX + 0.5 * triangleSize), int(bboxHeadY - 0.86 *triangleSize)) 
        triangle_cnt = np.array( [pt1, pt2, pt3] )
        
        tidText = "ID: " + str(person.tid)
        tidPosition = (bboxLeftUpPoint[0], bboxLeftUpPoint[1]-TEXT_UP_FROM_BBOX)
        
        if isConfirmed:
            # bboxColor = (0, 0, 255) # red
            tidColor = (0, 0, 255) # red
            # cv2.rectangle(frame, bboxLeftUpPoint, bboxRightDownPoint, bboxColor, 3)
            cv2.drawContours(frame, [triangle_cnt], 0, tidColor, -1)
        else:
            tidColor = (0, 255, 0) # green

        thick = int(Width / 3)
        if thick <= 0 : 
            thick = 1
        elif thick > 3 : 
            thick = 3 
            
        scale = Width / 140
        if scale <= 0.7 : 
            scale = 0.7
        elif scale > 2 : 
            scale = 2
        
        bboxColor = (255, 255, 255) # white 
        cv2.rectangle(frame, bboxLeftUpPoint, bboxRightDownPoint, bboxColor, 3)
        cv2.putText(frame, tidText, tidPosition, 0, scale, tidColor, thick)

        def save_contactor_images(frame, shm, person_indices : List[int], save_list) :
            from configs import runInfo
            output_contactors_path = runInfo.output_contactors_path
            
            for index, pIdx in enumerate(person_indices) : 
                if save_list[index] == None : 
                    continue 
                file_name = output_contactors_path + save_list[index]
                person = shm.data.people[pIdx]
                minX = int(person.bbox.minX) 
                minY = int(person.bbox.minY)
                maxX = int(person.bbox.maxX)
                maxY = int(person.bbox.maxY)
                cropped_person = frame[minY : maxY, minX: maxX] 
                cv2.imwrite(file_name, cropped_person)
                

        def update_output_json(shm, res_manager, frame_number:int, frame_index : int, person_indices : List[int]) -> List : 
            from configs import runInfo
            end_frame = runInfo.end_frame
            
            reid_index = shm.data.frames[frame_index].reid 
            is_target : bool = ( reid_index != -1 )
            is_lastframe : bool = (frame_number == end_frame)
            res_manager.update_targetinfo(frame_number, is_target, is_lastframe)
            
            target = shm.data.people[reid_index]
            target_mask = target.isMask 
            
            save_images = [] 
            for pIdx in person_indices:
                person = shm.data.people[pIdx]
                if not person.isClose:
                    save_images.append(None)
                    continue
                tid = person.tid 
                contactor_mask = person.isMask 
                save, filename = res_manager.update_contactorinfo(frame_number, tid, target_mask, contactor_mask)
                if save == True : 
                    save_images.append(filename)
                else : 
                    save_images.append(None)
            return save_images 
    
class ErrorAlertMessage(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Error")
        self.setIcon(QMessageBox.Critical)
        self.setStandardButtons(QMessageBox.Ok)

    def setCustomText(self, message):
        self.setText(message)
        
class FirstWindow(QDialog):
    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/first.ui", self)
        self.stackedWidget = widget
        self.proj_parent_dir_path = appInfo.repo_path # default??? ?????? ?????? repo??? ????????? ??????
        self.proj_dir_path = ""
        self.projParentDirLabel.setText( self.proj_parent_dir_path )
        self.findPathBtn.clicked.connect(self.findPathBtnClicked)
        
        if appInfo.only_app_test == False:
            self.errorMessage = ErrorAlertMessage() # ????????? ?????? ?????? ??????
            self.createBtn.clicked.connect(self.createBtnClicked)
        else:
            self.createBtn.clicked.connect(self.createBtnClickedNoValid)

    @pyqtSlot()
    def createBtnClickedNoValid(self):
        # DataInputWindow??? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)

    @pyqtSlot()
    def createBtnClicked(self):
        '''?????? ??? ???????????? ????????? ??????'''
        if (not isinstance( self.proj_parent_dir_path, str )) or \
                    (len(self.proj_parent_dir_path) == 0):
            self.errorMessage.setCustomText("???????????? ?????? ????????? ???????????????.")
            self.errorMessage.show()
        elif not os.path.exists( self.proj_parent_dir_path ):
            self.errorMessage.setText("{}??? ???????????? ?????? ???????????? ?????????.".format(self.proj_parent_dir_path))
            self.errorMessage.show()
        elif self.projNameLineEdit.text() == "":
            self.errorMessage.setCustomText("???????????? ????????? ???????????????.")
            self.errorMessage.show()
        else:
            self.proj_dir_path = "{}/{}".format(self.proj_parent_dir_path
                                                , self.projNameLineEdit.text())
            
            ### (??????) ?????? ???????????? ??????????????? ??????, ?????? ??? ??????????????? ??????????????? ??????.
            if os.path.exists(self.proj_dir_path):
                if os.path.isdir(self.proj_dir_path):
                    # ??????????????? ????????? ??????????????? ?????? ?????? ?????? ??????.
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
                        
                    # ?????? ???????????? ??????????????? ????????? ????????? ??????
                    buttonReply = QMessageBox.question(self, 'Warning', u"?????? ???????????? ?????????????????????. ?????????????????????????", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if buttonReply == QMessageBox.Yes:
                        # DataInputWindow??? ??????
                        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
                        # ???????????? ???????????? ??????dmf DataInputWindow??? ??????
                        self.stackedWidget.currentWidget().getProjectDirPath( self.proj_dir_path)
                        print('Yes clicked.')
                    else:
                        print('No clicked.')
                # ????????? ????????? ??????????????? ????????? ??????????????? ????????? ??? ??????. --> ?????? ?????? ????????? ?????? ???. ????????? ????????? ????????? ?????????
                else:
                    self.errorMessage.setCustomText("{}??? ??????????????? ????????????..".format(self.proj_dir_path))
                    self.errorMessage.show()
            ### ?????? ????????? ??????????????? ??????, ????????? ?????????????????? ?????? ??????.
            else:
                # ??????????????? ?????? ????????? ????????? ??????
                buttonReply = QMessageBox.question(self, 'Warning', u"??????????????? ?????? ???????????????. ?????????????????????????", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if buttonReply == QMessageBox.Yes:
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/input"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/output"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/input/query"))
                    os.makedirs("{}/{}".format(self.proj_dir_path, "data/output/analysis"))
                    # DataInputWindow??? ??????
                    self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
                    # ???????????? ???????????? ??????dmf DataInputWindow??? ??????
                    self.stackedWidget.currentWidget().getProjectDirPath( self.proj_dir_path)
                    print('Yes clicked.')
                else:
                    print('No clicked.')
    @pyqtSlot()
    def findPathBtnClicked(self):
        self.proj_parent_dir_path = QFileDialog.getExistingDirectory(self, 'Select a Directory')
        self.projParentDirLabel.setText(self.proj_parent_dir_path)
        print("click btn : {}".format(self.proj_parent_dir_path))
    
class DataInputWindow(QDialog):
    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/dataInput.ui", self)
        self.stackedWidget = widget
        self.project_dir_path = ""
        self.query_dir_path = ""
        self.video_dir_path = ""
        self.video_paths = []
        self.photo_paths = []

        self.addPhotoBtn.clicked.connect(self.addPhotoBtnClicked)
        self.addVideoBtn.clicked.connect(self.addVideoBtnClicked)
        self.showResultBtn.clicked.connect(self.showResultBtnClicked)
        
        if appInfo.only_app_test == False:
            self.errorMessage = ErrorAlertMessage() # ????????? ?????? ?????? ??????
            self.startAnalysisBtn.clicked.connect(self.startAnalysisBtnClicked)
        else:
            self.startAnalysisBtn.clicked.connect(self.startAnalysisBtnClickedNoValid)
        

    def getProjectDirPath(self, project_dir_path):
        self.project_dir_path = project_dir_path
        self.query_dir_path = "{}/{}".format(self.project_dir_path, "data/input/query")
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/input")
        # ????????? ?????? ??????????????? ??????, ?????? ???????????? ???????????? ListWidget??? ????????????.
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
        '''?????? ????????? ?????????'''
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
        '''?????? ????????? ?????????'''
        '''?????? ?????? ??? ?????? ????????? ??? ?????????'''
        '''????????? ???????????? ???????????? ?????????'''
        '''?????? ?????? ?????? ????????? ??????'''
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
        '''?????? ????????? ?????? ????????? ??????(???????????? ????????? ???)'''
        if len(self.photo_paths) == 0:
            self.errorMessage.setText("????????? ????????? ???????????????.")
            self.errorMessage.show()
        elif len(self.video_paths) == 0:
            self.errorMessage.setText("????????? ?????? ????????? ???????????????.")
            self.errorMessage.show()
        else:
            # AnalysisWindow??? ??????
            self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
            # ?????? thread ??????
            self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path, self.photo_paths, self.video_paths)
            self.stackedWidget.currentWidget().start()

    def startAnalysisBtnClickedNoValid(self):
        '''?????? ????????? ?????? ????????? ?????? X'''
        # AnalysisWindow??? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
        # ?????? thread ??????
        self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path, self.photo_paths, self.video_paths)
        self.stackedWidget.currentWidget().start()

    def showResultBtnClicked(self):
        '''?????? ?????? ???????????? ??????'''
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+2)
        self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path)

    def insertWidgetInListWidget(self, widget, listWidget):
        '''
            QListWidget??? QWidget ????????? ???????????? ??????
        '''
        item = QListWidgetItem( listWidget )
        item.setSizeHint( widget.sizeHint() )
        listWidget.setItemWidget( item, widget )
        listWidget.addItem( item )

class AnalysisWindow(QDialog):

    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/analysis.ui", self)
        self.stackedWidget = widget
        self.setting_path = "" # ?????? ????????? setting file; runInfo.py
        self.repo_path = ""
        self.project_dir_path = ""
        self.query_dir_path = ""
        self.input_video_dir_path = ""
        self.output_video_dir_path = ""
        self.result_dir_path = ""

        self.photo_paths = []
        self.video_paths = []
        
        self.running = False
        self.timer = 0
        self.playTime = 3
        self.currrentVideoCnt = 0
        self.showRsltBtn.clicked.connect(self.showRsltBtnClicked)
        self.nextVideoBtn.clicked.connect(self.nextVideoRunBtnClicked)

        # if appInfo.only_app_test == False and appInfo.sync_analysis_system == True:
        
            
    def getProjectDirPath(self, project_dir_path, photo_paths, video_paths):
        self.project_dir_path = project_dir_path
        self.query_dir_path = "{}/{}".format(self.project_dir_path, "data/input/query")
        self.input_video_dir_path = "{}/{}".format(self.project_dir_path, "data/input")
        self.output_video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")

        self.photo_paths = photo_paths
        self.video_paths = video_paths

        # project ??????????????? ?????? ????????? ?????? ?????? ????????? ????????? ???.
        self.repo_path = appInfo.repo_path
        print("repo path: {}".format(self.repo_path))
        # runInfo.py??? path
        self.setting_path = "{}/{}".format(self.repo_path, "configs/runInfo.py")
        print("runInfo path: {}".format(self.setting_path))
        
    def writeRunInfoFile(self, video_index):
        '''
            runInfo ????????? ????????????
        '''
        project_name        = self.project_dir_path.split('/')[-1]
        input_video_name    = self.video_paths[video_index].split('/')[-1]        # ****************************** ?????? video_paths??? ????????? ?????? ??????
        output_video_name   = input_video_name.split('.')[0] + ".avi"   # input video name?????? ???????????? avi??? ??????
        
        # setting_file = open(self.setting_path, "w", encoding="utf8")

        # contents = getRunInfoFileContents(  input_video_path        = project_name + '/data/input/' + input_video_name, 
        #                                     query_image_path        = project_name + '/data/input/query/',
        #                                     output_json_path        = project_name + '/data/output/analysis/' + input_video_name.split('.')[0] +'.json', 
        #                                     output_video_path       = project_name + '/data/output/' + output_video_name, 
        #                                     output_contactors_path  = project_name + '/data/output/analysis/',
        #                                     )
        # setting_file.write(contents)
        # setting_file.close()

        from configs import runInfo
        runInfo.input_video_path        = project_name + '/data/input/' + input_video_name
        runInfo.query_image_path        = project_name + '/data/input/query/'
        runInfo.output_json_path        = project_name + '/data/output/analysis/' + input_video_name.split('.')[0] +'.json'
        runInfo.output_video_path       = project_name + '/data/output/' + output_video_name
        runInfo.output_contactors_path  = project_name + '/data/output/analysis/'
        runInfo.start_frame             = appInfo.start_frame
        runInfo.end_frame               = appInfo.end_frame

        # runInfo.logfile_name            = appInfo.logfile_name
        # runInfo.console_log_level       = appInfo.console_log_level
        # runInfo.file_log_level          = appInfo.file_log_level

        # runInfo.write_result            = appInfo.write_result

        # runInfo.parallel_processing     = appInfo.parallel_processing
        # runInfo.use_mask_voting         = appInfo.use_mask_voting

        # runInfo.reid_model              = appInfo.reid_model

        # runInfo.trackingGPU             = appInfo.trackingGPU
        # runInfo.reidGPU                 = appInfo.reidGPU
        # runInfo.faceGPU                 = appInfo.faceGPU
        # runInfo.maskGPU                 = appInfo.maskGPU
        
    def stop(self):
        print(" *** stop - before call exit system")
        if appInfo.only_app_test == False and appInfo.sync_analysis_system == True:
            import psutil
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True): 
                child.kill()
            
        if self.running != False:
            self.running = False
        print("stopped..")

    def start(self):
        self.running = True
        self.displaySetNum = math.ceil(len(self.video_paths) / 4)
        # self.startTimer()
        print("started..")
    
        if appInfo.only_app_test == False:
            # runInfo ?????? ??????
            self.writeRunInfoFile(self.currrentVideoCnt)
            if appInfo.sync_analysis_system == True:
                # print(" *** start - run analysis system")
                
                self.videoNameLabel.setText(self.video_paths[ self.currrentVideoCnt ])
                self.videoShowLabel.setText("?????? ?????? ????????????. ????????? ????????? ?????????.")
                
                loop = QEventLoop()
                QTimer.singleShot(10, loop.quit) #25ms
                loop.exec_()
                # self.videoShowLabel.repaint()
                
                from multiprocessing import Process
                from utils.types import ShmManager, ShmSerialManager
                from utils.logger import make_logger 
                from configs import runInfo
                from timeit import time

                from deep_sort_yolo4.person_detection import detectAndTrack
                from distance import checkDistance
                from personReid.personReid import runPersonReid
                from maskdetect.maskProcess import runMaskDetection
                from write_video import writeVideo
                from accuracy_check.file_io import writeShmToJsonFile

                input_video_path = runInfo.input_video_path
                print(" *** start : input_video_path: {}".format(input_video_path))
                start_frame = runInfo.start_frame
                end_frame = runInfo.end_frame

                # ????????? ?????? ?????? ?????? ????????? ??????
                FRAMES_SIZE = end_frame - start_frame + 1
                # ?????? ?????? ?????? ?????? ????????? ??????
                PEOPLE_SIZE = FRAMES_SIZE * 5

                # ????????? ????????? ??? ??????
                FRAME_NUM = end_frame - start_frame + 1
                # ????????? ?????? ?????? ?????????
                MAX_PEOPLE_NUM = 8

                logger = make_logger(runInfo.logfile_name, 'root')

                startTime = time.time()
                if runInfo.parallel_processing:
                    shm = ShmManager(processNum=5, framesSize=FRAMES_SIZE, peopleSize=PEOPLE_SIZE)
                else:
                    shm = ShmSerialManager(processNum=5, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
                
                maskProc = Process(target=runMaskDetection, args=(shm, 3, os.getpid()))
                maskProc.start()

                distanceProc = Process(target=checkDistance, args=(shm, 2, maskProc.pid))
                distanceProc.start()
                
                reidProc = Process(target=runPersonReid, args=(shm, 1, distanceProc.pid, runInfo.reid_model, runInfo.reidGPU)) # (shm, procNo, nxtPid, reidmodel, gpuNo), reid model:'fake'/'topdb'/'la'
                reidProc.start()
                    
                detectTrackProc = Process(target=detectAndTrack, args=(shm, 0, reidProc.pid))
                detectTrackProc.start()
                
                logger.info("Running time: {}".format(time.time() - startTime))
                
                if (not runInfo.parallel_processing) and runInfo.write_result:
                    writeShmToJsonFile(shm.data, start_frame, end_frame, input_video_path)
                    
                self.analysisFromWriteVideo(shm=shm, processOrder=4, nextPid=detectTrackProc.pid)
            else:
                self.analysisWithoutThread()
        else:
            self.analysisWithoutThread()
    
    def analysisFromWriteVideo(self, shm, processOrder, nextPid):
        '''
            appInfo.sync_analysis_system == True ??? ??? ???????????? analysis Window
        '''
        from configs import runInfo
        input_video_path = runInfo.input_video_path
        output_video_path = runInfo.output_video_path
        
        # prepare ResultManager to write output json file  
        res_manager = ResultManager() 
        
        # Prepare input video
        video_capture = cv2.VideoCapture(input_video_path)
        print(" {}'s frame count: {}".format(input_video_path, video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        # ?????? ?????? ????????? runInfo??? ?????? frame ?????? ?????? ??????
        real_end_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if real_end_frame < runInfo.start_frame :
            runInfo.start_frame = real_end_frame
        if real_end_frame < runInfo.end_frame :
            runInfo.end_frame = real_end_frame
        start_frame = runInfo.start_frame
        end_frame = runInfo.end_frame
            
        frame_index = -1
        
        # Prepare output video
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        
        myPid = 'writeVideo'
        shm.init_process(processOrder, myPid, nextPid)
        
        # set to draw on window
        label = self.videoShowLabel
        # get label geometry
        qrect = label.geometry()
        width = qrect.width()
        height = qrect.height()
        
        while self.running:
            ret, frame = video_capture.read()
            if ret != True:
                break
            
            # for test
            frame_index += 1
            if frame_index < start_frame:
                continue
            if frame_index > end_frame:
                break
            if frame_index % 4 != 0 : 
                continue 
            
            # ?????? ?????? ???????????? ?????? ?????? ????????? ????????? ??? ????????? ???.
            # bef_loop = QEventLoop()
            # QTimer.singleShot(10, bef_loop.quit) #25ms
            # bef_loop.exec_()
            
            frameIdx, personIdx = shm.get_ready_to_read()
            
            #update result manager to write json file and contactor photos 
            if start_frame + 2 <= frame_index : 
                save_list = update_output_json(shm, res_manager, frame_index, frameIdx, personIdx) 
                save_contactor_images(frame, shm, personIdx, save_list) 
            
            # Draw detection and tracking result for a frame
            for pIdx in personIdx:
                draw_bbox_and_tid(frame=frame, person=shm.data.people[pIdx], isConfirmed=False)
                
            reid = shm.data.frames[frameIdx].reid
            if reid != -1: # if there is confirmed case
                confirmed = shm.data.people[reid]
                
                # Draw red bbox for confirmed case
                draw_bbox_and_tid(frame=frame, person=confirmed, isConfirmed=True)
                    
                # Draw distance result for a frame
                c_stand_point = getCentroid(bbox=confirmed.bbox, return_int=True)
                for pIdx in personIdx:
                    person = shm.data.people[pIdx]
                    if not person.isClose:
                        continue
                    stand_point = getCentroid(bbox=person.bbox, return_int=True)
                    cv2.line(frame, c_stand_point, stand_point, (0, 0, 255), 3) #red 

                # Draw mask result for a frame
                for pIdx in personIdx:
                    person = shm.data.people[pIdx]
                    square = person.bbox
                    if person.isMask == MaskToken.NotNear : 
                        continue 
                    # save_contactor_images()
                    if person.isMask == MaskToken.NotMasked : 
                        cv2.rectangle(frame, (int(square.minX), int(square.minY)), (int(square.maxX), int(square.maxY)), (127, 127, 255), 3) #light pink
                    elif person.isMask == MaskToken.Masked : 
                        cv2.rectangle(frame, (int(square.minX), int(square.minY)), (int(square.maxX), int(square.maxY)), (127, 255, 127), 3) #light green 
                    elif person.isMask == MaskToken.FaceNotFound : 
                        cv2.rectangle(frame, (int(square.minX), int(square.minY)), (int(square.maxX), int(square.maxY)), (0, 165, 255), 3) #orange 

            # draw on window
            img = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            h,w,c = img.shape
            qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            label.setPixmap(pixmap)
            
            aft_loop = QEventLoop()
            QTimer.singleShot(25, aft_loop.quit) #25ms
            aft_loop.exec_()
            
            # label.repaint()
            
            out.write(frame)

            shm.finish_a_frame()
        
        print(" ******** cv2 ret: ", ret)
        shm.finish_process()
        res_manager.write_jsonfile(runInfo.output_json_path, runInfo.output_video_path, runInfo.start_frame, runInfo.end_frame)
        out.release()
        video_capture.release()
        label.setText("finish")
        
    def analysisWithoutThread(self):
        cap = cv2.VideoCapture(self.video_paths[ self.currrentVideoCnt ])
        
        self.videoNameLabel.setText(self.video_paths[ self.currrentVideoCnt ])
        label = self.videoShowLabel
        # get label geometry
        qrect = label.geometry()
        width = qrect.width()
        height = qrect.height()

        while self.running:
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                h,w,c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                label.setPixmap(pixmap)
            else:
                break
            loop = QEventLoop()
            QTimer.singleShot(10, loop.quit) #25ms
            loop.exec_()
            
        cap.release()
        label.setText("finish")  
        
    def onExit(self):
        print("exit")
        self.stop()            

    # def startTimer(self):
    #     self.timer += 1
    #     self.timeLabel.setText(str(self.timer) + " sec")
    #     timerThread = threading.Timer(1, self.startTimer)
    #     timerThread.start()
    #     if self.running == False:
    #         print("Timer stop")
    #         timerThread.cancel()

    def showRsltBtnClicked(self):
        '''?????? ????????? ?????? ????????? ?????? ??????'''
        # ?????? ?????? ??????????????? ??????
        # print(" *** showRsltBtnClicked")
        self.stop()
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
        self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path)

    def nextVideoRunBtnClicked(self):
        # print(" *** nextVideoRunBtnClicked - before stop")
        self.stop()
        self.currrentVideoCnt += 1
        if self.currrentVideoCnt < len(self.video_paths):
            # print(" *** nextVideoRunBtnClicked - before start")
            self.start()
        else:
            QMessageBox.information(self, 'Complete', '?????? ?????? ????????? ?????????????????????.')
            
class ResultListWindow(QDialog):
    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/resultList.ui", self)
        self.stackedWidget = widget
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
        self.targetInfoList, self.contactorInfoList = loadJson(video_dir_path=self.video_dir_path,
                                                               contactor_dir= self.result_dir_path, 
                                                               result_json_dir= self.result_dir_path)

    def showRouteRsltBtnClicked(self):
        # RouteOfConfirmedCaseWindow??? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+1)
        # setting?????? ????????? ????????? ??????????????? ???
        if self.showRootRsltBtnClicked_Once == False:
            self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path, self.targetInfoList)
            self.stackedWidget.currentWidget().showResult()
            self.showRootRsltBtnClicked_Once = True

    def showContactorLstBtnClicked(self):
        # ContactorListWindow??? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()+2)
        # setting?????? ????????? ????????? ??????????????? ???
        if self.showContactorLstBtnClicked_Once == False:
            self.stackedWidget.currentWidget().getProjectDirPath(self.project_dir_path, self.contactorInfoList)
            self.stackedWidget.currentWidget().showContactor()
            self.showContactorLstBtnClicked_Once = True

class RouteOfConfirmedCaseWindow(QDialog):
    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/routeOfConfirmedCase.ui", self)
        self.stackedWidget = widget
        self.project_dir_path = ""
        self.video_dir_path = ""
        self.result_dir_path = ""
        self.targetInfoList = []
        # self.showResult()
        self.backBtn.clicked.connect(self.backBtnClicked)
        self.comboBox.currentIndexChanged.connect(self.comboBoxIndexChanged)

    def getProjectDirPath(self, project_dir_path, targetInfoList):
        self.project_dir_path = project_dir_path
        self.video_dir_path = "{}/{}".format(self.project_dir_path, "data/output")
        self.result_dir_path = "{}/{}".format(self.project_dir_path, "data/output/analysis")
        self.targetInfoList = targetInfoList

    def comboBoxIndexChanged(self):
        if self.comboBox.currentIndex() == 0:
            print("\t\tChoose ")
            # print(type(self.comboBox.currentText()))
            print(self.comboBox.currentText().encode('utf-8'))
            self.drawAboveTable('IN ??????')
        elif self.comboBox.currentIndex() == 1:
            print("\t\tChoose ")
            # print(type(self.comboBox.currentText()))
            print(self.comboBox.currentText().encode('utf-8'))
            self.drawAboveTable('?????? ??????')
        self.tableWidget.repaint()
        
    def drawAboveTable(self, sort='IN ??????'):
        # targetListInfo??? 1?????? list??? ?????????
        print(self.targetInfoList)
        if len(self.targetInfoList) == 0:
            print("There is no target information -> np.hstack(self.targetInfoList) is error")
        targetInfoFlattenList = np.hstack(self.targetInfoList)

        if sort == 'IN ??????':
            # IN ?????? sorting??? ?????? IN ?????? ????????? + ?????? ?????? ????????? ?????? ??????
            for row, targetInfo in enumerate(targetInfoFlattenList):
                # ?????? ?????? ??????, ?????? ?????? ?????????
                video_start_clock = get_video_start_clock(targetInfo['video_name'].split('/')[-1])
                targetInfo['real_in'] = targetInfo['in'] + getFrameFromClock(video_start_clock, targetInfo['fps'])
            targetInfoFlattenList = sorted(targetInfoFlattenList, key=lambda x : x['real_in'])
        elif sort == '?????? ??????':
            targetInfoFlattenList = sorted(targetInfoFlattenList, key=lambda x : x['video_name'].split('/')[-1])

        # ?????? tableWidget setting
        self.tableWidget.setRowCount( len(targetInfoFlattenList) )
        self.tableWidget.setColumnCount(4)        
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # ?????? table -> ?????? video ????????? ?????? ???????????????.
        # ????????? ?????? ?????? ?????? ??????, ?????? ?????? ?????? ?????????
        min_start_clock = {'month':99, 'day':99, 'h':99, 'm':99}
        max_end_clock = {'month':0, 'day':0, 'h':0, 'm':0, 's':0}
        for row, targetInfo in enumerate(targetInfoFlattenList):
            # ?????? ?????? ??????, ?????? ?????? ?????????
            video_start_clock = get_video_start_clock(targetInfo['video_name'].split('/')[-1])
            video_end_clock = get_video_end_clock(video_start_clock, targetInfo['frame_no'], targetInfo['fps'])
            # print('\033[42m video start clock: {} \033[0m'.format(video_start_clock) )
            # print('\033[42m video end clock: {} \033[0m'.format(video_end_clock) )
            min_start_clock = compare_video_clock(video_start_clock, min_start_clock, 'min')
            max_end_clock = compare_video_clock(video_end_clock, max_end_clock, 'max')
            
            print(targetInfo)
            result = [ targetInfo['video_name'].split('/')[-1],
                        str(targetInfo['index']),
                        getTimeFromFrame(targetInfo['in']+getFrameFromClock(video_start_clock, targetInfo['fps']), targetInfo['fps']), 
                        getTimeFromFrame(targetInfo['out']+getFrameFromClock(video_start_clock, targetInfo['fps']), targetInfo['fps'])]
            
            for col in range(4):
                self.tableWidget.setItem(row, col, 
                                        QTableWidgetItem(result[col]))
        return min_start_clock, max_end_clock
    
    def drawBelowList(self, min_start_clock, max_end_clock):
        # print('\033[102m min start clock: {} \033[0m'.format(min_start_clock) )
        # print('\033[102m max start clock: {} \033[0m'.format(max_start_clock) )
        timelineList = []
        # ????????? list -> ??? video????????? ?????? timeline??? ????????? ???.
        for targetInfoListOfEachVideo in self.targetInfoList:
            print('in showResult')

            if len(targetInfoListOfEachVideo) == 0:
                # ???????????? ?????? ????????? ????????? ???????????? ??????.
                continue
            else:
                # (????????? list) ?????? ?????? ??????
                videoName = targetInfoListOfEachVideo[0]['video_name'].split('/')[-1]
                videoNameWidget = QLabel( videoName )
                # print(videoName)
                
                video_start_clock = get_video_start_clock(videoName)
                video_end_clock = get_video_end_clock(video_start_clock, targetInfoListOfEachVideo[0]['frame_no'], targetInfoListOfEachVideo[0]['fps'])
                
                fps = targetInfoListOfEachVideo[0]['fps']
                interval_start_time = getFrameFromClock(video_start_clock, fps) - getFrameFromClock(min_start_clock, fps)
                interval_mid_time = getFrameFromClock(video_end_clock, fps) - getFrameFromClock(video_start_clock, fps)
                interval_end_time = getFrameFromClock(max_end_clock, fps) - getFrameFromClock(video_end_clock, fps)
                # print('\033[41m min start clock: \n\tis {}\n\tim {}\n\tie {} \033[0m'.format(interval_start_time, interval_mid_time, interval_end_time) )
        
                # ?????? ?????? ????????? 0?????? ????????? ???????????? ??????.
                # if interval_mid_time == 0:
                #     print("\033[42m [Warning] {}??? ?????? ????????? ????????? 0 frame????????? ??????????????? ???????????? ????????????. \033[0m".format(videoName))
                #     continue
                
                # (????????? list) Timeline widget ??????
                timelineWidget = TimeLineWidget(targetInfoListOfEachVideo, interval_start_time, interval_mid_time, interval_end_time)
            
                timelineList.append((timelineWidget.getFirstInStartTime(), timelineWidget, videoNameWidget))
        
        timelineList = sorted(timelineList, key=lambda x : x[0])
        for timeline in timelineList:
            print('get input time: ',timeline[0])
            self.insertWidgetInListWidget( timeline[1], self.listWidget ) # timeline
            
            timeline[2].setAlignment(Qt.AlignCenter)
            timeline[2].setFixedHeight( timeline[1].height()+4 )
            self.insertWidgetInListWidget( timeline[2], self.listWidget_2 ) # videoname
            
    def showResult(self):
        min_start_clock, max_end_clock = self.drawAboveTable()
        self.drawBelowList(min_start_clock, max_end_clock)
        
    def backBtnClicked(self):
        # ?????? ?????? ??????????????? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()-1)
    
    def insertWidgetInListWidget(self, widget, listWidget):
        '''
            QListWidget??? QWidget ????????? ???????????? ??????
        '''
        item = QListWidgetItem( listWidget )
        item.setSizeHint( widget.sizeHint() )
        listWidget.setItemWidget( item, widget )
        listWidget.addItem( item )

class ContactorListWindow(QDialog):
    '''
        ????????? ????????? ???????????? Window

        Args:
            contactorInfoList: ??????????????? ??????(??????, ?????? ??????..)??? ???????????? ContactorInfo()??? ?????????
    '''
    def __init__(self, widget):
        super().__init__()
        loadUi("./UI/ui/contactorList.ui", self)
        self.stackedWidget = widget
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
                # custom widget??? listWidgetItem?????? ??????????????? ??????.
                custom_widget = ContactorItem(contactorInfo, contactorInfo['video_name'], contactorInfo['fps'])
                item = QListWidgetItem(self.contactorList)

                # listWidgetItem??? custom widget??? ????????? ???????????? ???????????? ??????.
                item.setSizeHint(custom_widget.sizeHint())
                self.contactorList.setItemWidget(item, custom_widget)
                self.contactorList.addItem(item)
                
            else:
                print("Image is not exists: {}")

    def backBtnClicked(self):
        # ?????? ?????? ??????????????? ??????
        self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex()-2)