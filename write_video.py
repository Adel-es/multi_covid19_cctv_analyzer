import cv2
from typing import List
from distance import getCentroid
from utils.types import MaskToken
from configs import runInfo, appInfo
from utils.resultManager import Contactor, ResultManager
import numpy as np

# from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
# if appInfo.only_app_test==False and appInfo.sync_analysis_system==True:

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
output_contactors_path = runInfo.output_contactors_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame

# class SendResultFrameSignal(QObject):
#     sig = pyqtSignal(np.ndarray)
#     def __init__(self, func):
#         super().__init__()
#         self.sig.connect(func)
        
#     def run(self, frame):
#         self.sig.emit(frame)
        
def writeVideoSyncWithUI(shm, processOrder, nextPid):
    # from UI.implement.mainWindows import AnalysisWindow
    # sendResultFrameSignal = SendResultFrameSignal(analysisClass.receiveAnalysisResultPersonInfo)
    # print(" *************** [write_video] sendResultFrameSignal 객체 생성")
    # print(" *************** writeVideoSyncWithUI")
    
    # prepare ResultManager to write output json file  
    res_manager = ResultManager() 
    
    # Prepare input video
    video_capture = cv2.VideoCapture(input_video_path)
    frame_index = -1
    
    # Prepare output video
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
    
    myPid = 'writeVideo'
    shm.init_process(processOrder, myPid, nextPid)
    
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        
        # for test
        frame_index += 1
        if frame_index < start_frame:
            continue
        if frame_index > end_frame:
            break
        # for test 
        
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

        # sendResultFrameSignal.run(np.array([frameIdx, personIdx, shm.data.frames[frameIdx].reid, shm.data.people]))
        out.write(frame)
        shm.finish_a_frame()
        
    shm.finish_process()
    res_manager.write_jsonfile(runInfo.output_json_path, runInfo.output_video_path, runInfo.start_frame, runInfo.end_frame)
    out.release()
    video_capture.release()

def writeVideo(shm, processOrder, nextPid):            
    # prepare ResultManager to write output json file  
    res_manager = ResultManager() 
    
    # Prepare input video
    video_capture = cv2.VideoCapture(input_video_path)
    frame_index = -1
    
    # Prepare output video
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
    
    myPid = 'writeVideo'
    shm.init_process(processOrder, myPid, nextPid)
    
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        
        # for test
        frame_index += 1
        if frame_index < start_frame:
            continue
        if frame_index > end_frame:
            break
        # for test 
        
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

        out.write(frame)

        shm.finish_a_frame()
        
    shm.finish_process()
    res_manager.write_jsonfile(runInfo.output_json_path, runInfo.output_video_path, runInfo.start_frame, runInfo.end_frame)
    out.release()
    video_capture.release()
    
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