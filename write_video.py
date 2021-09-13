import cv2
from typing import List
from distance import getCentroid
from utils.types import MaskToken
from configs import runInfo
from utils.resultManager import Contactor, ResultManager

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
output_contactors_path = runInfo.output_contactors_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame

def writeVideo(shm, processOrder, nextPid, shmQueue = None):
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
                cv2.line(frame, c_stand_point, stand_point, (0, 0, 255), 2) #red 

            # Draw mask result for a frame
            for pIdx in personIdx:
                person = shm.data.people[pIdx]
                square = person.bbox
                if person.isMask == MaskToken.NotNear : 
                    continue 
                # save_contactor_images()
                if person.isMask == MaskToken.NotMasked : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (127, 127, 255), 2) #light pink
                elif person.isMask == MaskToken.Masked : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (127, 255, 127), 2) #light green 
                elif person.isMask == MaskToken.FaceNotFound : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (0, 165, 255), 2) #orange 
        out.write(frame)
        if shmQueue != None: # app에 frame보내는 shared memory
            shmQueue.put(frame)
        shm.finish_a_frame()
        
    shm.finish_process()
    res_manager.write_jsonfile(runInfo.output_json_path, runInfo.output_video_path)
    out.release()
    video_capture.release()

def draw_bbox_and_tid(frame, person, isConfirmed):
    TEXT_UP_FROM_BBOX = 2
    bboxLeftUpPoint = (int(person.bbox.minX), int(person.bbox.minY))
    bboxRightDownPoint = (int(person.bbox.maxX), int(person.bbox.maxY))
    tidText = "ID: " + str(person.tid)
    tidPosition = (bboxLeftUpPoint[0], bboxLeftUpPoint[1]-TEXT_UP_FROM_BBOX)
    
    if isConfirmed:
        bboxColor = (0, 0, 255) # red
        tidColor = (0, 0, 255) # red
    else:
        bboxColor = (255, 255, 255) # white
        tidColor = (0, 255, 0) # green
        
    cv2.rectangle(frame, bboxLeftUpPoint, bboxRightDownPoint, bboxColor, 2)
    cv2.putText(frame, tidText, tidPosition, 0, 8e-4 * frame.shape[0], tidColor, 3)

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