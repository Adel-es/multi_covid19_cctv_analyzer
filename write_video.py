import cv2
from distance import getCentroid
from utils.types import Masked, NotMasked, NotNear, FaceNotFound
from configs import runInfo

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame

def writeVideo(shm, processOrder, nextPid):
    
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
                if person.isMask == NotNear : 
                    continue 
                elif person.isMask == NotMasked : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (127, 127, 255), 2) #light pink
                elif person.isMask == Masked : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (127, 255, 127), 2) #light green 
                elif person.isMask == FaceNotFound : 
                    cv2.rectangle(frame, (int(square.minX+5), int(square.minY+5)), (int(square.maxX-5), int(square.maxY-5)), (0, 165, 255), 2) #orange 
        out.write(frame)
        shm.finish_a_frame()
        
    shm.finish_process()
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
