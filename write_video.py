import cv2
from distance import getCentroid
from utils.types import MaskToken
from configs import runInfo

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame

def writeVideo(tracking, reid, distance, mask):
    # Prepare input video
    video_capture = cv2.VideoCapture(input_video_path)
    frame_index = -1
        
    # Prepare output video
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        
    # for test
    while True:
        if start_frame == 0:
            break
        ret, frame = video_capture.read()
        frame_index += 1
        if ret != True:
            break
        if frame_index < start_frame-1:
            continue
        elif frame_index == start_frame-1:
            break
        else:
            print("Frame capture error! Check start_frame and end_frame: {}, {}".format(start_frame, end_frame))
    # for test
        
    for aFrameTracking, aFrameReid, aFrameDistance, aFrameMask in zip(tracking, reid, distance, mask):
        ret, frame = video_capture.read()
        frame_index += 1
        if ret != True:
            break
        # for test
        if frame_index > end_frame:
            break
        # for test
            
        # Draw detection and tracking result for a frame
        TEXT_UP_FROM_BBOX = 2
        for person in aFrameTracking:
            cv2.rectangle(frame, (int(person.bbox[0]), int(person.bbox[1])), (int(person.bbox[2]), int(person.bbox[3])), (255, 255, 255), 2) #white 
            cv2.putText(frame, "ID: " + str(person.tid), (int(person.bbox[0]), int(person.bbox[1])-TEXT_UP_FROM_BBOX), 0,
                            8e-4 * frame.shape[0], (0, 255, 0), 3) #green 
            
        if aFrameReid != -1: # if there is confirmed case
            # Draw red bbox for confirmed case
            confirmed = aFrameTracking[aFrameReid]
            cv2.rectangle(frame, (int(confirmed.bbox[0]), int(confirmed.bbox[1])), (int(confirmed.bbox[2]), int(confirmed.bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, "ID: " + str(confirmed.tid), (int(confirmed.bbox[0]), int(confirmed.bbox[1])-TEXT_UP_FROM_BBOX), 0,
                            8e-4 * frame.shape[0], (0, 0, 255), 3) #red 
                
            # Draw distance result for a frame
            c_stand_point = getCentroid(bbox=confirmed.bbox, return_int=True)
            for idx, is_close in enumerate(aFrameDistance):
                if not is_close:
                    continue
                closePerson = aFrameTracking[idx]
                stand_point = getCentroid(bbox=closePerson.bbox, return_int=True)
                cv2.line(frame, c_stand_point, stand_point, (0, 0, 255), 2) #red 

            for idx, is_masked in enumerate(aFrameMask) : 
                square = aFrameTracking[idx].bbox
                if is_masked == MaskToken.NotNear : 
                    continue 
                elif is_masked == MaskToken.NotMasked : 
                    cv2.rectangle(frame, (int(square[0]+5), int(square[1]+5)), (int(square[2]-5), int(square[3]-5)), (127, 127, 255), 2) #light pink
                elif is_masked == MaskToken.Masked : 
                    cv2.rectangle(frame, (int(square[0]+5), int(square[1]+5)), (int(square[2]-5), int(square[3]-5)), (127, 255, 127), 2) #light green 
                elif is_masked == MaskToken.FaceNotFound : 
                    cv2.rectangle(frame, (int(square[0]+5), int(square[1]+5)), (int(square[2]-5), int(square[3]-5)), (0, 165, 255), 2) #orange 
        out.write(frame)
        
    out.release()
    video_capture.release()
