
import logging
    
# anaylsis time, file setting 
input_video_path        = "proj/data/input/08_14_2020_1_1.mp4"
query_image_path        = "proj/data/input/query/" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
output_video_path       = "proj/data/output/08_14_2020_1_1.avi"
output_json_path        = "proj/data/output/analysis/08_14_2020_1_1.json"
output_contactors_path  = "proj/data/output/analysis/" 
start_frame = 0
end_frame = -1

if end_frame == -1 : 
    print("end frame is updated!")
    import cv2 
    video_capture = cv2.VideoCapture(input_video_path)
    real_end_frame = int(video_capture.get( cv2.CAP_PROP_FRAME_COUNT )) - 1
    end_frame = real_end_frame

# log setting 
logfile_name = "log.txt"
console_log_level = 20 # select in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
file_log_level = 10

# create output.json or not 
write_result = False

# system setting 
parallel_processing     = True
use_mask_voting = False
use_reid_voting = False

# reid model setting
reid_model              = 'topdb' # 'fake2' / 'fake' / 'topdb' / 'la'

# allocate specific gpu device
trackingGPU             = 0
reidGPU = 6
faceGPU                 = 3
maskGPU                 = 4
    