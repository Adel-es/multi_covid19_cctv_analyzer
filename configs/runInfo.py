
import logging
    
# anaylsis time, file setting 
input_video_path = 'accuracy_check/test_video_data/08_14_2020_4_1.mp4'
query_image_path = "data/input/query/" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
output_video_path = 'tmp.avi'
output_json_path = "data/output/analysis/08_14_2020_1_1.json"
output_contactors_path  = "data/output/analysis/" 
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
use_mask_voting         = True
use_reid_voting         = True

# reid model setting
reid_model              = 'topdb' # 'fake2' / 'fake' / 'topdb' / 'la'

# allocate specific gpu device
trackingGPU             = 0
reidGPU                 = 1
faceGPU                 = 3
maskGPU                 = 4
    