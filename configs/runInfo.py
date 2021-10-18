
import logging
    
# anaylsis time, file setting 
input_video_path = 'data/input/08_14_2020_3_1.mp4'
query_image_path = "data/input/query/" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
output_video_path = 'data/output/08_14_2020_3_1.avi'
output_json_path = "data/output/analysis/08_14_2020_3_1.json"
output_contactors_path  = "data/output/analysis/"  # Save path of images displayed on contactor list
start_frame = 0
end_frame = -1  # If end_frame is -1, run to the end of the video

if end_frame == -1 : 
    import cv2 
    video_capture = cv2.VideoCapture(input_video_path)
    real_end_frame = int(video_capture.get( cv2.CAP_PROP_FRAME_COUNT )) - 1
    end_frame = real_end_frame

# log setting 
logfile_name = "log.txt"
console_log_level = logging.WARNING # select in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
file_log_level = logging.DEBUG

# create output.json or not 
write_result = False

# system setting 
parallel_processing     = True  # If false, run serially
use_mask_voting         = True
use_reid_voting         = True

# reid model setting
reid_model              = 'topdb' # 'topdb' / 'la' (and for test 'fake' / 'fake2')
reid_threshold          = 0.975

# allocate specific gpu device
trackingGPU             = 0
reidGPU                 = 1
faceGPU                 = 2
maskGPU                 = 3