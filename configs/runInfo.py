import logging 

# anaylsis time, file setting 
input_video_path = 'data/input/3people.mp4' # 'testVideo.mp4'
output_video_path = 'data/output/3people.mp4' #'output_test_final.avi'
start_frame = 0
end_frame = 2
query_image_path = 'tempData/query/' # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg

# log setting 
logfile_name = "log.txt"
console_log_level = logging.INFO # select in [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL] 
file_log_level = logging.DEBUG 

# system setting 
parallel_processing = True
use_mask_voting = False 

# reid model setting
reid_model = 'fake' # 'fake' / 'topdb' / 'la'

# allocate specific gpu device
trackingGPU = 0
reidGPU = 1
faceGPU = 2
maskGPU = 3 
