
import logging

# anaylsis time, file setting 
input_video_path        = 'data/input/08_14_2020_1_1.mp4'
output_video_path       = 'data/output/08_14_2020_1_1.avi'

output_json_path        = 'data/output/analysis/output.json'
output_contactors_path  = 'data/output/analysis/' 
start_frame             = 750
end_frame               = 760
query_image_path        = 'data/input/query/' # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg

# log setting 
logfile_name            = "log.txt"
console_log_level       = logging.INFO # select in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
file_log_level          = logging.DEBUG 

# system setting 
parallel_processing     = True
use_mask_voting         = False 

# reid model setting
reid_model              = 'topdb' # 'fake2' / 'fake' / 'topdb' / 'la'


# allocate specific gpu device
trackingGPU             = 0
reidGPU                 = 1
faceGPU                 = 3
maskGPU                 = 4
    