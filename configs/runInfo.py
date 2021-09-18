
import logging

# anaylsis time, file setting 
input_video_path        = "proj/data/input/08_14_2020_1_1.avi"
query_image_path        = "proj/data/input/query/" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
output_video_path       = "proj/data/output/08_14_2020_1_1.avi"
output_json_path        = "proj/data/output/analysis/08_14_2020_1_1.json"
output_contactors_path  = "proj/data/output/analysis/" 
start_frame             = 750
end_frame               = 800

# log setting 
logfile_name            = "log.txt"
console_log_level       = 20 # select in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
file_log_level          = 10

# accuracy check setting
write_result            = False

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
    