
import logging

# anaylsis time, file setting 
input_video_path        = "proa/data/input/1_1.avi"
query_image_path        = "proa/data/output/1_1.avi" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
output_video_path       = "proa/data/output/analysis/"
output_json_path        = "proa/data/output/analysis/1_1.json"
output_contactors_path  = "proa/data/input/query/" 
start_frame             = 750
end_frame               = 800

# log setting 
logfile_name            = "log.txt"
console_log_level       = 20 # select in [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL] 
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
    