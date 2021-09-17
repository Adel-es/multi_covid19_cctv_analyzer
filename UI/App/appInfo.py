import os, sys
import logging

# repo_path : ~/muti_covid19_cctv_analyzer
repo_path = os.path.dirname(
    os.path.abspath(os.path.dirname(
        os.path.abspath(os.path.dirname(__file__)))))

# contactor_dir = repo_path + "/data/output/analysis/"
# result_json_dir = repo_path + "/data/output/analysis/"

'''
* only_app_test
    - Value
        True: App ui의 동작만 보고 싶을 때 (프로젝트 생성 및 데이터 입력 동작 생략)
        False: 프로젝트 & 파일 생성 동작 확인하고 싶을 때
'''
only_app_test = False
'''
* sync_system
    - Condition
        only_app_test = True
    - Value
        True: 시스템과 App의 상호작용을 확인하고 싶을 때
        False: 시스템 동작 X
'''
sync_analysis_system = False

''' In runInfo setting '''
start_frame  = 750
end_frame = 800

logfile_name = "log.txt"
# select in [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]
console_log_level = logging.INFO 
file_log_level = logging.DEBUG

write_result = False

parallel_processing = True
use_mask_voting = False

reid_model = 'topdb' # 'fake2' / 'fake' / 'topdb' / 'la'

trackingGPU = 0
reidGPU = 1
faceGPU = 3 
maskGPU = 4