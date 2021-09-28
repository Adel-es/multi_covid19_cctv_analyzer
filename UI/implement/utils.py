import json
import os
import cv2
import logging

from PyQt5.QtWidgets import *
from configs import appInfo
from IPython.utils.timing import clock2

def center(self):
    qr = self.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())
    
def getTimeFromFrame(frame, fps):
    sec = frame/int(fps)

    s = int(sec % 60)
    sec /= 60
    m = int(sec % 60)
    h = int(sec / 60)

    # return {'hour': h, 'minute': m, 'second': s}
    return "{}:{}:{}".format(h,m,s)

        
def loadJson(video_dir_path, contactor_dir, result_json_dir):
    '''
        system 출력 결과 json파일 로드
    '''
    # 접촉자 이미지 디렉토리 경로 유효성 검사
    input_file_validation_test(contactor_dir) 

    # 지정된 dir에서 result json 파일들을 찾는다.
    result_json_dir = result_json_dir
    result_json_path = [ "{}/{}".format(result_json_dir, _) for _ in os.listdir(result_json_dir) if _.endswith(".json")]    

    targetInfoList, contactorInfoList = [], []
    for json_path in result_json_path:
        # json 파일 로드
        json_file_validation_test(json_path) #json 파일 유효성 검사
        with open(json_path) as json_file:
            result_json = json.load(json_file)
        _targetInfoList, _contactorInfoList = result_json['target'], result_json['contactor']

        # video의 frame no, fps 구하기
        video_file_name = result_json['video_name'].split('/')[-1]
        output_video_path = "{}/{}".format(video_dir_path, video_file_name)
        print(output_video_path)
        output_file_validation_test(output_video_path) # 출력 영상 경로 유효성 검사
        video_capture = cv2.VideoCapture(output_video_path)
        video_frameno = int(video_capture.get( cv2.CAP_PROP_FRAME_COUNT ))
        video_fps = int(video_capture.get( cv2.CAP_PROP_FPS ))
        video_start_frame = result_json['start_frame']
        video_end_frame = result_json['end_frame']
        
        # targetInfoList에 video 정보 추가하기
        for idx, info in enumerate(_targetInfoList):
            info['video_name'] = result_json['video_name'].split('/')[-1]
            info['frame_no'] = video_frameno
            info['frame_start'] = video_start_frame
            info['frame_end'] = video_end_frame
            info['fps'] = video_fps
            info['index'] = idx

        # contactorInfoList에 video정보 추가하기
        for info in _contactorInfoList:
            info['video_name'] = result_json['video_name'].split('/')[-1]
            info['frame_no'] = video_frameno
            # info['frame_start'] = video_start_frame
            # info['frame_end'] = video_end_frame
            info['fps'] = video_fps
            info['image_path'] = contactor_dir + "/fr{}_tid{}.jpg".format(info['capture_time'], info['tid'])
            
            # 만약 json에 mask 라벨이 안붙어있다면 (구버전) 임의로 추가함
            if not (info.get('target_mask') and info.get('contactor_mask')):
                # print('\033[42m'+ '[Warning] {} is old version result file.'.format(json_path) + '\033[0m')
                info['target_mask'] = "UnKnown"
                info['contactor_mask'] = "UnKnown"
            # else:
                # print('\033[43m'+ '[Warning] {} is new version result file.'.format(json_path) + '\033[0m')
                

        # targetInfoList는 video간 timeline을 그려야 하므로 append
        targetInfoList.append(_targetInfoList)
        # contactorList는 전체 결과를 sorting해야 하므로 extend
        contactorInfoList.extend(_contactorInfoList)

    # 전체 결과를 넣은 contactorList를 danger_level에 대해 sorting
    contactorInfoList = sorted( contactorInfoList, key=lambda info : info['danger_level'], reverse=True)

    return targetInfoList, contactorInfoList

def input_file_validation_test(contactor_dir):
    # 접촉자 사진 경로 존재 여부 확인
    if os.path.exists(contactor_dir):
        if len(os.listdir(contactor_dir)) == 0:
            print(" [Error] {}: Empty Directory".format(contactor_dir))
            exit(-1)
    else:
        print(" [Error] {}: There is no such path".format(contactor_dir))
        exit(-1)

def json_file_validation_test(json_path):
    # json 파일 경로 존재 여부 확인
    if not os.path.exists(json_path):
        print(" [Error] {}: There is no such json file".format(json_path))
        exit(-1)

def output_file_validation_test(output_video_path):    
    # 출력 비디오 존재 여부 확인
    if not os.path.exists(output_video_path):
        print(" [Error] {}: There is no such video file".format(output_video_path))
        exit(-1)

def get_video_start_clock(video_name):
    import re
    # print(video_name)
    p = re.compile(r"(\d+).(\d+).(\d\d)(\d\d).(\d+).(\d+).+")
    m = re.findall(p, video_name)
    # print(m)
    clock = {'month':int(m[0][0]), 'day':int(m[0][1]),'h':int(m[0][2]),'m':int(m[0][3]), 's':0}
    return clock

def compare_video_clock(clock1, clock2, cmp='min'):
    if cmp == 'min':
        for k in ['month', 'day', 'h', 'm', 's']:
            if(clock1[k] < clock2[k]):
                return clock1
            elif(clock1[k] > clock2[k]):
                return clock2
        return clock1
    elif cmp == 'max':
        for k in ['month', 'day', 'h', 'm', 's']:
            if(clock1[k] > clock2[k]):
                return clock1
            elif(clock1[k] < clock2[k]):
                return clock2
        return clock1
    else:
        print("\033[91m [Error] utils.py - compare_video_clock() : parameter 'cmp' is expected 'min' or 'max', but now is {}. \033[0m".format(cmp))

def get_video_end_clock(video_start_clock, video_frame_no, video_fps):
    total_sec = int(video_frame_no / video_fps)
    hour = int(total_sec / 3600)
    total_sec %= 3600
    min = int(total_sec / 60)
    total_sec %= 60
    sec = total_sec
    
    month = video_start_clock['month']
    day = video_start_clock['day']
    hour += video_start_clock['h']
    min += video_start_clock['m']
    sec += video_start_clock['s']
    # if(sec >= 60):
    #     min += sec/60
    #     sec %= 60
    if sec >= 60:
        min += int(sec/60)
        sec %= 60
    if min >= 60:
        hour += int(min/60)
        min %= 60
    if hour >= 24:
        day += int(hour/24)
        hour %= 24
    # day는 아직 필요없으므로 패쓰
    return {'month': month, 'day':day, 'h':hour, 'm':min, 's':sec}

def getFrameFromClock(clock, fps):
    # sec = clock['month'] * 12*24*60*60
    sec = 0
    # sec += clock['day'] * 24*60*60
    sec += clock['h'] * 3600
    sec += clock['m'] * 60
    sec += clock['s']
    return sec * fps
# def getRunInfoFileContents(input_video_path, 
#                     query_image_path,
#                     output_video_path, 
#                     output_json_path, 
#                     output_contactors_path,

#                     start_frame  = appInfo.start_frame, 
#                     end_frame = appInfo.end_frame,

#                     console_log_level = appInfo.console_log_level, 
#                     file_log_level = appInfo.file_log_level, 

#                     write_result = appInfo.write_result,

#                     parallel_processing = appInfo.parallel_processing, 
#                     use_mask_voting = appInfo.use_mask_voting,

#                     reid_model = appInfo.reid_model,

#                     trackingGPU = appInfo.trackingGPU, 
#                     reidGPU = appInfo.reidGPU, 
#                     faceGPU = appInfo.faceGPU, 
#                     maskGPU = appInfo.maskGPU,):
#     contents=\
#     u"""
# import logging

# # anaylsis time, file setting 
# input_video_path        = "{}"
# query_image_path        = "{}" # query image의 이름은 "숫자_숫자_숫자" 로 설정. ex) 1_0_0.jpg
# output_video_path       = "{}"
# output_json_path        = "{}"
# output_contactors_path  = "{}" 
# start_frame             = {}
# end_frame               = {}

# # log setting 
# console_log_level       = {} # select in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL] 
# file_log_level          = {}

# # accuracy check setting
# write_result            = {}

# # system setting 
# parallel_processing     = {}
# use_mask_voting         = {} 

# # reid model setting
# reid_model              = '{}' # 'fake2' / 'fake' / 'topdb' / 'la'

# # allocate specific gpu device
# trackingGPU             = {}
# reidGPU                 = {}
# faceGPU                 = {}
# maskGPU                 = {}
#     """.format(input_video_path, query_image_path,
#                     output_video_path, output_json_path, output_contactors_path,
#                     start_frame, end_frame,
#                     console_log_level, file_log_level, 
#                     write_result,
#                     parallel_processing, use_mask_voting,
#                     reid_model,
#                     trackingGPU, reidGPU, faceGPU, maskGPU,)
#     return contents
