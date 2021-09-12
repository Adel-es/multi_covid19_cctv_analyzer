import json
import os
import cv2

from . import appInfo

def getTimeFromFrame(frame, fps):
    sec = frame/int(fps)

    s = int(sec % 60)
    sec /= 60
    m = int(sec % 60)
    h = int(sec / 60)

    # return {'hour': h, 'minute': m, 'second': s}
    return "{}:{}:{}".format(h,m,s)

        
def loadJson():
    '''
        system 출력 결과 json파일 로드
    '''

    # 지정된 dir에서 result json 파일들을 찾는다.
    result_json_dir = appInfo.result_json_dir
    result_json_path = [ "{}/{}".format(result_json_dir, _) for _ in os.listdir(result_json_dir) if _.endswith(".json")]

    targetInfoList, contactorInfoList = [], []
    for json_path in result_json_path:
        # json 파일 로드
        with open(json_path) as json_file:
            result_json = json.load(json_file)
        _targetInfoList, _contactorInfoList = result_json['target'], result_json['contactor']

        # video의 frame no, fps 구하기    
        video_capture = cv2.VideoCapture( "{}/{}".format(appInfo.repo_path, result_json['video_name']))
        video_frameno = video_capture.get( cv2.CAP_PROP_FRAME_COUNT )
        video_fps = video_capture.get( cv2.CAP_PROP_FPS )

        # targetInfoList에 video 정보 추가하기
        for idx, info in enumerate(_targetInfoList):
            info['video_name'] = result_json['video_name']
            info['frame_no'] = video_frameno
            info['fps'] = video_fps
            info['index'] = idx

        # contactorInfoList에 video정보 추가하기
        for info in _contactorInfoList:
            info['video_name'] = result_json['video_name']
            info['frame_no'] = video_frameno
            info['fps'] = video_fps
            info['image_path'] = appInfo.contactor_dir + "/fr{}_tid{}.png".format(info['capture_time'], info['tid'])

        # targetInfoList는 video간 timeline을 그려야 하므로 append
        targetInfoList.append(_targetInfoList)
        # contactorList는 전체 결과를 sorting해야 하므로 extend
        contactorInfoList.extend(_contactorInfoList)

    # 전체 결과를 넣은 contactorList를 danger_level에 대해 sorting
    contactorInfoList = sorted( contactorInfoList, key=lambda info : info['danger_level'], reverse=True)

    return targetInfoList, contactorInfoList
