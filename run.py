
import os, sys
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)
sys.path.append(root_path + '/deep_sort_yolo4')
sys.path.append(root_path + '/top-dropblock')

from multiprocessing import Process, Manager
from configs import runInfo
from timeit import time

from demo import detectAndTrack
from distance import checkDistance
from collections import Counter # for test(fake reid)
from main import config_for_topdb, run_top_db_test
from maskdetect.maskDetect import runMaskDetect
from write_video import writeVideo

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

def fakeReid(trackingRslt, reidRslt):
    # Find most frequent tid(person) in video frames
    idList = []
    for aFrameTracking in trackingRslt:
        for idx, person in enumerate(aFrameTracking):
            idList.append(person.tid)
    if len(idList) == 0:
        print("Nobody in this video: {}".format(input_video_path))
        print("Tracking result: {}".format(trackingRslt))
        confirmed_id = -1
    else:
        confirmed_id = Counter(idList).most_common(n=1)[0][0]
    
    # Fill in the reidRslt
    for aFrameTracking in trackingRslt:
        confirmed_idx = -1
        for idx, person in enumerate(aFrameTracking):
            if person.tid == confirmed_id:
                confirmed_idx = idx
                break
        reidRslt.append(confirmed_idx)

def personReid(trackingRslt, reidRslt):
    # CUDA_VISIBLE_DEVICES를 0으로 설정하지 않으면 topdb 돌릴 때 아래와 같은 err가 뜬다
    # TypeError: forward() missing 1 required positional argument: 'x'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    top_db_engine, top_db_cfg = config_for_topdb( root_path , query_image_path=query_image_path)
    run_top_db_test(engine=top_db_engine, cfg=top_db_cfg, 
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path, 
                    tracking_list=trackingRslt, reid_list=reidRslt, 
                    query_image_path=query_image_path)
    # 지금 reidRslt에서 확진자가 없는 경우(-1)는 나오지 않는다. (reid 정확성 문제 때문에)


if __name__ == '__main__':
    startTime = time.time()
    with Manager() as manager:
        # 공유 객체 생성
        tracking = manager.list()
        reid = manager.list()
        distance = manager.list()
        mask = manager.list()
        
        # 프로세스 실행 (영상 단위 처리)
        detectTrackProc = Process(target=detectAndTrack, args=(tracking, ))
        # reidProc = Process(target=fakeReid, args=(tracking, reid))        
        reidProc = Process(target=personReid, args=(tracking, reid))
        distanceProc = Process(target=checkDistance, args=(tracking, reid, distance))
        maskProc = Process(target=runMaskDetect, args=(tracking, reid, distance, mask))
        
        detectTrackProc.start()
        detectTrackProc.join()
        
        reidProc.start()
        reidProc.join()
        
        distanceProc.start()
        distanceProc.join()
        
        maskProc.start() 
        maskProc.join()
        
        print('tracking len : ', len(tracking))
        print('reid len : ', len(reid))
        print('distance len : ', len(distance))
        print('mask len : ', len(mask))
        
        #DEBUG
        if runInfo.debug : 
            print("TrackToken : ")
            print(tracking)
            print("reidResult : ")
            print(reid)
            print("distanceResult : ")
            print(distance)       
            print("MaskToken : ")
            print(mask)
            
        writeVideo(tracking, reid, distance, mask)
        print("Runing time:", time.time() - startTime)