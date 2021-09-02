
import os, sys
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)
sys.path.append(root_path + '/deep_sort_yolo4')

from multiprocessing import Process, Manager
from configs import runInfo
from timeit import time

from demo import detectAndTrack
from distance import checkDistance
from personReid.personReid import runPersonReid
from maskdetect.maskDetect import runMaskDetect
from write_video import writeVideo

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

if __name__ == '__main__':
    startTime = time.time()
    
    startTime = time.time()
    
    if runInfo.parallel_processing:
        shm = ShmManager(processNum=3, framesSize=FRAMES_SIZE, peopleSize=PEOPLE_SIZE)
    else:
        shm = ShmSerialManager(processNum=3, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*max(PEOPLE_NUM_LIST))
    
    p2 = Process(target=reader_and_writer, args=(shm, 1, os.getpid()))
    p2.start()

    p1 = Process(target=writer, args=(shm, 0, p2.pid))
    p1.start()

    reader_and_remover(shm, 2, p1.pid)
    
    print("Runing time:", time.time() - startTime)
    
    with Manager() as manager:
        # 공유 객체 생성
        tracking = manager.list()
        reid = manager.list()
        distance = manager.list()
        mask = manager.list()
        
        # 프로세스 실행 (영상 단위 처리)
        detectTrackProc = Process(target=detectAndTrack, args=(tracking, ))
        reidProc = Process(target=runPersonReid, args=(tracking, reid, 'topdb')) # select topdb / la / fake
        # distanceProc = Process(target=checkDistance, args=(tracking, reid, distance))
        # maskProc = Process(target=runMaskDetect, args=(tracking, reid, distance, mask))
        
        detectTrackProc.start()
        detectTrackProc.join()
        
        reidProc.start()
        reidProc.join()
        
        # distanceProc.start()
        # distanceProc.join()
        
        # maskProc.start() 
        # maskProc.join()
        
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