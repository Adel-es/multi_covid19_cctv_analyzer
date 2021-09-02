import os
from multiprocessing import Process
from utils.types import ShmManager, ShmSerialManager
from configs import runInfo
from timeit import time

from deep_sort_yolo4.person_detection import detectAndTrack
from distance import checkDistance
import random # for fakeReid
from personReid.personReid import runPersonReid
from maskdetect.maskDetect import runMaskDetect
from write_video import writeVideo

input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

def fakeReid(shm, processOrder, nextPid):
    myPid = 'fakeReid'
    shm.init_process(processOrder, myPid, nextPid)
    
    # select confirmed case randomly
    FRAME_NUM = end_frame - start_frame + 1
    doneFIdx = FRAME_NUM - 1
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        if len(personIdx) > 0:
            random_pIdx = random.choice(personIdx)
            confirmed_tid = shm.data.people[random_pIdx].tid
            shm.data.frames[frameIdx].reid = random_pIdx
            shm.finish_a_frame()
            doneFIdx = fIdx
            break
        shm.data.frames[frameIdx].reid = -1
        shm.finish_a_frame()
    
    # find confirmed_tid
    for fIdx in range(doneFIdx + 1, FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        shm.data.frames[frameIdx].reid = -1
        for pIdx in personIdx:
            if shm.data.people[pIdx].tid == confirmed_tid:
                shm.data.frames[frameIdx].reid = pIdx
                break
        shm.finish_a_frame()
        
    shm.finish_process()

# 프레임 단위 정보 저장 배열의 크기
FRAMES_SIZE = end_frame - start_frame + 1
# 사람 단위 정보 저장 배열의 크기
PEOPLE_SIZE = FRAMES_SIZE * 5

# 처리할 프레임 총 개수
FRAME_NUM = end_frame - start_frame + 1
# 프레임 인원 수의 상한선
MAX_PEOPLE_NUM = 10

if __name__ == '__main__':
    startTime = time.time()
    
    if runInfo.parallel_processing:
        # shm = ShmManager(processNum=5, framesSize=FRAMES_SIZE, peopleSize=PEOPLE_SIZE)
        shm = ShmManager(processNum=4, framesSize=FRAMES_SIZE, peopleSize=PEOPLE_SIZE)
    else:
        # shm = ShmSerialManager(processNum=5, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
        shm = ShmSerialManager(processNum=4, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
    
    # maskProc = Process(target=runMaskDetect, args=(shm, 3, os.getpid()))
    # maskProc.start()

    # distanceProc = Process(target=checkDistance, args=(shm, 2, maskProc.pid))
    distanceProc = Process(target=checkDistance, args=(shm, 2, os.getpid()))
    distanceProc.start()
    
    reidProc = Process(target=fakeReid, args=(shm, 1, distanceProc.pid))
    reidProc.start()

    detectTrackProc = Process(target=detectAndTrack, args=(shm, 0, reidProc.pid))
    detectTrackProc.start()

    # writeVideo(shm, 4, detectTrackProc.pid)
    writeVideo(shm, 3, detectTrackProc.pid)
    
    print("Running time:", time.time() - startTime)
