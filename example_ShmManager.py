import os
from configs import runInfo
from timeit import time
from multiprocessing import Process
from utils.types import ShmManager, BBox, ShmSerialManager

# 프레임 단위 정보 저장 배열의 크기
FRAMES_SIZE = 4
# 사람 단위 정보 저장 배열의 크기
PEOPLE_SIZE = 14 

# 영상 프레임 개수
FRAME_NUM = 7
# 각 프레임별 인원 수
PEOPLE_NUM_LIST = [3, 1, 3, 0, 7, 8, 5]
        

def writer(shm, processOrder, nextPid):
    myPid = 'A'
    shm.process_init(processOrder, myPid, nextPid)
    for fIdx in range(FRAME_NUM):
        peopleNum = PEOPLE_NUM_LIST[fIdx]
        frameIdx, personIdx = shm.get_ready_to_write(peopleNum)
        
        # Write at frames
        shm.data.frames[frameIdx].reid = fIdx + 1
        print("{} writer: write frame = {}".format(myPid, shm.data.frames[frameIdx].reid))
        
        for pIdx in personIdx:
            # Write at people
            shm.data.people[pIdx].bbox = BBox(1, 2, 3, 4)
            shm.data.people[pIdx].tid = pIdx + 1
            print("{} writer: write person = {}".format(myPid, shm.data.people[pIdx].tid))
            
        shm.finish_a_frame()
    print("{} writer: finish".format(myPid))
    shm.process_finish()
    
    
def reader_and_writer(shm, processOrder, nextPid):
    myPid = '\tB'
    shm.process_init(processOrder, myPid, nextPid)
    
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        
        # Read frames
        data = shm.data.frames[frameIdx].reid
        print("{} reader_and_writer: read frame {}".format(myPid, data))
        # Write at frames
        shm.data.frames[frameIdx].reid = data * 10
        print("{} reader_and_writer: write frame = {}".format(myPid, shm.data.frames[frameIdx].reid))
        
        for pIdx in personIdx:
            # Read people
            bbox = shm.data.people[pIdx].bbox
            print("{} reader_and_writer: read person {}, {}, {}, {}".format(myPid, bbox.minX, bbox.minY, bbox.maxX, bbox.maxY))
            data = shm.data.people[pIdx].tid
            print("{} reader_and_writer: read person {}".format(myPid, data))
            # Write at people
            shm.data.people[pIdx].tid = data * 10
            print("{} reader_and_writer: write person = {}".format(myPid, shm.data.people[pIdx].tid))
            
        shm.finish_a_frame()
    print("{} reader_and_writer: finish".format(myPid))
    shm.process_finish()
    
    
def reader_and_remover(shm, processOrder, nextPid):
    myPid = '\t\tC'
    shm.process_init(processOrder, myPid, nextPid)
    
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        
        # Read frames
        data = shm.data.frames[frameIdx].reid
        print("{} reader_and_remover: read frame {}".format(myPid, data))
        
        for pIdx in personIdx:
            # Read people
            data = shm.data.people[pIdx].tid
            print("{} reader_and_remover: read person {}".format(myPid, data))
                
        shm.finish_a_frame()
    print("{} reader_and_remover: finish".format(myPid))
    shm.process_finish()
    
    
if __name__ == '__main__':
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