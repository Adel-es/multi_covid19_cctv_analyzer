import signal
import os
import sys
from timeit import time
from multiprocessing import Process
from multiprocessing.sharedctypes import Array
from ctypes import Structure, c_int32, c_float, c_bool, c_uint8

# 프레임 단위 정보 저장 배열의 크기
FRAMES_SIZE = 4
# 사람 단위 정보 저장 배열의 크기
PEOPLE_SIZE = 14 

# 영상 프레임 개수
FRAME_NUM = 7
# 각 프레임별 인원 수
PEOPLE_NUM_LIST = [3, 1, 3, 0, 7, 8, 5]

class FrameInfo(Structure):
    _fields_ = [('reid', c_int32)]

class bbox(Structure):
    _fields_ = [('minX', c_float), ('minY', c_float), ('maxX', c_float), ('maxY', c_float)]

Masked, NotMasked, NotNear = 1, 2, 3 # + FaceNotFound

class PersonInfo(Structure):
    _fields_ = [('bbox', bbox), ('tid', c_int32), ('isClose', c_bool), ('isMask', c_uint8)]


class Data():
    def __init__(self, processNum, framesSize, peopleSize):
        # 프레임 단위의 정보를 저장하는 배열.
        self.frames = Array(FrameInfo, framesSize, lock=False)
        # 사람 단위의 정보를 저장하는 배열.
        self.people = Array(PersonInfo, peopleSize, lock=False)
        # 각 프레임에 존재하는 사람의 수를 누적해서 저장하는 배열. 
        # 특정 프레임 번호에 대한 people 배열의 인덱스를 제공한다.
        self.peopleIdx = Array('l', framesSize+1, lock=False)
        # 각각의 프로세스에 대해 가장 최근에 처리를 완료한 프레임 번호를 저장하는 배열.
        self.finishedFrames = Array('l', processNum, lock=False)
        
    def set_process_order(self, myOrder):
        # myOrder를 통해 몇 번째 프로세스인지, 몇 번째 프레임을 처리해야할 차례인지 구분한다.
        self.myOrder = myOrder
    
    def is_ready_to_write(self, newPeopleNum):
        '''
        첫 번째 프로세스에서 새로운 프레임을 처리하여 공유 메모리에 기록하고자 할 때,
        공유 메모리 공간이 충분히 비어있는지 확인한다.
        '''
        # 첫 번째 프로세스에서만 이 함수를 사용할 수 있다.
        if self.myOrder != 0:
            sys.exit("Write Fail: is_ready_to_write() can be used by first process")
        
        # Check if there is space in the frames array
        writeUpTo = self.finishedFrames[0]
        eraseUpTo = self.finishedFrames[-1]
        filledFrameNum = writeUpTo - eraseUpTo
        if not (filledFrameNum + 1 <= len(self.frames)):
            return False
        
        # Check if there is space in the person array
        recordedPeopleNum = self.peopleIdx[ self.get_index_of(array=self.peopleIdx) ]
        removedPeopleNum = self.peopleIdx[ self.get_index_of(array=self.peopleIdx, processOrder=-1) ]
        filledPeopleNum = recordedPeopleNum - removedPeopleNum
        if not (filledPeopleNum  + newPeopleNum <= len(self.people)):
            return False
        
        return True

    def is_ready_to_read(self):
        '''
        프로세스들이 이전 프로세스의 결과 데이터를 공유 메모리로부터 읽고자 할 때,
        현재 읽고자 하는 프레임을 이전 프로세스가 처리를 완료했는지 확인한다.
        '''
        return self.finishedFrames[self.myOrder] < self.finishedFrames[self.myOrder-1]
    
    def get_next_frame_index(self):
        '''
        다음으로 처리할 프레임의 인덱스를 반환한다.
        framesIdx: frames 배열의 인덱스. 정수 타입.
        peopleIndices: people 배열의 인데스들. List 타입
        '''
        # Get framesIdx
        framesIdx = self.get_index_of(array=self.frames, frameOffset=1)
        
        # Get peopleIndices
        logicalStart = self.peopleIdx[ self.get_index_of(array=self.peopleIdx) ]
        logicalEnd = self.peopleIdx[ self.get_index_of(array=self.peopleIdx, frameOffset=1) ]
        
        physicalStart = self.convert_logical_idx_to_physical(len(self.people), logicalStart)
        physicalEnd = self.convert_logical_idx_to_physical(len(self.people), logicalEnd)
        
        if physicalEnd < physicalStart:
            peopleIndices = list(range(physicalStart, len(self.people))) + list(range(0, physicalEnd))
        else:
            peopleIndices = list(range(physicalStart, physicalEnd))
    
        return framesIdx, peopleIndices
    
    def update_peopleIdx(self, newPeopleNum):
        '''
        첫 번째 프로세스에서 새로운 프레임을 공유 메모리에 쓸 때
        peopleIdx 배열의 끝에 '이전 프레임까지 누적된 사람 수 + newPeopleNum'을 추가한다.
        '''
        # 첫 번째 프로세스에서만 이 함수를 사용할 수 있다.
        if self.myOrder != 0:
            sys.exit("Write Fail: update_peopleIdx() can be used by first process")
        lastPhysicalIdx = self.get_index_of(array=self.peopleIdx)
        updatePhysicalIdx = self.get_index_of(array=self.peopleIdx, frameOffset=1)
        self.peopleIdx[updatePhysicalIdx] = self.peopleIdx[lastPhysicalIdx] + newPeopleNum
    
    def finish_a_frame(self):
        '''
        현재 프레임에 대한 처리를 완료했음을 finishedFrames 배열에 기록하는 함수.
        '''
        self.finishedFrames[self.myOrder] += 1
    
    def get_index_of(self, array, frameOffset=0, processOrder=False):
        '''
        주어진 array 배열에서 '프로세스가 마지막으로 처리한 프레임 + frameOffset'에 해당하는 프레임의 index를 반환한다.
        array로는 frames, peopleIdx 배열이 주어질 수 있다.
        '''
        if processOrder == False:
            logicalIdx = self.finishedFrames[self.myOrder] + frameOffset
        else:
            logicalIdx = self.finishedFrames[processOrder] + frameOffset
        return self.convert_logical_idx_to_physical(len(array), logicalIdx)
    
    def convert_logical_idx_to_physical(self, arraySize, logicalIdx):
        '''
        logical index를 physical index로 바꿔주는 역할.
        arraySize는 어떤 배열의 인덱스로 바꿀 것인지에 따라 frames, peopleIdx, people 배열의 크기가 올 수 있다.
        logical index는 프레임 번호 또는 프레임 누적 사람 번호를 말하고
        physical index는 실제 배열에서 해당 프레임 번호(또는 누적된 사람 수)가 차지하는 index를 말한다.
        '''
        physicalIdx = logicalIdx % arraySize
        return physicalIdx
    
    def get_frame_in_processing(self):
        '''
        처리 중인 프레임 번호를 반환하는 함수. 프레임 번호는 1부터 시작함.
        '''
        return self.finishedFrames[self.myOrder] + 1

class ShmManager():
    def __init__(self, processNum, framesSize, peopleSize):
        self.data = Data(processNum, framesSize, peopleSize)
    
    def process_init(self, myOrder, myPid, nextPid):
        '''
        메모리를 공유하는 각 프로세스에 대한 정보를 초기화하는 함수.
        '''
        self.data.set_process_order(myOrder)
        self.myPid = myPid
        self.nextPid = nextPid
        self.sigList = [signal.SIGUSR1, signal.SIGALRM]
        # handler 등록을 안 하면 시그널  발생 시 프로세스가 종료되어서 dummy handler를 추가함.
        for sig in self.sigList:
            signal.signal(sig, self.dummy_sig_handler)
        print("{} init".format(self.myPid))
    
    def send_ready_signal(self):
        '''
        한 프레임에 대한 처리를 마친 뒤 다음 차례의 프로세스에게 신호를 주는 함수.
        '''
        os.kill(self.nextPid, signal.SIGUSR1)
        print("{} send_ready_signal".format(self.myPid))

    def dummy_sig_handler(self, signum, frame):
        pass
    
    def finish_a_frame(self):
        '''
        한 프레임에 대한 처리를 완료했을 때 호출하는 함수.
        '''
        self.data.finish_a_frame()
        self.send_ready_signal()

    def get_ready_to_write(self, peopleNum):
        '''
        다음 프레임에 대한 정보를 공유 메모리에 쓰기 전에 호출하는 함수.
        준비가 다 되면 data.frames와 data.people 배열에 접근하기 위한 인덱스를 반환한다.
        '''
        while not self.data.is_ready_to_write(peopleNum):
            print("{} get_ready_to_write: Not yet.".format(self.myPid))
            result = signal.sigtimedwait(self.sigList, 60)
            # timeout이 발생했을 경우
            if result == None:
                print("{} get_ready_to_write: wait 1 min (frame: {}))".format(self.myPid, self.data.get_frame_in_processing()))
        print("{} get_ready_to_write: I'm ready!".format(self.myPid))
        self.data.update_peopleIdx(peopleNum);
        framesIdx, peopleIndices = self.data.get_next_frame_index()
        return framesIdx, peopleIndices

    def get_ready_to_read(self):
        '''
        다음 프레임에 대한 정보를 공유 메모리로부터 읽기 전에 호출하는 함수.
        준비가 다 되면 data.frames와 data.people 배열에 접근하기 위한 인덱스를 반환한다.
        '''
        while not self.data.is_ready_to_read():
            print("{} get_ready_to_read: Not yet.".format(self.myPid))
            result = signal.sigtimedwait(self.sigList, 60)
            # timeout이 발생했을 경우
            if result == None:
                print("{} get_ready_to_read: wait 1 min (frame: {}))".format(self.myPid, self.data.get_frame_in_processing()))     
        print("{} get_ready_to_read: I'm ready!".format(self.myPid))
        framesIdx, peopleIndices = self.data.get_next_frame_index()
        return framesIdx, peopleIndices
        

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
            shm.data.people[pIdx].tid = pIdx + 1
            print("{} writer: write person = {}".format(myPid, shm.data.people[pIdx].tid))
            
        shm.finish_a_frame()
    print("{} writer: finish".format(myPid))
    
    
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
            data = shm.data.people[pIdx].tid
            print("{} reader_and_writer: read person {}".format(myPid, data))
            # Write at people
            shm.data.people[pIdx].tid = data * 10
            print("{} reader_and_writer: write person = {}".format(myPid, shm.data.people[pIdx].tid))
            
        shm.finish_a_frame()
    print("{} reader_and_writer: finish".format(myPid))
    
    
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
    
    
if __name__ == '__main__':
    startTime = time.time()
    
    shm = ShmManager(processNum=3, framesSize=FRAMES_SIZE, peopleSize=PEOPLE_SIZE)
    
    p2 = Process(target=reader_and_writer, args=(shm, 1, os.getpid()))
    p2.start()

    p1 = Process(target=writer, args=(shm, 0, p2.pid))
    p1.start()

    reader_and_remover(shm, 2, p1.pid)
    
    print("Runing time:", time.time() - startTime)