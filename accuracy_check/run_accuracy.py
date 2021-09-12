import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from multiprocessing import Process
from utils.types import ShmSerialManager
from utils.logger import make_logger 
from configs import runInfo
from timeit import time

from deep_sort_yolo4.person_detection import detectAndTrack
from distance import checkDistance
from personReid.personReid import runPersonReid
from maskdetect.maskProcess import runMaskDetection
from write_video import writeVideo
from accuracy_check.file_io import writeShmToJsonFile, gTruthDetectAndTrack

# 따로 runInfo_accuracy.py를 만들까?
fillBboxWithGTruth = True
gTruthQuery = "P7"


input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

# 프레임 단위 정보 저장 배열의 크기
FRAMES_SIZE = end_frame - start_frame + 1
# 사람 단위 정보 저장 배열의 크기
PEOPLE_SIZE = FRAMES_SIZE * 5

# 처리할 프레임 총 개수
FRAME_NUM = end_frame - start_frame + 1
# 프레임 인원 수의 상한선
MAX_PEOPLE_NUM = 8


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')   
    logger = make_logger(runInfo.logfile_name, 'root')
    if os.path.exists(runInfo.input_video_path) == False:
        logger.critical("[IO Error] Input video path: {} is not exists".format(runInfo.input_video_path))
        exit(-1)
    if os.path.exists(runInfo.query_image_path) == False:
        logger.critical("[IO Error] Query image directory path: {} is not exists".format(runInfo.query_image_path))
        exit(-1)
        
    if fillBboxWithGTruth:
        detectProc = gTruthDetectAndTrack
    else:
        detectProc = detectAndTrack

    startTime = time.time()
    
    shm = ShmSerialManager(processNum=5, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
    
    maskProc = Process(target=runMaskDetection, args=(shm, 3, os.getpid()))
    maskProc.start()


    distanceProc = Process(target=checkDistance, args=(shm, 2, maskProc.pid))
    distanceProc.start()
    
    reidProc = Process(target=runPersonReid, args=(shm, 1, distanceProc.pid, runInfo.reid_model, runInfo.reidGPU)) # (shm, procNo, nxtPid, reidmodel, gpuNo), reid model:'fake'/'topdb'/'la'
    reidProc.start()
        
    detectTrackProc = Process(target=detectProc, args=(shm, 0, reidProc.pid))
    detectTrackProc.start()

    writeVideo(shm, 4, detectTrackProc.pid)
    
    logger.info("Running time: {}".format(time.time() - startTime))

    writeShmToJsonFile(shm.data, start_frame, end_frame, input_video_path)