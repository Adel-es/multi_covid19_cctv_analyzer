import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.types import ShmSerialManager
from utils.logger import make_logger 
from configs import runInfo
from timeit import time

from deep_sort_yolo4.person_detection import detectAndTrack
from distance import checkDistance, makeAllIsCloseTrue
from personReid.personReid import runPersonReid
from maskdetect.maskProcess import runMaskDetection
from write_video import writeVideo
from accuracy_check.file_io import *
from accuracy_check.check_bbox import getBboxAccuracyAndMapping
from accuracy_check.check_reid import getReidAccuracy
from accuracy_check.check_system import getSystemAccuracy
from accuracy_check.mask_accuracy import copy_from_gtruth, get_f1_score, print_confusion_matrix, print_precisions, print_recalls, update_confusion_matrix
from personReid.personReid import fakeReid3

from utils.types import MaskToken

TEST_BBOX = False
TEST_REID = False
TEST_MASK = True
TEST_SYSTEM = False
QUERY_GROUND_TRUTH = "P8"
WRITE_VIDEO = False

print(TEST_MASK)
print(runInfo.input_video_path)

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
    
    gTruth_file_path = getGTruthFilePath(runInfo.input_video_path) 
    if os.path.exists(gTruth_file_path) == False : 
        logger.critical("[IOError] gTruth_file_path : {} is not exists.".format(gTruth_file_path))
        exit(-1)
    
    startTime = time.time()
    if TEST_BBOX:
        shm = ShmSerialManager(processNum=2, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)        
        detectAndTrack(shm, 0, os.getpid())
        writeVideoProcessOrder = 1
    
    elif TEST_REID:
        shm = ShmSerialManager(processNum=3, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
        gTruthDetectAndTrack(shm, 0, os.getpid())   # gtruth => shm
        runPersonReid(shm, 1, os.getpid(), runInfo.reid_model, runInfo.reidGPU)
        writeVideoProcessOrder = 2
         
    elif TEST_MASK:
        shm = ShmSerialManager(processNum=4, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
        copy_from_gtruth(shm, 0, os.getpid())   # gtruth => shm
        fakeReid3(shm, 1, os.getpid())          # fakeReid3 random pick one if there is a person 
        runMaskDetection(shm, 2, os.getpid())   # shm 바탕 mask detection 
        writeVideoProcessOrder = 3
        
    elif TEST_SYSTEM:
        shm = ShmSerialManager(processNum=5, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
        detectAndTrack(shm, 0, os.getpid())
        runPersonReid(shm, 1, os.getpid(), runInfo.reid_model, runInfo.reidGPU)  # (shm, procNo, nxtPid, reidmodel, gpuNo), reid model:'fake'/'topdb'/'la'
        makeAllIsCloseTrue(shm, 2, os.getpid())
        runMaskDetection(shm, 3, os.getpid())
        writeVideoProcessOrder = 4
    else:
        logger.critical("[VariableSettingError] One of TEST_BBOX, TEST_REID, TEST_MASK, or TEST_SYSTEM must be True.")
        exit(-1)
    
    if WRITE_VIDEO:
        writeVideo(shm, writeVideoProcessOrder, os.getpid())
        
    logger.info("Running time: {}".format(time.time() - startTime))

    # Write system result(shm) to json file
    writeShmToJsonFile(shm.data, start_frame, end_frame, input_video_path, QUERY_GROUND_TRUTH)
    
    # Get shm and gTruth from each file
    shm_file_path = getShmFilePath(input_video_path)
    shm = convertShmFileToJsonObject(shm_file_path)
    gTruth_file_path = getGTruthFilePath(input_video_path) 
    gTruth = convertGTruthFileToJsonObject(gTruth_file_path)

    # Check accuracy and print
    if TEST_BBOX:
        getBboxAccuracyAndMapping(shm, gTruth)
    elif TEST_REID:
        getReidAccuracy(shm, gTruth)
    elif TEST_MASK:
        for i in range(1, 9) : 
            update_confusion_matrix(shm, gTruth, "P{}".format(i), [i], runInfo.start_frame, runInfo.end_frame)
        print_confusion_matrix()
        print_precisions() 
        print_recalls() 
        print("F1-score : {}".format(get_f1_score()))
    elif TEST_SYSTEM:
        getSystemAccuracy(shm, gTruth)