import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch 
import file_io
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
from accuracy_check.file_io import writeShmToJsonFile, gTruthDetectAndTrack, getGTruthFilePath
from accuracy_check.mask_accuracy import copy_from_gtruth, get_f1_score, print_confusion_matrix, print_precisions, print_recalls, update_confusion_matrix
from personReid.personReid import fakeReid3

import maskdetect.faceDetector.faceDetector as face
import maskdetect.maskClassifier.maskClassifier as mask

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
    # torch.multiprocessing.set_start_method('fork')   
    runInfo.parallel_processing = False  
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
    
    startTime = time.time()
    
    shm = ShmSerialManager(processNum=4, framesSize=FRAME_NUM, peopleSize=FRAME_NUM*MAX_PEOPLE_NUM)
    
    copy_from_gtruth(shm, 0, os.getpid())   # gtruth => shm
    fakeReid3(shm, 1, os.getpid())          # fakeReid3 random pick one if there is a person 
    runMaskDetection(shm, 2, os.getpid())   # shm 바탕 mask detection 
    writeVideo(shm, 3, os.getpid())         #write video 
    logger.info("Running time: {}".format(time.time() - startTime))

    # =================================================================================================
    # run mask accuracy test 
    video_paths = runInfo.input_video_path.split('/')
    videoName = video_paths[-1]
    
    writeShmToJsonFile(shm.data, start_frame, end_frame, input_video_path, "MaskTest")
    shm_file_path = file_io.getShmFilePath(videoName) 
    shm = file_io.convertShmFileToJsonObject(shm_file_path)
    
    # Create gTruth_file_path based on runInfo.input_video_path
    gTruth_file_path = file_io.getGTruthFilePath(videoName) 
    gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
    
    for i in range(1, 9) : 
        update_confusion_matrix(shm, gTruth, "P{}".format(i), [i], runInfo.start_frame, runInfo.end_frame)
    print_confusion_matrix()
    print_precisions() 
    print_recalls() 
    print("F1-score : {}".format(get_f1_score()))
    # update_confusion_matrix(shm, gTruth, "P1", [1], runInfo.start_frame, runInfo.end_frame)
    # update_confusion_matrix(shm, gTruth, "P2", [2], runInfo.start_frame, runInfo.end_frame)