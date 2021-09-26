import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from configs import runInfo
import file_io
from check_bbox import getBboxAccuracyAndMapping
from check_reid import getReidAccuracy
from mask_accuracy import get_maskAccuracy
from utils.logger import make_logger
import logging

def getSystemAccuracy(shm, gTruth):
    logger = logging.getLogger('root') 
    
    queryKey = shm['gTruth_query']
    start_frame = shm['start_frame']
    end_frame = shm['end_frame']
    num_of_frames = end_frame - start_frame + 1

    if len(shm['people']) != num_of_frames:
        logger.critical("The number of frames doesn't match.")
        sys.exit(-1)
    
    # Remove data from gTruth except for query
    for pKey in gTruth:
        if pKey == queryKey:
            continue
        for fIdx in range(2, num_of_frames):  # First and second frames are skipped since there are no detections in shm (for tracking)
            frameNum = start_frame + fIdx
            if type(gTruth[pKey][frameNum]) != list: # If labeled
                gTruth[pKey][frameNum] = []     # Remove
    
    # Remove data from shm except for inferred query
    # and if the frame has no ground truth query
    k = 0
    for fIdx in range(2, num_of_frames):
        
        inferredQueryPIdx = shm['frames'][fIdx]['reid']
        # If there are no inferred query
        if inferredQueryPIdx == -1:
            shm['people'][fIdx] = []
            continue
        
        frameNum = start_frame + fIdx
        # If there are no ground truth query
        if type(gTruth[queryKey][frameNum]) != dict:
            k = k+1
            print("minus: {}".format(k))
            shm['frames'][fIdx]['reid'] = -1
            shm['people'][fIdx] = []
            continue
        
        shm['people'][fIdx] = [ shm['people'][fIdx][inferredQueryPIdx] ]

    # Count groundTruthsNum
    groundTruthsNum = 0
    for i in range(2, num_of_frames):
        frameNum = start_frame + i
        if type(gTruth[queryKey][frameNum]) == dict:
            groundTruthsNum += 1

    # Count detectionsNum
    detectionsNum = 0
    for i in range(num_of_frames):
        if shm['frames'][i]['reid'] != -1:
            detectionsNum += 1

    if groundTruthsNum == 0 and detectionsNum == 0:
        logger.info("========== System ===========")
        logger.info("groundTruths: 0, detections: 0")
        logger.info("Cannot calculate accuracy")
        logger.info("=============================")
        return
    
    # Count TP, FP
    TP, FP = 0, 0

    shmToGTruthMapping = []
    for aFrame in shm["people"]:
        shmToGTruthMapping.append([{"pKey": "", "bboxTP": False, "reidTP": False, "maskTP": False} for i in range(len(aFrame))])

    getBboxAccuracyAndMapping(shm, gTruth, shmToGTruthMapping, makeLog=False)
    getReidAccuracy(shm, gTruth, shmToGTruthMapping, makeLog=True)
    get_maskAccuracy(gTruth, shm, shmToGTruthMapping)
    # logger.info(shmToGTruthMapping)

    for aFrame in shmToGTruthMapping:
        for person in aFrame:
            if person['bboxTP'] and person['reidTP'] and person['maskTP']:
                TP += 1
            else:
                FP += 1

    # Calculate precision, recall, f1-score
    
    # Input                         Output
    # detectionsNum groundTruthsNum precision   recall  F-score
    # 0             0               X           X       X
    # n             0               0           0       0
    # 0             n               0           0       0
    
    precision, recall, f_score = 0.0, 0.0, 0.0
    
    if detectionsNum != 0:
        precision = TP / detectionsNum
    
    if groundTruthsNum != 0:
        recall = TP / groundTruthsNum

    if precision + recall > 0.0:
        f_score = (2 * precision * recall) / (precision + recall)
    
    # Logging
    logger.info("========== System ===========")
    logger.info("groundTruths: {}, detections: {}".format(groundTruthsNum, detectionsNum))
    logger.info("TP: {}, FP: {}".format(TP, FP))
    logger.info("precision: {}".format(precision))
    logger.info("recall: {}".format(recall))
    logger.info("F1-score: {}".format(f_score))
    logger.info("=============================")


if __name__ == '__main__':
    logger = make_logger(runInfo.logfile_name, 'root')

    # Prepare shm and gTruth
    videoName = "08_14_2020_1_1.mp4"
    # Create shm_file_path based on runInfo.input_video_path
    shm_file_path = file_io.getShmFilePath(runInfo.input_video_path) 
    shm = file_io.convertShmFileToJsonObject(shm_file_path)
    # Create gTruth_file_path based on runInfo.input_video_path
    gTruth_file_path = file_io.getGTruthFilePath(runInfo.input_video_path) 
    gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
    
    getSystemAccuracy(shm, gTruth)