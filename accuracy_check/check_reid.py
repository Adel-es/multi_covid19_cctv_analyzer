import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from configs import runInfo
import sys
import file_io
import numpy as np
from collections import Counter
from check_bbox import getBboxAccuracyAndMapping
from AP_utils import calculateAveragePrecision, ElevenPointInterpolatedAP
from utils.logger import make_logger
import logging 

def reidResult(shm, gTruth):
    logger = logging.getLogger('root')
    detections, groundtruths = [], []
    
    start_frame = shm['start_frame']
    end_frame = shm['end_frame']
    num_of_frames = end_frame - start_frame + 1
    
    if len(shm['people']) != num_of_frames:
        logger.critical("The number of frames doesn't match.")
        sys.exit(-1)
    
    # Fill detections
    for fIdx in range(num_of_frames):
        for pIdx in range(len(shm['people'][fIdx])):
            confidence = shm['people'][fIdx][pIdx]['reidConf']
            detections.append({'fIdx': fIdx, 
                            'pIdx': pIdx,
                            'confidence': confidence})
    
    pKeyQuery = shm['gTruth_query']
    # Fill groundtruths
    # First and second frames are skipped since there are no detections (due to tracking)
    for fIdx in range(2, num_of_frames):
        frameNum = start_frame + fIdx
        if type(gTruth[pKeyQuery][frameNum]) == dict:
            groundtruths.append({'fIdx': fIdx})
                
    return detections, groundtruths

def AP(detections, groundtruths, shmToGTruthMapping, pKeyQuery, Interpolated11Points = True):
    
    npos = len(groundtruths)

    detections = sorted(detections, key = lambda detection : detection['confidence'], reverse=True)

    TP = np.zeros(len(detections), dtype=np.int)
    FP = np.zeros(len(detections), dtype=np.int)

    for d in range(len(detections)):
        fIdx = detections[d]['fIdx']
        pIdx = detections[d]['pIdx']
        pKey = shmToGTruthMapping[fIdx][pIdx]['pKey']
        if pKey == pKeyQuery:
            TP[d] = 1
            shmToGTruthMapping[fIdx][pIdx]['reidTP'] = True
            if npos == np.sum(TP):
                break
        else:
            FP[d] = 1

    confList = []
    for d in range(len(detections)):
        fIdx = detections[d]['fIdx']
        pIdx = detections[d]['pIdx']
        pKey = shmToGTruthMapping[fIdx][pIdx]['pKey']
        conf = np.round(detections[d]['confidence'],5)
        confList.append( [TP[d], conf, pKey] )
    print(confList)
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)

    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    if Interpolated11Points:
        [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
    else:
        [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)

    result = {
        'precision' : prec,
        'recall' : rec,
        'AP' : ap,
        'interpolated precision' : mpre,
        'interpolated recall' : mrec,
        'total TP' : np.sum(TP),
        'total FP' : np.sum(FP)
    }

    return result


def getReidAccuracy(shm, gTruth, shmToGTruthMapping=[], makeLog=True):
    logger = logging.getLogger('root')
    
    detections, groundtruths = reidResult(shm, gTruth)

    if len(detections) == 0 and len(groundtruths) == 0:
        if makeLog:
            logger.info("========== Reid ===========")
            logger.info("groundTruths: 0, detections: 0")
            logger.info("Cannot calculate accuracy")
            logger.info("===========================")
    elif len(detections) == 0 or len(groundtruths) == 0:
        if makeLog:
            logger.info("========== Reid ===========")
            logger.info("groundTruths: {}, detections: {}".format(len(groundtruths), len(detections)))
            logger.info("total TP: 0, total FP: {}".format(len(detections)))
            logger.info("AP: 0.0")
            logger.info("===========================")
    else: # if len(detections) > 0 and len(groundtruths) > 0
        if len(shmToGTruthMapping) == 0:
            getBboxAccuracyAndMapping(shm, gTruth, shmToGTruthMapping, makeLog=False)
        
        result = AP(detections, groundtruths, shmToGTruthMapping, shm['gTruth_query'])
        
        if makeLog:
            logger.info("========== Reid ===========")
            # Long log
            # for resultKey in result:
            #     logger.info("{}: {}".format(resultKey, result[resultKey]))
            # Short log
            logger.info("groundTruths: {}, detections: {}".format(len(groundtruths), len(detections)))
            logger.info("total TP: {}, total FP: {}".format(result['total TP'], result['total FP']))
            logger.info("AP: {}".format(result['AP']))
            logger.info("===========================")
    
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

    getReidAccuracy(shm, gTruth)
