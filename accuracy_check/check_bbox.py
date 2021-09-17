import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from configs import runInfo
import file_io
import numpy as np
from collections import Counter
from AP_utils import calculateAveragePrecision, ElevenPointInterpolatedAP
from utils.logger import make_logger
import logging 

def boundingBoxes(shm, gTruth):
    detections, groundtruths = [], []
    
    start_frame = shm['start_frame']
    end_frame = shm['end_frame']
    num_of_frames = end_frame - start_frame + 1
    
    if len(shm['people']) != num_of_frames:
        sys.exit("accuracy_detection.py line 45: The number of frames doesn't match.")

    # Fill detections
    for fIdx in range(num_of_frames):
        for pIdx in range(len(shm['people'][fIdx])):
            confidence = shm['people'][fIdx][pIdx]['bbox'][4]
            bbox = shm['people'][fIdx][pIdx]['bbox'][:4]
            detections.append({'fIdx': fIdx, 
                               'pIdx': pIdx, 
                               'confidence': confidence, 
                               'bbox': bbox})
    
    # Fill groundtruths
    gTruthConfidence = 1.0
    for pKey in gTruth:
        # First and second frames are skipped since there are no detections (due to tracking)
        for fIdx in range(2, num_of_frames):
            frameNum = start_frame + fIdx
            if type(gTruth[pKey][frameNum]) == dict:
                bbox = gTruth[pKey][frameNum]['Position']
                groundtruths.append({'fIdx': fIdx, 
                                     'pKey': pKey, 
                                     'confidence': gTruthConfidence, 
                                     'bbox': bbox})
    return detections, groundtruths

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def iou(boxA, boxB):
    # if boxes dont intersect
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    
    # intersection over union
    result = interArea / union
    assert result >= 0
    return result

def AP(detections, groundtruths, shmToGTruthMapping, IOUThreshold = 0.3, Interpolated11Points = True):
    
    npos = len(groundtruths)

    detections = sorted(detections, key = lambda detection : detection['confidence'], reverse=True)

    TP = np.zeros(len(detections), dtype=np.int)
    FP = np.zeros(len(detections), dtype=np.int)

    numOfBboxPerFrame = Counter(groundtruth['fIdx'] for groundtruth in groundtruths)

    isAssigned = dict()
    for frameIdx, bboxNum in numOfBboxPerFrame.items():
        isAssigned[frameIdx] = np.zeros(bboxNum, dtype=np.bool)

    for d in range(len(detections)):

        gt = [gt for gt in groundtruths if gt['fIdx'] == detections[d]['fIdx']]

        iouMax = 0

        for j in range(len(gt)):
            iou1 = iou(detections[d]['bbox'], gt[j]['bbox'])
            if iou1 > iouMax:
                iouMax = iou1
                jmax = j

        if iouMax >= IOUThreshold:
            fIdx = detections[d]['fIdx']
            if isAssigned[fIdx][jmax] == False:
                TP[d] = 1
                isAssigned[fIdx][jmax] = True
                pIdx = detections[d]['pIdx']
                shmToGTruthMapping[fIdx][pIdx]['pKey'] = gt[jmax]['pKey']
                shmToGTruthMapping[fIdx][pIdx]['bboxTP'] = True
            else:
                FP[d] = 1
        else:
            FP[d] = 1

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

def getBboxAccuracyAndMapping(shm, gTruth, shmToGTruthMapping=[], makeLog=True):
    logger = logging.getLogger('root')

    detections, groundtruths = boundingBoxes(shm, gTruth)
    
    if len(detections) == 0 and len(groundtruths) == 0:
        if makeLog:
            logger.info("========== Bbox ===========")
            logger.info("groundTruths: 0, detections: 0")
            logger.info("Cannot calculate accuracy")
            logger.info("===========================")
    elif len(detections) == 0 or len(groundtruths) == 0:
        if makeLog:
            logger.info("========== Bbox ===========")
            logger.info("groundTruths: {}, detections: {}".format(len(groundtruths), len(detections)))
            logger.info("total TP: 0, total FP: {}".format(len(detections)))
            logger.info("AP: 0.0")
            logger.info("===========================")
    else: # if len(detections) > 0 and len(groundtruths) > 0    
        if len(shmToGTruthMapping) == 0:
            for aFrame in shm["people"]:
                shmToGTruthMapping.append([{"pKey": "", "bboxTP": False, "reidTP": False, "maskTP": False} for i in range(len(aFrame))])

        result = AP(detections, groundtruths, shmToGTruthMapping)

        if makeLog:
            logger.info("========== Bbox ===========")
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
    
    getBboxAccuracyAndMapping(shm, gTruth)