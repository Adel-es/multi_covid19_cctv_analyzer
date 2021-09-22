import numpy as np
from collections import Counter

import sys
import file_io

# Prepare shm and gTruth
videoName = "08_14_2020_1_1.mp4"
# Create shm_file_path based on runInfo.input_video_path
shm_file_path = file_io.getShmFilePath(videoName) 
shm = file_io.convertShmFileToJsonObject(shm_file_path)
# Create gTruth_file_path based on runInfo.input_video_path
gTruth_file_path = file_io.getGTruthFilePath(videoName) 
gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)

def boundingBoxes(shm, gTruth):
    detections, groundtruths, classes = [], [], []
    
    start_frame = shm['start_frame']
    end_frame = shm['end_frame']
    num_of_frames = end_frame - start_frame + 1
    
    if len(shm['people']) != num_of_frames:
        sys.exit("accuracy_detection.py line 45: The number of frames doesn't match.")
    
    # Fill classes
    personClass = 0.0
    classes = [personClass]  # person is the only class

    # Fill detections
    for i in range(num_of_frames):
        frameNum = start_frame + i
        for person in shm['people'][i]:
            confidence = person['bbox'][4]
            bbox = person['bbox'][:4]
            detections.append([str(frameNum), personClass, confidence, bbox])
    
    # Fill groundtruths
    gTruthConfidence = 1.0
    for key in gTruth:
        # First and second frames are skipped since there are no detections (due to tracking)
        for i in range(2, num_of_frames):
            frameNum = start_frame + i
            if type(gTruth[key][frameNum]) == dict:
                bbox = gTruth[key][frameNum]['Position']
                groundtruths.append([str(frameNum), personClass, gTruthConfidence, bbox])
    
    return detections, groundtruths, classes

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

def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

def AP(detections, groundtruths, IOUThreshold = 0.3):
    
    npos = len(groundtruths)

    detections = sorted(detections, key = lambda conf : conf[2], reverse=True)

    TP = np.zeros(len(detections))
    FP = np.zeros(len(detections))

    det = Counter(cc[0] for cc in groundtruths)

    # 각 이미지별 ground truth box의 수
    # {99 : 2, 380 : 4, ....}
    # {99 : [0, 0], 380 : [0, 0, 0, 0], ...}
    for key, val in det.items():
        det[key] = np.zeros(val)

    for d in range(len(detections)):

        gt = [gt for gt in groundtruths if gt[0] == detections[d][0]]

        iouMax = 0

        for j in range(len(gt)):
            iou1 = iou(detections[d][3], gt[j][3])
            if iou1 > iouMax:
                iouMax = iou1
                jmax = j

        if iouMax >= IOUThreshold:
            if det[detections[d][0]][jmax] == 0:
                TP[d] = 1
                det[detections[d][0]][jmax] = 1
            else:
                FP[d] = 1
                print(detections[d][3])
        else:
            FP[d] = 1
            print(detections[d][3])

    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / npos
    prec = np.divide(acc_TP, (acc_FP + acc_TP))

    [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)

    result = {
        'precision' : prec,
        'recall' : rec,
        'AP' : ap,
        'interpolated precision' : mpre,
        'interpolated recall' : mrec,
        'total positives' : npos,
        'total TP' : np.sum(TP),
        'total FP' : np.sum(FP)
    }

    return result


detections, groundtruths, classes = boundingBoxes(shm, gTruth)

print("detections len: {}".format(len(detections)))
print("groundtruths len: {}\n".format(len(groundtruths)))

result = AP(detections, groundtruths)

print("===result===")
for resultKey in result:
    print("{}: {}".format(resultKey, result[resultKey]))
print("============")