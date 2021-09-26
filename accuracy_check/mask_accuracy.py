import math
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.types import MaskToken, BBox 
import file_io 
import configs.runInfo as runInfo 
import logging
from typing import List

confusion_matrix_index = {
    MaskToken.FaceNotFound : 0, 
    MaskToken.Masked : 1, 
    MaskToken.NotMasked : 2 
}
confusion_matrix = [
    # => predicition 
    # | groundTruth 
    [0, 0, 0], 
    [0, 0, 0], 
    [0, 0, 0]
]


new_tid = 9
tid_to_currentTid = {
        1 : [1], 
        2 : [2], 
        3 : [3],
        4 : [4],
        5 : [5],
        6 : [6],
        7 : [7],
        8 : [8]
}

def copy_from_gtruth (shm, processOrder, nextPid) : 
    global new_tid
    myPid = 'mask_accuracy.copy_from_gtruth'
    
    shm.init_process(processOrder, myPid, nextPid)
    logger = logging.getLogger('root') 
    
    # Create gTruth_file_path based on runInfo.input_video_path
    gTruth_file_path = file_io.getGTruthFilePath(runInfo.input_video_path) 
    gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
    
    FRAME_NUM = runInfo.end_frame - runInfo.start_frame + 1
    
    for fIdx in range(FRAME_NUM):
        logger.debug("frame {}".format(fIdx))
        frameNumber = fIdx + runInfo.start_frame 
        print("frameNumber : {}".format(frameNumber))
        tids = []
        bboxes = []
        confidences = []
        for pKey in gTruth:
            person = gTruth[pKey][frameNumber]
            if type(person) == dict:
                if frameNumber != runInfo.start_frame and type(gTruth[pKey][frameNumber - 1]) != dict :   
                    # tids.append(new_tid)
                    new_tid = new_tid + 1 
                    tid_to_currentTid[int(pKey[-1])].append(new_tid)
                tids.append(tid_to_currentTid[int(pKey[-1])][-1]) # 수정
                bboxes.append(person['Position'])
                confidences.append(1.0)
        
        peopleNum = len(tids)
        frameIdx, personIdx = shm.get_ready_to_write(peopleNum)
        for i in range(peopleNum):
            # Write at people
            shm.data.people[ personIdx[i] ].bbox = BBox(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], confidences[i])
            shm.data.people[ personIdx[i] ].tid = tids[i]
            shm.data.people[ personIdx[i] ].isClose = True 
            
        shm.finish_a_frame()
    shm.finish_process()


def string_to_maskToken(mask_string : str) -> MaskToken : 
    if mask_string == "mask" :
        return MaskToken.Masked
    elif mask_string == "unmask" : 
        return MaskToken.NotMasked
    else : 
        return MaskToken.FaceNotFound  

def get_precision() : 
    precisions = [] 
    for i in range(0, len(confusion_matrix_index)) : 
        positive_per_class = 0 
        TF_per_class = 0 
        for j in range(0, len(confusion_matrix_index)) : 
            if i == j : 
                TF_per_class += confusion_matrix[j][i]
            positive_per_class += confusion_matrix[j][i]
        precision = 0 
        if positive_per_class != 0 : 
            precision = TF_per_class / float(positive_per_class)
        precisions.append(precision)
    
    sum_precision = 0 
    average_percisions = 0 
    sum_count = 0 
    for p in precisions : 
        if p != 0 : 
            sum_count += 1 
        sum_precision += p  
    if sum_count != 0 : 
        average_percisions = sum_precision / float(sum_count)
    return precisions, average_percisions 

def get_recall() : 
    recalls = [] 
    for i in range(0, len(confusion_matrix_index)) : 
        true_per_class = 0 
        TF_per_class = 0 
        for j in range(0, len(confusion_matrix_index)) : 
            if i == j : 
                TF_per_class += confusion_matrix[i][j] 
            true_per_class += confusion_matrix[i][j]
        recall = 0 
        if true_per_class != 0 : 
            recall = TF_per_class / float(true_per_class)
        recalls.append(recall)
    
    sum_recalls = 0 
    sum_count = 0
    average_recall = 0
    for p in recalls :
        if p != 0 : 
            sum_count += 1  
        sum_recalls += p 
    if sum_count != 0 :  
        average_recall = sum_recalls / float(sum_count) 
    return recalls, average_recall

def print_precisions() : 
    label = [MaskToken.FaceNotFound, MaskToken.Masked, MaskToken.NotMasked]
    pres, avg = get_precision()  
    print("============ PRECISION =================")
    for index, l in enumerate(label) : 
        print(l, end ='')
        print(" : {}".format(pres[index]))
    print("average precision : {}".format(avg)) 
    print("=========================================")


def print_recalls() : 
    label = [MaskToken.FaceNotFound, MaskToken.Masked, MaskToken.NotMasked]
    recalls, avg = get_recall()  
    print("============ RECALLS =================")
    for index, l in enumerate(label) : 
        print(l, end ='')
        print(" : {}".format(recalls[index]))
    print("average recall : {}".format(avg)) 
    print("=========================================")


def print_confusion_matrix() : 
    label = [MaskToken.FaceNotFound, MaskToken.Masked, MaskToken.NotMasked]
    
    print("=================================")
    for l in label : 
        print("\t\t", end = '')
        print(l, end = '')
    print("\n")
    
    for l in label : 
        print(l, end = '')
        for i in confusion_matrix[confusion_matrix_index[l]] : 
            print("\t\t{}".format(i), end = '')
        print("\n") 
    
    print("==================================")
    
def get_f1_score() : 
    _, avg_precision = get_precision() 
    _, avg_recall = get_recall() 
    if(avg_precision + avg_recall) == 0 :
        return 0  
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    return f1_score 

def update_confusion_matrix (shm, gTruth, person_id : str, matching_tid : List[int], start_frame : int, end_frame : int) :  
    ''' update confusion matrix about one person 
    
    Args : 
        person_id       : (ex. "P2") labeled person id 
        matching_tid    : (ex. [1, 6, 12]) matching person tid about labeled pid  
        start_frame     : config.runInfo.start_frame, this is for matching shared memory start, end with gtruth 
        end_frame       : config.runInfo.end_frame  , this is for matching shared memory start, end with gtruth 
    '''
    gtruth_by_person : List = gTruth[person_id]
    
    num_of_frames = end_frame - start_frame + 1 
    for f_idx in range(2, num_of_frames) : 
        current_frame = start_frame + f_idx  
        if type(gtruth_by_person[current_frame]) == list : 
            continue #no one is in this frame .. just keep going ... >_<!! 
        
        gtruth_mask = string_to_maskToken(gtruth_by_person[current_frame]["ismask"]) 
        model_mask = None 
        for p_idx in range(len(shm['people'][f_idx])) : 
            person = shm['people'][f_idx][p_idx]
            if person['tid'] in matching_tid : #matching_tid  
                model_mask = MaskToken(person["isMask"]) 
                break 
        
        if model_mask == None : 
            print("[frame{}] model failed to find person id {}. ".format(current_frame, person_id))
        elif model_mask == MaskToken.NotNear : 
            print("[frame{}] model MaskToken is not near, so just pass it!")
        else : 
            gtruth_index = confusion_matrix_index[gtruth_mask]
            model_index = confusion_matrix_index[model_mask]
            confusion_matrix[gtruth_index][model_index] += 1
    # print_confusion_matrix() 
    
def get_maskAccuracy(gTruth, shm, shmToGTruthMapping) :
    ''' update shmToGTruthMapping.maskTP by comparing shm and gTruth 
    
    Args : 
        gTruth : gTruth list    (get from file_io.convertGTruthFileToJsonObject)
        shm    : shm list       (get from file_io.convertShmFileToJsonObject)
        shmToGTruthMapping :    
    '''
    
    start_frame = shm['start_frame']
    end_frame = shm['end_frame']
    frame_num = end_frame - start_frame + 1
    
    for fIdx in range(frame_num):
        for pIdx in range(len(shm['people'][fIdx])):
            # get masktoken from shm
            person_in_shm = shm['people'][fIdx][pIdx]
            model_mask_token = MaskToken(person_in_shm["isMask"]) 
            
            # get gTruth Person id(like P2) from shmToGTruthMapping 
            gTruth_person_num : str = shmToGTruthMapping[fIdx][pIdx]['pKey'] 
            
            # get masktoken from gTruth 
            if gTruth_person_num != '':
                gTruth_per_person : List = gTruth[gTruth_person_num]
                gTruth_mask_str : str = gTruth_per_person[start_frame + fIdx]["ismask"] #"mask" / "unmask" ... 
                gTruth_mask_token = string_to_maskToken(gTruth_mask_str) 
                
                # compare masktoken from shm and gTruth 
                # update shmToGTruthMapping 
                mask_TP = (model_mask_token == gTruth_mask_token)
                shmToGTruthMapping[fIdx][pIdx]['maskTP'] = mask_TP
            
if __name__ == "__main__" : 
    # Create shm_file_path based on runInfo.input_video_path
    shm_file_path = file_io.getShmFilePath(runInfo.input_video_path) 
    shm = file_io.convertShmFileToJsonObject(shm_file_path)
    
    # Create gTruth_file_path based on runInfo.input_video_path
    gTruth_file_path = file_io.getGTruthFilePath(runInfo.input_video_path) 
    gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
    
    update_confusion_matrix(shm, gTruth, "P2", [1], 700, 710)
    print(confusion_matrix)