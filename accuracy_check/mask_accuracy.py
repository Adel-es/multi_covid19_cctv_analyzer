import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.types import MaskToken 
import file_io 
from typing import List

confusion_matrix_index = {
    MaskToken.FaceNotFound : 0, 
    MaskToken.Masked : 1, 
    MaskToken.NotMasked : 2 
}
confusion_matrix = [
    [0, 0, 0], 
    [0, 0, 0], 
    [0, 0, 0]
]

def string_to_maskToken(mask_string : str) -> MaskToken : 
    if mask_string == "mask" :
        return MaskToken.Masked
    elif mask_string == "unmask" : 
        return MaskToken.NotMasked
    else : 
        return MaskToken.FaceNotFound  

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
            
if __name__ == "__main__" : 
    videoName = "08_14_2020_1_1.mp4"
    # Create shm_file_path based on runInfo.input_video_path
    
    shm_file_path = file_io.getShmFilePath(videoName) 
    shm = file_io.convertShmFileToJsonObject(shm_file_path)
    
    # Create gTruth_file_path based on runInfo.input_video_path
    gTruth_file_path = file_io.getGTruthFilePath(videoName) 
    gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
    
    update_confusion_matrix(shm, gTruth, "P2", [1], 700, 710)
    print(confusion_matrix)