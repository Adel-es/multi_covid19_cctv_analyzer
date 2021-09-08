import json 
from typing import List
from collections import OrderedDict
from enum import IntEnum
import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.types import MaskToken

class DangerLevel(IntEnum) : 
    BothUnmasked                        = 15 
    TargetUnmasked_ContactorMasked      = 10
    TargetUnmaksed_ContactorUnKnown     = 10
    TargetUnKnown_ContactorUnmasked     = 5
    BothUnknown                         = 5
    TargetUnknown_ContactorMasked       = 5
    TargetMasked_ContactorUnmasked      = 1  
    TargetMasked_ContactorUnknown       = 1
    BothMasked                          = 0 

    
def maskToken_to_dangerLevel(target : MaskToken, contactor : MaskToken) -> DangerLevel :
    if target == MaskToken.NotMasked : 
        if contactor == MaskToken.NotMasked : 
            return DangerLevel.BothUnmasked 
        elif contactor == MaskToken.Masked : 
            return DangerLevel.TargetUnmasked_ContactorMasked 
        else : 
            return DangerLevel.TargetUnmaksed_ContactorUnKnown 
    
    elif target == MaskToken.FaceNotFound or target == MaskToken.UnKnown : 
        if contactor == MaskToken.NotMasked : 
            return DangerLevel.TargetUnKnown_ContactorUnmasked
        elif contactor == MaskToken.Masked :
            return DangerLevel.TargetUnknown_ContactorMasked
        else : 
            return DangerLevel.BothUnknown 
    
    elif target == MaskToken.Masked :
        if contactor == MaskToken.NotMasked :
            return DangerLevel.TargetMasked_ContactorUnmasked
        elif contactor == MaskToken.Masked : 
            return DangerLevel.BothMasked
        else : 
            return DangerLevel.TargetMasked_ContactorUnknown


class Contactor : 
    threshold = 75 # our video data is 25fps => so, more than 3sec contact is considered
    
    def __init__(self, current_frame_num : int) : 
        self.continued_contact      : int           = 0
        self.last_contact_frame     : int           = current_frame_num 
        self.is_contactor           : bool          = False
        self.most_danger            : DangerLevel   = DangerLevel.BothMasked
        self.start_contact_frame    : int           = current_frame_num
        self.end_contact_frame      : int           = current_frame_num 
        self.bbox                   : List[int]     = []
        
    def update(self, current_frame_num : int, target_mask : MaskToken, contactor_mask : MaskToken, bbox : List[int]) : 
        if self.last_contact_frame + 1 == current_frame_num : 
            self.continued_contact = self.continued_contact + 1 
        else : 
            self.continued_contact = 1
        
        if self.continued_contact > Contactor.threshold :
            self.is_contactor = True 
            self.most_danger = max(self.most_danger, maskToken_to_dangerLevel(target_mask, contactor_mask))
            self.end_contact_frame = current_frame_num
            self.bbox = bbox
            
        self.last_contact_frame = current_frame_num 


class ResultManager : 
    def __init__(self) :         
        self.target_in              : int           = 0 
        self.target_out             : int           = 0 
        self.continued              : bool          = False 
        
        self.result_dict            = OrderedDict() 
        self.result_dict["target"]  = [] 
        
        self.contactor_dict         = {} #tid-contactor object pair 
        
    def update_targetinfo(self, frame_num : int, is_target : bool) :
        """ This function must be called when each frame is processed. 
            to check target appearence time and exit time. 
        
        Args : 
            frame_num : frame number
            is_target : whether target is appeard in frame(True) or not(False)
        """ 
        
        if is_target == True  and self.continued == False : 
            self.target_in = frame_num 
            self.continued = True 
        
        if is_target == False and self.continued == True : 
            self.target_out = frame_num 
            self.continued = False 
            self.result_dict["target"].append(
                {"in" : self.target_in, 
                 "out" : self.target_out})
            
    def transfer_contactor_dict_format(self) : 
        contactor_list = [] 
        for key, value in self.contactor_dict.items() : 
            if value.is_contactor == False : 
                continue 
            contactor_list.append({
                "tid" : key, 
                "danger_level" : value.most_danger, 
                "bbox" : value.bbox, 
                "start_time" : value.start_contact_frame, 
                "end_time" : value.end_contact_frame
                })
        return contactor_list 
    
    def update_contactorinfo(self, 
                             frame_num : int, 
                             tid : int, 
                             target_mask : MaskToken, 
                             contactor_mask : MaskToken,
                             bbox : List[int]) : 
        if tid in self.contactor_dict.keys() : 
            self.contactor_dict[tid].update(frame_num, target_mask, contactor_mask, bbox) 
        else : 
            self.contactor_dict[tid] = Contactor(frame_num) 
    
    
    def write_jsonfile(self, filename : str) : 
        """ write result_dict as json format 
        
        Args : 
            filename : filename output file 
        """
        formatted_contactor = dict() 
        formatted_contactor["contactor"] = self.transfer_contactor_dict_format(); 

        with open(filename, 'w', encoding='utf-8') as write_file : 
            json.dump(self.result_dict, write_file, indent="\t")
            json.dump(formatted_contactor, write_file, indent="\t")
            