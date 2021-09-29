import json 
from typing import List
from collections import OrderedDict
from enum import IntEnum
import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.types import MaskToken
import logging

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
    InitialValue                        = -1

    
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
    
    else : # target == MaskToken.Masked :
        if contactor == MaskToken.NotMasked :
            return DangerLevel.TargetMasked_ContactorUnmasked
        elif contactor == MaskToken.Masked : 
            return DangerLevel.BothMasked
        else : 
            return DangerLevel.TargetMasked_ContactorUnknown

def maskToken_to_str(mask : MaskToken) -> str : 
    if mask == MaskToken.Masked : 
        return "masked"
    elif mask == MaskToken.NotMasked : 
        return "notMasked"
    elif mask == MaskToken.FaceNotFound : 
        return "faceNotFound"
    elif mask == MaskToken.UnKnown : 
        return "UnKnown"

class Contactor : 
    threshold = 75 / 4 # our video data is 25fps => so, more than 3sec contact is considered
    
    def __init__(self, current_frame_num : int) : 
        self.continued_contact      : int           = 0
        self.last_contact_frame     : int           = current_frame_num 
        self.is_contactor           : bool          = False
        self.most_danger            : DangerLevel   = DangerLevel.InitialValue
        self.target_mask            : MaskToken     = MaskToken.UnKnown 
        self.contactor_mask         : MaskToken     = MaskToken.UnKnown 
        self.capture_frame          : int           = -1
        self.start_contact_frame    : int           = current_frame_num
        self.end_contact_frame      : int           = current_frame_num 
        
        
    def update(self, current_frame_num : int, target_mask : MaskToken, contactor_mask : MaskToken) -> bool: 
        haveto_save_image = False 
        
        if self.last_contact_frame + 4 >= current_frame_num : 
            self.continued_contact = self.continued_contact + 1 
        else : 
            self.continued_contact = 1
        
        if self.continued_contact > Contactor.threshold :
            self.is_contactor = True 
            self.end_contact_frame = current_frame_num
            new_danger_level = maskToken_to_dangerLevel(target_mask, contactor_mask) 
            if int(self.most_danger) < int(new_danger_level) :
                self.most_danger = new_danger_level 
                self.capture_frame = current_frame_num
                self.target_mask = maskToken_to_str(target_mask) 
                self.contactor_mask = maskToken_to_str(contactor_mask)  
                haveto_save_image = True 
        self.last_contact_frame = current_frame_num 
        return haveto_save_image 


class ResultManager : 
    def __init__(self) :
        self.logger                 = logging.getLogger('root')         
        self.target_in              : int           = 0 
        self.target_out             : int           = 0 
        self.continued              : bool          = False 
        
        self.result_dict            = OrderedDict() 
        self.result_dict["target"]  = [] 
        
        self.contactor_dict         = {} #tid-contactor object pair 
        
    def update_targetinfo(self, frame_num : int, is_target : bool, is_lastframe : bool) :
        """ This function must be called when each frame is processed. 
            to check target appearence time and exit time. 
        
        Args : 
            frame_num : frame number
            is_target : whether target is appeard in frame(True) or not(False)
        """ 
        # print("[ResManager] frame : {} , target is here".format(frame_num))
        if is_target == True  and self.continued == False : 
            self.target_in = frame_num 
            self.continued = True 
            self.logger.debug("resultmanager - target appeared at {}".format(frame_num))
        
        if is_target == False and self.continued == True : 
            self.target_out = frame_num 
            self.continued = False 
            self.result_dict["target"].append({"in" : self.target_in, "out" : self.target_out})
            self.logger.debug("resultmanager - target disappeared at {}".format(frame_num))
        
        if is_lastframe == True and self.continued == True : 
            self.target_out = frame_num 
            self.continued = False 
            self.result_dict["target"].append({"in" : self.target_in, "out" : self.target_out})
            self.logger.debug("resultmanager - video end at {}, so make sure target is disappeared.".format(frame_num)) 
    
    def transfer_contactor_dict_format(self) : 
        contactor_list = [] 
        for key, value in self.contactor_dict.items() : 
            if value.is_contactor == False : 
                continue 
            contactor_list.append({
                "tid" : key, 
                "danger_level" : value.most_danger,
                "target_mask"  : value.target_mask,
                "contactor_mask" : value.contactor_mask,
                "capture_time" : value.capture_frame,
                "start_time" : value.start_contact_frame, 
                "end_time" : value.end_contact_frame
                })
        return contactor_list 
    
    
    def update_contactorinfo(self, 
                            frame_num : int, 
                            tid : int, 
                            target_mask : MaskToken, 
                            contactor_mask : MaskToken) : 
        
        """ update contactor info and return list to indicate if the image should be saved or not. 
        
        Return : (bool, String) 
            if string, save as image(cut-off image by bbox) as string name. 
            if None, don't save image(just skip it)
        """
        
        result_bool = False 
        if tid in self.contactor_dict.keys() : 
            result_bool = self.contactor_dict[tid].update(frame_num, target_mask, contactor_mask) 
        else : 
            self.contactor_dict[tid] = Contactor(frame_num) 
    
        if result_bool == True : 
            return True, "fr{}_tid{}.jpg".format(frame_num, tid) 
        else : 
            return False, ""
    
    
    def write_jsonfile(self, filename : str, outputVideo : str, startFrame : int, endFrame : int ) : 
        """ write result_dict as json format 
        
        Args : 
            filename : filename output file 
        """
        self.result_dict["video_name"]  = outputVideo
        self.result_dict["start_frame"] = startFrame
        self.result_dict["end_frame"]   = endFrame
        self.result_dict["contactor"]   = self.transfer_contactor_dict_format()

        with open(filename, 'w', encoding='utf-8') as write_file : 
            json.dump(self.result_dict, write_file, indent="\t")
            