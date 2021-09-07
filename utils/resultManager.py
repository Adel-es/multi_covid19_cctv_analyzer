import json 
from collections import OrderedDict


class ResultManager : 
    def __init__(self) :         
        self.target_in :int = 0 
        self.target_out :int = 0 
        self.continued : bool= False 
        
        self.result_dict = OrderedDict() 
        self.result_dict["target"] = [] 
    
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
    
    def write_jsonfile(self, filename : str) : 
        """ write result_dict as json format 
        
        Args : 
            filename : filename output file 
        """
        print(json.dumps(self.result_dict, indent="\t"))
        with open(filename, 'w', encoding='utf-8') as write_file : 
            json.dump(self.result_dict, write_file, indent="\t")
        
            