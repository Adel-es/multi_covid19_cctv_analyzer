import json
import scipy.io
import numpy as np
from collections import OrderedDict

def getShmFilePath(input_video_path):
    '''
    ex: input_video_path = "data/input/video.mp4"
        -> return "accuracy_check/shm_file/video_shm.json"
    '''
    videoName = input_video_path.split('/')[-1]
    videoName = videoName.split('.')[0]
    return 'accuracy_check/shm_file/{}_shm.json'.format(videoName)

def getGTruthFilePath(input_video_path):
    '''
    ex: input_video_path = "data/input/video.mp4"
        -> return "accuracy_check/gTruth_file/video_gTruth.mat"
    '''
    videoName = input_video_path.split('/')[-1]
    videoName = videoName.split('.')[0]
    return 'accuracy_check/gTruth_file/{}_gTruth.mat'.format(videoName)

def writeShmToJsonFile(data, start_frame, end_frame, input_video_path):
    '''
    Param: 
        - data: ShmSerialManger.data
        - start_frame, end_frame: int type
        - input_video_path: string type
    Return: 
        None
    Output: 
        .json file 
    '''
    shm = OrderedDict()
    
    FRAME_NUM = end_frame - start_frame + 1
    
    shm["start_frame"] = start_frame
    shm["end_frame"] = end_frame
    
    frames = []
    people = []
    
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = data.get_index_of_frame(fIdx+1)
    
        frames.append( {"reid": data.frames[frameIdx].reid} )
        aFramePeople = []
        for pIdx in personIdx:
            person = data.people[pIdx]
            aFramePeople.append({
                "bbox": [person.bbox.minX, person.bbox.minY, person.bbox.maxX, person.bbox.maxY], 
                "tid": person.tid, 
                "isClose": person.isClose, 
                "isMask": person.isMask
            })
        people.append(aFramePeople)
            
    
    shm["frames"] = frames
    shm["people"] = people

    # Write JSON
    with open(getShmFilePath(input_video_path), 'w', encoding="utf-8") as make_file:
        json.dump(shm, make_file, ensure_ascii=False, indent="\t")
        
        
def convertShmFileToJsonObject(shm_file_path):
    '''
    Param: 
        shm_file_path: path of .json file written by writeShmToFile() 
    Return: 
        shm = {
            "start_frame": 0,
            "end_frame": 9,
            "frames": [
                {"reid": -1}, 
                ... , 
                {"reid": 8}
            ],
            "people": [
                [],
                ... , 
                [{"bbox": [5.5,7.9,7.8,7.0], "tid": 1, "isClose": false, "isMask": 4}, ...]
            ]
        }
    '''
    with open(shm_file_path) as json_file:
        shm = json.load(json_file)
    return shm


def convertGTruthFileToJsonObject(gTruth_file_path):
    '''
    Param: 
        gTruth_file_path: path of .mat file converted by jsonencode() in MATLAB
            (How to make .mat file: https://www.notion.so/bob8ysh/Python-mat-60e5bb65c5864930b756def24c4daafa)
    Return: 
        gTruth = {
            "P1": [[], [], ... , {'Position': [94.0, 75.0, 190.1, 264.6], 'ismask': 'notfound'}, ... , [], []],
            ...
            "P8": [[], ... , []]
        }
    '''
    mat_file = scipy.io.loadmat(gTruth_file_path)

    # Convert mat format file to json object
    gTruth = dict()
    for key in mat_file:
        if type(mat_file[key]) == np.ndarray:
            string_data = mat_file[key][0]
            json_data = json.loads(string_data)
            gTruth[key] = json_data
    
    # Change gTruth's bounding box format from [minX, minY, width, height] to [minX, minY, maxX, maxY]
    for key in gTruth:
        for i in range(len(gTruth[key])):
            if type(gTruth[key][i]) == dict:
                minX, minY, width, height = gTruth[key][i]['Position']
                maxX = minX + width
                maxY = minY + height
                gTruth[key][i]['Position'] = [minX, minY, maxX, maxY]
    
    return gTruth