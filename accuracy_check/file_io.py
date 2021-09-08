import json
import scipy.io
import numpy as np
from collections import OrderedDict

def getOutputFilePath(input_video_path):
    '''
    ex: input_video_path = "data/input/video.mp4"
        -> output = "accuracy_check/system_output_file/video_output.json"
    '''
    videoName = input_video_path.split('/')[-1]
    videoName = videoName.split('.')[0]
    return 'accuracy_check/system_output_file/{}_output.json'.format(videoName)

def getGTruthFilePath(input_video_path):
    '''
    ex: input_video_path = "data/input/video.mp4"
        -> output = "accuracy_check/gTruth_file/video_gTruth.mat"
    '''
    videoName = input_video_path.split('/')[-1]
    videoName = videoName.split('.')[0]
    return 'accuracy_check/gTruth_file/{}_gTruth.mat'.format(videoName)

def writeShmToJsonFile(data, start_frame, end_frame, input_video_path):
    '''
    input: 
        data: ShmSerialManger.data
    output: 
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
    with open(getOutputFilePath(input_video_path), 'w', encoding="utf-8") as make_file:
        json.dump(shm, make_file, ensure_ascii=False, indent="\t")
        
        
def convertOutputFileToJsonObject(output_file_path):
    '''
    input: 
        output_file_path: path of .json file written by writeShmToFile() 
    output: 
        systemOutput = {
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
    with open(output_file_path) as json_file:
        system_output = json.load(json_file)
    return system_output


def convertGTruthFileToJsonObject(gTruth_file_path):
    '''
    input: 
        gTruth_file_path: path of .mat file converted by jsonencode() in MATLAB
            (How to make .mat file: https://www.notion.so/bob8ysh/Python-mat-60e5bb65c5864930b756def24c4daafa)
    output: 
        gTruth = {
            "P1": [[], [], ... , {'Position': [94.0, 7.0, 27.1, 64.6], 'ismask': 'notfound'}, ... , [], []],
            ...
            "P8": [[], ... , []]
        }
    '''
    mat_file = scipy.io.loadmat(gTruth_file_path)

    gTruth = dict()
    for key in mat_file:
        if type(mat_file[key]) == np.ndarray:
            string_data = mat_file[key][0]
            json_data = json.loads(string_data)
            gTruth[key] = json_data
    
    return gTruth