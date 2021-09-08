import math
from configs import runInfo

start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
FRAME_NUM = end_frame - start_frame + 1

DISTANCE_BBOX_RATIO_CRITERIA = 5

def getCentroid(bbox, return_int=False):
    '''
    Returns the coordinate of the center point of the lower side of the bbox.
    If return_int is true, the type of return value is (int, int), otherwise, (float, float)
    '''
    centroid_x = (bbox.minX + bbox.maxX) / 2
    max_y = bbox.maxY
    if return_int:
        centroid_x = int(centroid_x)
        max_y = int(max_y)
    centroid = (centroid_x, max_y)
    return centroid

def checkDistance(shm, processOrder, nextPid):
    myPid = 'checkDistance'
    shm.init_process(processOrder, myPid, nextPid)
    
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
    
        reid = shm.data.frames[frameIdx].reid
        # If there is the confirmed case in this frame
        if reid != -1:
            confirmed = shm.data.people[reid]
            c_width = confirmed.bbox.maxX - confirmed.bbox.minX
            c_centroid = getCentroid(confirmed.bbox)
            
            for pIdx in personIdx:
                # Find the average bounding box width of two people
                person = shm.data.people[pIdx]
                width = person.bbox.maxX - person.bbox.minX
                bbox_average_width = (c_width + width) / 2
                # Find the distance between centroids for two people
                centroid = getCentroid(person.bbox)
                distance = math.sqrt(math.pow((c_centroid[0] - centroid[0]), 2) + math.pow((c_centroid[1] - centroid[1]), 2)) 
                # Compare the bbox_average_width and distance to determine if the two people are close
                is_close = distance <= DISTANCE_BBOX_RATIO_CRITERIA * bbox_average_width
                if is_close:
                    shm.data.people[pIdx].isClose = True
                else:
                    shm.data.people[pIdx].isClose = False
            
            # Set the confirmed case's isClose value to False
            shm.data.people[reid].isClose = False
            
        shm.finish_a_frame()
    shm.finish_process()