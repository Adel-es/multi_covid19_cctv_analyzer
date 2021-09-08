import cv2 
import numpy as np 
import os.path
from pypreprocessor import pypreprocessor
import logging
import maskdetect.faceDetector.faceDetector as face
import maskdetect.maskClassifier.maskClassifier as mask
import math 
import libs.votingSystem as vs
from configs import runInfo
from utils.types import MaskToken

start_frame = runInfo.start_frame 
end_frame = runInfo.end_frame 
input_video = runInfo.input_video_path 
FRAME_NUM = end_frame - start_frame + 1 

def cutoff_face(person_image, face_bbox) : 
	person_width = math.trunc(person_image.shape[0]) 
	person_height = math.trunc(person_image.shape[1])  
	face_xmin = int(max(0, face_bbox[0])) 
	face_ymin = int(max(0, face_bbox[1])) 
	face_xmax = int(min(person_width, face_bbox[2])) 
	face_ymax = int(min(person_height, face_bbox[3])) 
	face = person_image[face_ymin : face_ymax, face_xmin : face_xmax]
	return face 
    
def runMaskDetection(shm, processOrder, nextPid):
	logger = logging.getLogger('root') 
	faceDetector = face.Detector(runInfo.faceGPU)
	maskClassifier = mask.Classifier(runInfo.maskGPU)

	frame_index = -1
	input_capture = cv2.VideoCapture(input_video)
	logger.info("mask detection process starts")
 
	myPid = 'maskDetection'
	shm.init_process(processOrder, myPid, nextPid) 

	if os.path.exists(runInfo.input_video_path) == False:
		logger.critical("[IO Error] Input video path: '{}' is not exists".format(runInfo.input_video_path))
		exit(-1)
 
	while (input_capture.isOpened()) :
		frame_index = frame_index + 1 
		ret, raw_image = input_capture.read() 

		if ret == False : 
			logger.critical("mask detection process last frame :)")
			break 
		if frame_index < start_frame : 
			continue 
		if frame_index > end_frame : 
			break 

		shm_frame_index, person_indices = shm.get_ready_to_read() 
		reid = shm.data.frames[shm_frame_index].reid
		if reid == -1 : 
			shm.finish_a_frame()
			continue         
        
		faces_in_frame = [] 
		tids_in_frame = [] 
		
		for p_index in person_indices : 
			person = shm.data.people[p_index] 
			bbox = person.bbox 
			if (person.isClose == False) : 
				shm.data.people[p_index].isMask = int(MaskToken.NotNear)
			else : 
				cropped_person = raw_image[int(bbox.minY): int(bbox.maxY), int(bbox.minX):int(bbox.maxX)] 
				rgb_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
				face_list = faceDetector.inference(rgb_person)
				if len(face_list) == 0 : 
					shm.data.people[p_index].isMask = int(MaskToken.FaceNotFound)
				else :
					cropped_face = cutoff_face(cropped_person, face_list[0])
					cropped_face = cv2.resize(cropped_face, (maskClassifier.width, maskClassifier.height)) 
					cropped_face = cropped_face/255.0 
					faces_in_frame.append(cropped_face)
					tids_in_frame.append(p_index)
		
		if len(faces_in_frame) != 0 : 
			mask_results, scores = maskClassifier.inference(np.array(faces_in_frame))
			for index, result in enumerate(mask_results) : 
				if result == 0 : 
					shm.data.people[tids_in_frame[index]].isMask = int(MaskToken.Masked)
				else : 
					shm.data.people[tids_in_frame[index]].isMask = int(MaskToken.NotMasked)
		
		shm.finish_a_frame()	
	print("{} reader_and_writer: finish".format(myPid))
	shm.finish_process()
