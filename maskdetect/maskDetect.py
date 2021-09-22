import cv2 
import numpy as np 
from pypreprocessor import pypreprocessor
import logging
import maskdetect.faceDetection.faceDetector as face
import maskdetect.maskClassifier.maskDetector as mask
import libs.votingSystem as vs
from configs import runInfo
from utils.types import MaskToken

start_frame = runInfo.start_frame 
end_frame = runInfo.end_frame 
input_video = runInfo.input_video_path 
FRAME_NUM = end_frame - start_frame + 1 



def runMaskDetection(shm, processOrder, nextPid):
	# 영상이 바뀐 runInfo의 값을 갱신
	start_frame = runInfo.start_frame 
	end_frame = runInfo.end_frame 
	input_video = runInfo.input_video_path 
	FRAME_NUM = end_frame - start_frame + 1 
	print(" ********************** in maskDetection: input file path: {}".format(input_video))

	myPid = 'maskDetection'
	shm.init_process(processOrder, myPid, nextPid) 
	logger = logging.getLogger('root') 
	faceDetector = face.Detector(runInfo.faceGPU)
	maskDetector = mask.Classifier(runInfo.maskGPU)

	frame_index = -1
	logger.info("mask detection process starts")
	input_capture = cv2.VideoCapture(input_video)
	
	while (input_capture.isOpened()) :
		frame_index = frame_index + 1 
		ret, raw_image = input_capture.read() 

		if ret == False : 
			logger.critical("mask detection can not read the video")
			break 
		if frame_index < start_frame : 
			continue 
		if frame_index > end_frame : 
			break 

		shm_frame_index, person_indices = shm.get_ready_to_read() 
		reid = shm.data.frames[shm_frame_index].reid
		logger.debug("{} reader_and_writer: read frame {}".format(myPid, reid))
		if reid == -1 : 
			shm.finish_a_frame()
			continue         
        
		faces_in_frame = [] 
		tids_in_frame = [] 
		
		for p_index in person_indices : 
			person = shm.data.people[p_index] 
			bbox = person.bbox 
			person_width = bbox.maxX - bbox.minX 
			person_height = bbox.maxY - bbox.minY 
			if (person.isClose == False) : 
				shm.data.people[p_index].isMask = int(MaskToken.NotNear)
			else : 
				cropped_person = raw_image[int(bbox.minY): int(bbox.maxY), int(bbox.minX):int(bbox.maxX)] 
				rgb_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
				face_list = faceDetector.inference(rgb_person)
				if len(face_list) == 0 : 
					shm.data.people[p_index].isMask = int(MaskToken.FaceNotFound)
				else :
					face_xmin = int(max(0, face_list[0][0])) 
					face_ymin = int(max(0, face_list[0][1])) 
					face_xmax = int(min(person_width, face_list[0][2])) 
					face_ymax = int(min(person_height, face_list[0][3])) 
					cropped_face = cropped_person[face_ymin : face_ymax, face_xmin : face_xmax]
					cropped_face = cv2.resize(cropped_face, (maskDetector.width, maskDetector.height)) 
					cropped_face = cropped_face/255.0 
					faces_in_frame.append(cropped_face)
					tids_in_frame.append(p_index)
		
		if len(faces_in_frame) != 0 : 
			mask_results, scores = maskDetector.inference(np.array(faces_in_frame))
			for index, result in enumerate(mask_results) : 
				if result == 0 : 
					shm.data.people[tids_in_frame[index]].isMask = int(MaskToken.Masked)
				else : 
					shm.data.people[tids_in_frame[index]].isMask = int(MaskToken.NotMasked)
		
		shm.finish_a_frame()	
	print("{} reader_and_writer: finish".format(myPid))
	shm.finish_process()
