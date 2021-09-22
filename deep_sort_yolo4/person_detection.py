#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import sys
root_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_path)

import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import tensorflow as tf

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from configs import runInfo
from utils.types import BBox

warnings.filterwarnings('ignore')

input_video_path = runInfo.input_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
gpuNum = runInfo.trackingGPU

def positioning_in_frame(bbox, f_width, f_height):
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[2] > f_width:
        bbox[2] = f_width
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[3] > f_height:
        bbox[3] = f_height

def detectAndTrack(shm, processOrder, nextPid):
    # 영상이 바뀐 runInfo의 값을 갱신
    input_video_path = runInfo.input_video_path
    start_frame = runInfo.start_frame
    end_frame = runInfo.end_frame
    gpuNum = runInfo.trackingGPU
    print(" ************ detectAndTrack: input_video_path: {}".format(input_video_path))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Limit TensorFlow to use only the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpuNum], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpuNum], True)
        except RuntimeError as e:
            print(e)
                
    # Get detection model
    yolo = YOLO()

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Get tracker (Deep SORT)
    model_filename = f'{root_path}/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Prepare input videovideo_path
    video_capture = cv2.VideoCapture(input_video_path)
    frame_width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_index = -1
    
    myPid = 'detectAndTrack'
    shm.init_process(processOrder, myPid, nextPid)
    
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        
        # for test
        frame_index += 1
        if frame_index < start_frame:
            continue
        if frame_index > end_frame:
            break
        # for test
        
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)

        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                        zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        tids = []
        bboxes = []
        confidences = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            
            positioning_in_frame(bbox, frame_width, frame_height)
            
            tids.append(track.track_id)
            bboxes.append(bbox)
            confidences.append(track.confidence)
            
        peopleNum = len(tids)
        frameIdx, personIdx = shm.get_ready_to_write(peopleNum)
        for i in range(peopleNum):
            # Write at people
            shm.data.people[ personIdx[i] ].bbox = BBox(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], confidences[i])
            shm.data.people[ personIdx[i] ].tid = tids[i]
            
        shm.finish_a_frame()
    # end of while()
    
    shm.finish_process()
    video_capture.release()
