import openpifpaf
import torch
import numpy as np
import PIL
import os, sys 
import logging
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import libs.fpsCalculator as FPST

class OpenPPWrapper:
    """
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self, gpu_num):
        USE_CUDA = torch.cuda.is_available()
        self.logger = logging.getLogger('root')
        self.gpu_num = gpu_num
        self.device = torch.device("cuda:{}".format(self.gpu_num) if USE_CUDA else 'cpu')
        # self.timer = FPST.FPSCalc()
        # self.fps = 0; 
        self.net, self.processor = self.load_model()
        self.w = 30 # TODO 
        self.h = 80 #TODO
        
        self.logger.info("openpifpaf based face detection run on GPU : {}".format(self.gpu_num))
        self.logger.info("openpifpaf based face detector input image width : {}".format(self.w))
        self.logger.info("openpifpaf based face detector input imgae height : {}".format(self.h))

      
    def load_model(self):
        net_cpu, _ = openpifpaf.network.factory(checkpoint="resnet50", download_progress=False)
        net = net_cpu.to(self.device)
        openpifpaf.decoder.CifSeeds.threshold = 0.5
        openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
        openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
        processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)
        return net, processor

    def inference(self, resized_rgb_image):
        """
        This method will perform inference and return the detected bounding boxes
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        pil_im = PIL.Image.fromarray(resized_rgb_image)
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        # self.timer.start(); 
        for images_batch, _, __ in loader:
            predictions = self.processor.batch(self.net, images_batch, device=self.device)[0]
        # self.fps = self.timer.end(); 

        results = []
        for i, pred in enumerate(predictions):
            pred = pred.data
            check, facebox = self.getFacePosition(pred)
            if check == True : 
                results.append(facebox)
        return results 
        
    def getMedianValue(self, left, right) : 
        x = 0 
        y = 0 
        if left[2] <= 0 and right[2] <= 0 : 
            x = -1 
            y = -1 
        elif left[2] <= 0  : 
            x = right[0]
            y = right[1]
        elif right[2] <= 0 :
            x = left[0]
            y = left[1] 
        else : 
            x = (left[0] + right[0]) / 2.0 
            y = (left[1] + right[1]) / 2.0 
        return (x, y)  

    
    def getFacePosition(self, keypoints) : 
        '''
        keypoints : keypoint 2d array [[x, y, score], [x, y, score]....]
        returns : tuple (false, []) or (true, [xmin, ymin, xmax, ymax])
        
        (xmin, ymin) ...............
        ............................
        ............................
        ................(xmax, ymax)
        
        '''
        # eye, shoudler and hip check
        limitScore = 0
        isEye = keypoints[1][2] > limitScore or keypoints[2][2] > limitScore 
        isShoudler = keypoints[5][2] > limitScore or keypoints[6][2] > limitScore 
        isHeap = keypoints[11][2] > limitScore or keypoints[12][2] > limitScore
        if not (isShoudler and isHeap and isEye) : 
            return (False, [])
        
        shoulderX, soulderY = self.getMedianValue(keypoints[5], keypoints[6])
        heapX, heapY = self.getMedianValue(keypoints[11], keypoints[12])
        eyeX, eyeY = self.getMedianValue(keypoints[1], keypoints[2])
        # print("-------------------------------------------")
        # print("eyeX = {}, eyeY = {}".format(eyeX, eyeY))
        # print("-------------------------------------------")
        
        torsoH = heapY - soulderY 
        faceH = 5.0 / 11.0 * torsoH 
        faceW = faceH
        # faceW = faceH * 2.0 / 3.0 
        YMIN = int( eyeY - faceH / 3.0 )  
        YMAX = int( eyeY + faceH * 2.0 / 3.0 ) 
        XMIN = int( eyeX - faceW / 2.0 ) 
        XMAX = int( eyeX + faceW / 2.0 )
        # print("========================================================")
        # print("torsoX = {}".format(torsoH)) 
        # print("faceW, faceH = {}, {}".format(faceW, faceH))
        # print("BBOX = {}, {}, {}, {}".format(XMIN, YMIN, XMAX, YMAX))
        # print("========================================================")
        return (True, [XMIN, YMIN, XMAX, YMAX])