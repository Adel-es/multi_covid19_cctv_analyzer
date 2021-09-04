import openpifpaf
import torch
import numpy as np
import PIL
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import libs.fpsCalculator as FPST


class OpenPPWrapper:
    """
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self):
        USE_CUDA = torch.cuda.is_available()
        self.gpu_num = 6; #TODO; remove hard coded  
        self.checkpoint = "shufflenetv2k16"
        self.device = torch.device("cuda:{}".format(self.gpu_num) if USE_CUDA else 'cpu')
        self.timer = FPST.FPSCalc()
        self.fps = 0; 
        self.model, self.preprocess, self.processor = self.load_model()
        # self.w = 30 # TODO 
        # self.h = 80 #TODO
        
      
    def load_model(self):
        openpifpaf.network.Factory.checkpoint = self.checkpoint 
        model_cpu, _ = openpifpaf.network.Factory().factory()          
        model = model_cpu.to(self.device)
        # model = torch.nn.DataParallel(model)  
        rescale_t = None  #setting for long edige = default None 
        pad_t = None     
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
        processor = openpifpaf.decoder.factory(model_cpu.head_metas)
        return model, preprocess, processor 
        
        
    def dataset(self, data):
        batch_size = 1
        loader_workers = batch_size if len(data) > 1 else 0
        loader_workers = 0 # for avoding cuda reinitialize error 
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False,
            pin_memory=self.device.type != 'cpu',
            num_workers=loader_workers,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        yield from self.dataloader(dataloader)

    def dataloader(self, dataloader):
        for batch_i, item in enumerate(dataloader):
            if len(item) == 3:
                processed_image_batch, gt_anns_batch, meta_batch = item
                image_batch = [None for _ in processed_image_batch]
            elif len(item) == 4:
                image_batch, processed_image_batch, gt_anns_batch, meta_batch = item

            pred_batch = self.processor.batch(self.model, processed_image_batch, device=self.device)

            # un-batch
            for image, pred, gt_anns, meta in \
                    zip(image_batch, pred_batch, gt_anns_batch, meta_batch):
                      
                pred = [ann.inverse_transform(meta) for ann in pred]
                gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]
                yield pred, gt_anns, meta
                
    def _inference(self, resized_rgb_image) : 
        data = openpifpaf.datasets.NumpyImageList(
            [resized_rgb_image], preprocess=self.preprocess, with_raw_image=True)
        return next(iter(self.dataset(data)))

    def inference(self, resized_rgb_image):
        predictions, gt_anns, meta = self._inference(resized_rgb_image)
        
        result = []
        for i in range (0, len(predictions)) :
            score = predictions[i].score 
            if(score < 0.2 ) : 
                continue 
            keypoints = predictions[i].data 
            bbox = predictions[i].bbox()  
            
            isFace, faceBox = self.getFacePosition(keypoints)
            if isFace == True : 
                result.append(faceBox)
        return result
    
    
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