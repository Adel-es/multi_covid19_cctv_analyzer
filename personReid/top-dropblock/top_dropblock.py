from posix import sched_param
from personReid.utils.votingSystem import VotingSystem
import sys
import os
import os.path as osp
import warnings
import time
import argparse

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.votingSystem as vs 

from default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs, engine_test_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)

from torchreid.utils import read_image
import glob

import cv2
import imutils.video
from PIL import Image
import numpy as np

def build_datamanager(cfg, query_image_path= ''):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg), query_image_path=query_image_path)
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch':
            engine = torchreid.engine.ImageTripletDropBatchEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet_dropbatch_dropbotfeatures':
            engine = torchreid.engine.ImageTripletDropBatchDropBotFeaturesEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                weight_db_t=cfg.loss.dropbatch.weight_db_t,
                weight_db_x=cfg.loss.dropbatch.weight_db_x,
                weight_b_db_t=cfg.loss.dropbatch.weight_b_db_t,
                weight_b_db_x=cfg.loss.dropbatch.weight_b_db_x,
                top_drop_epoch=cfg.loss.dropbatch.top_drop_epoch,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        elif cfg.loss.name == 'triplet':
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        else:
            exit("ERROR")
    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms

def main_concat_with_track( config_file_path, data_root_path , query_image_path, gpu_idx):
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    # if args.config_file:
    #     cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(config_file_path)
    cfg.data.root = data_root_path
    
    # reset_config(cfg, args)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_random_seed(cfg.train.seed)

    # if cfg.use_gpu and args.gpu_devices:
    #     # if gpu_devices is not specified, all available gpus will be used
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    
    # print('Show configuration\n{}\n'.format(cfg))
    # print('Collecting env info ...')
    # print('** System info **\n{}\n'.format(collect_env_info()))
    
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    datamanager = build_datamanager(cfg, query_image_path)
    
    # print(type(datamanager))
    # print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids, # class 종류 개수를 특정할 수 있나?
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    # print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    # load weights 경로가 없으면 error. 무조건 유저가 weights 넣어주게 하기
    topdb_dir = os.path.dirname(os.path.abspath(__file__))
    '''
    weights_path = "{}/{}".format(topdb_dir, cfg.model.load_weights)
    if cfg.model.load_weights == '' or not check_isfile(weights_path):
        print(" [Reid Error] {}: Weights file of top-dropblock is not exist\n\n".format(weights_path))
        exit(-1)
    # 경로가 있으면 load 하기
    load_pretrained_weights(model, weights_path)
    '''
    
    if cfg.use_gpu:
        device = torch.device("cuda:{}".format(gpu_idx))
        model = nn.DataParallel(model, device_ids=[gpu_idx]).to(device)
        # model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    # if cfg.model.resume and check_isfile(cfg.model.resume):
    #     args.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer)

    # print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)

    return engine, cfg

def config_for_topdb(root_path, query_image_path, gpu_idx):
    query_image_path = root_path + "/" + query_image_path
    config_file_path = root_path + "/personReid/top-dropblock/configs/im_top_bdnet_test_concat_track.yaml"
    data_root_path = root_path + "/personReid/top-dropblock/data"
    return main_concat_with_track(config_file_path, data_root_path, query_image_path, gpu_idx)

def crop_frame_image(frame, bbox):
    return Image.fromarray(frame).crop( (int(bbox.minX),int(bbox.minY), 
                                         int(bbox.maxX),int(bbox.maxY)) ) # (start_x, start_y, start_x + width, start_y + height) 
     
def run_top_db_test(engine, cfg, start_frame, end_frame, use_vote, reid_threshold,
                    input_video_path,
                    shm, processOrder, myPid, nextPid,
                    query_image_path):
    #DEBUG
    print("++++++++++++debug+++++++++++++++++")
    print(start_frame)
    print(end_frame)
    print("+++++++++++++++++++++++++++++++++++")

    if use_vote : 
        votingSystem = vs.VotingSystem()
    video_capture = cv2.VideoCapture(input_video_path)

    fps_imutils = imutils.video.FPS().start()

    cam_id = 0;     # 임의로 cam_no 정의
    frame_no = -1   
    
    shm.init_process(processOrder, myPid, nextPid)
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        frame_no += 1 # frame no 부여
        if frame_no < start_frame:
            continue
        if frame_no > end_frame:
            break
        if frame_no % 4 != 0 : 
            continue 
        # if frame_no % 10 == 0:
            # print("\tFrame no in topdb: {}".format(frame_no))
        # print("\tFrame no in topdb: {}".format(frame_no))
        frameIdx, personIdx = shm.get_ready_to_read()
        
        # frame에 사람이 없다면 pass
        if len(personIdx) == 0: 
            shm.data.frames[frameIdx].reid = -1
            shm.finish_a_frame()
            continue
        
        # frame에 있는 사람들 사진을 수집
        gallery = []
        for pIdx in personIdx:
            bbox = shm.data.people[pIdx].bbox
            tid = shm.data.people[pIdx].tid
            image = crop_frame_image(frame, bbox)       # PIL type
            gallery.append( (image, tid, cam_id, pIdx))
        
        # reid 수행
        top1_gpIdx, confidenceList = engine.test_only(gallery_data = gallery, query_image_path=query_image_path, **engine_test_kwargs(cfg)) # top1의 index
        top1_tid = shm.data.people[top1_gpIdx].tid 
        for i, pIdx in enumerate(personIdx):
            shm.data.people[pIdx].reidConf = confidenceList[i]
        top1_conf = shm.data.people[top1_gpIdx].reidConf

        if use_vote :         
            if top1_conf >= reid_threshold :
                vote_tid = votingSystem.vote(top1_tid)
            else :
                vote_tid = votingSystem.get_most_vote_tid()
            
            vote_tid_pIdx = -1
            for _pIdx in personIdx : 
                if shm.data.people[_pIdx].tid == vote_tid :
                    vote_tid_pIdx = _pIdx
                    if _pIdx != top1_gpIdx : 
                        # print("after voting reid target is chagned before({}) -> after({})".format(top1_tid, vote_tid))
                        shm.data.people[_pIdx].reidConf = top1_conf
                    break 
            shm.data.frames[frameIdx].reid = vote_tid_pIdx
                
        else : # not voting
            if top1_conf >= reid_threshold : 
                shm.data.frames[frameIdx].reid = top1_gpIdx
            else:
                shm.data.frames[frameIdx].reid = -1
        
        # print("********** distance :", top1_conf)
            
        fps_imutils.update()
        shm.finish_a_frame()
            
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    video_capture.release()
    
    shm.finish_process()