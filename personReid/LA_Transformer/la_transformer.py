from __future__ import print_function

import os
import time
import glob
import random
import zipfile
from itertools import chain

import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from LATransformer.model import ClassBlock, LATransformer, LATransformerTest
from LATransformer.utils import save_network, update_summary, get_id
from LATransformer.metrics import rank1, rank5, rank10, calc_map
from LATransformer.distance import compute_distance_matrix

import faiss
import numpy as np
import cv2
import imutils

from gpuinfo import GPUInfo

# Config parameters
batch_size = 8
gamma = 0.7
seed = 42

device = ""
model_save_path = ""
image_data_dir = ""
query_data_dir = "" 
# image_data_dir = "./data/Market-Pytorch/Market/"

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def extract_feature(model,image):
    img = image.to(device)
    output = model(img)
    # output = output.view([1,-1]) # tensor 값을 (1,x,y) -> (1,x*y)로 조정
    return output

def search(index, query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k

def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not os.path.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            id = int(os.path.basename(path).split('_')[0])
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img, id

def config_la_transformer(root_path, gpu_idx, gpu_usage_check=False):
    global device
    device = torch.device("cuda:{}".format(gpu_idx))
    
    model_save_path = root_path + "/personReid/LA_Transformer/model/"
    # image_data_dir = root_path + "/personReid/LA_Transformer/data/"
    
    print(model_save_path)
    ## Load Model
    # Load ViT
    print("Load ViT......")
    
    if gpu_usage_check == True:
        print(" * before loading LA-trans model * ")
        GPUInfo.get_user(1)
        GPUInfo.get_info()        
        
    vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
    vit_base= vit_base.to(device)

    if gpu_usage_check == True:
        print(" * After loading LA-trans model * ")
        GPUInfo.get_user(1)
        GPUInfo.get_info()       
        
    # Create La-Transformer
    print("Create La-Transformer......")
    model = LATransformerTest(vit_base, lmbd=8).to(device)

    # Load LA-Transformer
    print("Load LA-Transformer......")
    name = "la_with_lmbd_8"
    save_path = os.path.join(model_save_path, name,'net_best.pth')
    if not os.path.exists(save_path):
        print(" [LA-transformer Error] {}: Weights file of LA-transformer is not exists".format(model_save_path))
    model.load_state_dict(torch.load(save_path), strict=False)
    model.eval()
    
    ## Data Loader
    print("Load DataLoader......")
    transform_query_list = [
        transforms.Resize((224,224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_gallery_list = [
        transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = {
    'query': transforms.Compose( transform_query_list ),
    'gallery': transforms.Compose(transform_gallery_list),
    }
    
    return model, data_transforms

def get_query_data_paths(query_data_dir):
    query_data_paths = glob.glob(os.path.join(query_data_dir, '*.jpg'))
    query_data_paths += glob.glob(os.path.join(query_data_dir, '*.png'))
    query_data_paths += glob.glob(os.path.join(query_data_dir, '*.PNG'))
    
    return query_data_paths

def get_query_features(model, query_data_dir, query_transforms, calculation_mode):
    ## Read query image
    query_data_paths = get_query_data_paths(query_data_dir)
    
    if len(query_data_paths) == 0:
        print('Query image doesn\'t exist - directory path: \"{}\"'.format(query_data_dir))
        exit(-1)
    
    query_features = []
    qid = []
    
    for qpath in query_data_paths:
        qimage, id = read_image(qpath)      # img type : <class 'PIL.Image.Image'>
        qimage = query_transforms(qimage)   # img type : <class 'torch.Tensor'>
        qimage = torch.unsqueeze(qimage, 0)         # 차원을 강제로 늘려줌
        # image_datasets['query'].append(qimage) 
        
        feature = extract_feature(model, qimage)
        if calculation_mode != 'original':
            feature = feature.view([1,-1]) # tensor 값을 (1,x,y) -> (1,x*y)로 조정  
            
        feature = feature.data.cpu()
        query_features.append(feature)
        qid.append(id)
        
    query_features = torch.cat(query_features, 0)
    # print("* query features size", query_features.size())
    return query_features, qid


def get_gallery_features(model, gallery_data, gallery_transforms, calculation_mode):
    gallery_features = []
    # gid = []
    gpIdx = []
    for image, tid, camid, pIdx in gallery_data:
        gimage = gallery_transforms(image)
        gimage = torch.unsqueeze(gimage, 0)
        
        feature = extract_feature(model, gimage)
        if calculation_mode != 'original':
            feature = feature.view([1,-1]) # tensor 값을 (1,x,y) -> (1,x*y)로 조정  
        feature = feature.data.cpu()
        gallery_features.append(feature)
        # gid.append(tid)
        gpIdx.append(pIdx)
        
    gallery_features = torch.cat(gallery_features, 0)
    
    return gallery_features, gpIdx
    
def run_test_original(model, gallery_data, gallery_transforms, 
             query_features, qid=[],
             debug_logging_mode=False, debug_file=None):
    '''
    Args:
        gallery_data : list of tuple(PIL Image, tracking id, cam id)
        query_features : list of tensor
        model : model
        gallery_transforms : torchvision.transforms.Compose()
    '''
        
    gallery_features, gpIdx = get_gallery_features(model, gallery_data, gallery_transforms, 'original')
    
    ## Concat Averaged GELTs
    print("Concat Averaged GELTs......")
    concatenated_query_vectors = []
    for query in tqdm(query_features):
        fnorm = torch.norm(query, p=2, dim=1, keepdim=True)*np.sqrt(14)
        query_norm = query.div(fnorm.expand_as(query))
        concatenated_query_vectors.append(query_norm.view((-1))) # 14*768 -> 10752
        
    concatenated_gallery_vectors = []
    for gallery in tqdm(gallery_features):
        fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True) *np.sqrt(14)
        gallery_norm = gallery.div(fnorm.expand_as(gallery))
        concatenated_gallery_vectors.append(gallery_norm.view((-1))) # 14*768 -> 10752
 
    ## Calculate Similarity using FAISS
    index = faiss.IndexIDMap(faiss.IndexFlatIP(10752))
    index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gpIdx))
    
    top1_gpIdx_list = []    
    for query, id in zip(concatenated_query_vectors, qid):
        output = search(index, query, k=10) # rank10 까지 찾음
        # print("output: ", output)
        top1_gpIdx_list.append( (output[1][0][0], output[1][0][:-1]) )
        # output[1][0][0] == rank1의 label
        # print(" #{}'s calculated result: ".format(id), end='')
        print(output[1][0][:-1])
    
    return top1_gpIdx_list[0], top1_gpIdx_list[1] # gpIdx, gpConf

def run_test_custom(model, gallery_data, gallery_transforms, 
             query_features, qid=[], 
             debug_logging_mode=False, debug_file=None):
    '''
    Args:
        gallery_data : list of tuple(PIL Image, tracking id, cam id)
        query_features : list of tensor
        model : model
        gallery_transforms : torchvision.transforms.Compose()
    '''
        
    gallery_features, gpIdx = get_gallery_features(model, gallery_data, gallery_transforms, 'custom')
    
    dist_metric = 'cosine' #euclidean'
    distmat = compute_distance_matrix(query_features, gallery_features, dist_metric)
    distmat = distmat.numpy()
    # print("* distmat: ", distmat)
    
    # distmat을 점수에 따라 sorting
    num_q, num_g = distmat.shape
    dist_indices = np.argsort(distmat, axis=1)
    if dist_metric == 'cosine':
        dist_indices = dist_indices[:, ::-1]
        
    if debug_logging_mode == True:
        dist_sorting = np.sort(distmat, axis=1)
        if dist_metric == 'cosine':
            dist_sorting = dist_sorting[:, ::-1]
        # print("* sorting distmat: ", dist_sorting)
        debug_file.write("* sorting distmat: \n")
        str_distmat = ""
        for mat in distmat:
            str_mat = "["
            for m in mat:
                str_mat += str(m) + "\t"
            str_mat += "],\n"
            str_distmat += str_mat
        debug_file.write("{}\n".format(str_distmat))
        
        
    dist_indices = np.transpose(dist_indices) # dist_indices[0] = index list of top1-gallery
    top1_gpIdx_list = []
    top1_conf_list = []
    for idx in dist_indices[0]:
        top1_conf_list.append((distmat[0][ dist_indices[0][0] ] + 1.) / 2.)
        top1_gpIdx_list.append(gpIdx[idx])
    return top1_gpIdx_list, top1_conf_list

def crop_frame_image(frame, bbox):
    return Image.fromarray(frame).crop( (int(bbox.minX),int(bbox.minY), 
                                         int(bbox.maxX),int(bbox.maxY)) ) # (start_x, start_y, start_x + width, start_y + height) 
   
def run_la_transformer(model, data_transforms, 
                       root_path, query_image_path,
                    start_frame, end_frame, 
                    use_vote,
                    input_video_path, output_video_path, 
                    shm, processOrder, myPid, nextPid,
                    calculation_mode='custom',
                    debug_enable=False,
                    debug_logging_file_path=""):
    #DEBUG
    
    # top1 gallery의 결과를 저장하는 용도
    debug_logging_mode = False
    if debug_enable == True:
        print("++++++++++++debug+++++++++++++++++")
        print(start_frame)
        print(end_frame)
        # print(tracking_list)
        if debug_logging_file_path == "":
            print("Save file path fgor debugging is not exist.")
        else:
            if os.path.exists( os.path.dirname(debug_logging_file_path) ) == False:
                print("{}: Directory path is not exist")
            else:
                debug_file = open(debug_logging_file_path, 'w')                
                debug_logging_mode = True
                top1_counting_dict = {}
                tid_image_save_chk = {}
        print("+++++++++++++++++++++++++++++++++++")
    else:
        debug_file = None
             
    # extract query features
    query_data_dir = root_path + "/" + query_image_path
    query_features, qid_list = get_query_features(model, query_data_dir, data_transforms['query'],
                                                  calculation_mode)
    if debug_logging_mode == True:
        for qid in qid_list:
            top1_counting_dict[ str(qid) ] = {}

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
        print("\tFrame no in la-transformer: {}".format(frame_no))
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
        # query가 여러 명 주어졌을 경우, 각 query에 대한 top1 gid들의 list가 주어짐
        # if calculation_mode == 'original':
        #     top1_gpIdx_list, top1_conf_list = run_test_original(model = model, gallery_data = gallery, gallery_transforms = data_transforms['gallery'],
        #                         query_features = query_features, qid=qid_list,
        #                         debug_logging_mode=debug_logging_mode, debug_file=debug_file)
        # else:
        top1_gpIdx_list, top1_conf_list = run_test_custom(model = model, gallery_data = gallery, gallery_transforms = data_transforms['gallery'],
                            query_features = query_features, qid=qid_list,
                            debug_logging_mode=debug_logging_mode, debug_file=debug_file)
        top1_gpIdx = top1_gpIdx_list[0] # 첫번째 query의 gid를 임의로 reidRslt의 output으로 정함
        shm.data.frames[frameIdx].reid = top1_gpIdx
        shm.data.frames[frameIdx].confidence = top1_conf_list[0]
        
        if debug_logging_mode == True:
            # 각 query마다 어떤 gid가 top1으로 가장 많이 나왔는지 counting 함.
            for pIdx_list_idx, qid in enumerate(qid_list):
                pIdx = top1_gpIdx_list[pIdx_list_idx]
                gid = str( shm.data.people[pIdx].tid )
                if gid in top1_counting_dict[ str(qid) ]:
                    top1_counting_dict[ str(qid) ][gid] += 1
                else:
                    top1_counting_dict[ str(qid) ][gid] = 0
            
        # 결과 확인용 - gallery의 사진 출력
        if debug_logging_mode == True and len(gallery) != 0:
            for data in gallery:
                if str(data[1]) in tid_image_save_chk:
                    pass
                else:
                    tid_image_save_chk[ str(data[1]) ] = 1
                    cv2.imwrite(root_path+"/deep-sort-yolo4/tempData/video_gallery/"
                    +str( data[1] )+'_'+str(frame_no)+'.jpg', #gpid_frameno.jpg
                    np.asarray( data[0] , dtype=np.uint8) )
            
        fps_imutils.update()
        shm.finish_a_frame()
        
    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    video_capture.release()
        
    if debug_logging_mode == True:
        for qid in top1_counting_dict:
            debug_file.write(" #{}'s gid counting log:\n".format(qid))
            for gid in top1_counting_dict[qid]:
                debug_file.write("\t * gid {}: {}\n".format(gid, top1_counting_dict[qid][gid]))
        debug_file.close()
    
    shm.finish_process()
