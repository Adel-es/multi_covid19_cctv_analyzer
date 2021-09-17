
import os, sys
# 상위 디렉토리 절대 경로 추가
# ~/covid19_cctv_analyzer
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path + "/personReid/top-dropblock")
sys.path.append(root_path + "/personReid/LA_Transformer")

from top_dropblock import config_for_topdb, run_top_db_test
from la_transformer import config_la_transformer, run_la_transformer

from configs import runInfo
from collections import Counter # for test
import random

# input_video_path = 'OxfordTownCentreDataset.avi'
input_video_path = runInfo.input_video_path
output_video_path = runInfo.output_video_path
start_frame = runInfo.start_frame
end_frame = runInfo.end_frame
query_image_path = runInfo.query_image_path

def fakeReid3(shm, processOrder, nextPid):
    '''
        pick the first one
        if there is no one in frame, reid is -1 
        this is usually used for mask system evaluation! (see code in run_mask_accuracy.py) 
    '''
    myPid = 'fakeReid3'
    shm.init_process(processOrder, myPid, nextPid)
    
    t_conf = 0.6
    f_conf = 0.1
    # find confirmed_tid
    for fIdx in range(start_frame, end_frame + 1):
        frameIdx, personIdx = shm.get_ready_to_read()
        if len(personIdx) == 0 : 
            shm.data.frames[frameIdx].reid = -1
            shm.data.frames[frameIdx].confidence = f_conf 
        else : 
            shm.data.frames[frameIdx].reid = shm.data.people[personIdx[0]].tid
            shm.data.frames[frameIdx].confidence = t_conf
        shm.finish_a_frame()
    shm.finish_process()

def fakeReid2(shm, processOrder, nextPid):
    myPid = 'fakeReid2'
    shm.init_process(processOrder, myPid, nextPid)
    
    # decide confirmed tid 
    confirmed_tid = {1}
    
    t_conf = 0.6
    f_conf = 0.1
    # find confirmed_tid
    for fIdx in range(start_frame, end_frame):
        frameIdx, personIdx = shm.get_ready_to_read()
        shm.data.frames[frameIdx].reid = -1
        shm.data.frames[frameIdx].confidence = f_conf
        for pIdx in personIdx:
            if shm.data.people[pIdx].tid in confirmed_tid:
                shm.data.frames[frameIdx].reid = pIdx
                shm.data.frames[frameIdx].confidence = t_conf
                break
        shm.finish_a_frame()
        
    shm.finish_process()

def fakeReid(shm, processOrder, nextPid):
    myPid = 'fakeReid'
    shm.init_process(processOrder, myPid, nextPid)
    
    t_conf = 0.6
    f_conf = 0.1
    # select confirmed case randomly
    FRAME_NUM = end_frame - start_frame + 1
    doneFIdx = FRAME_NUM - 1
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        if len(personIdx) > 0:
            random_pIdx = random.choice(personIdx)
            confirmed_tid = shm.data.people[random_pIdx].tid
            shm.data.frames[frameIdx].reid = random_pIdx
            shm.data.frames[frameIdx].confidence = t_conf
            shm.finish_a_frame()
            doneFIdx = fIdx
            break
        shm.data.frames[frameIdx].reid = -1
        shm.data.frames[frameIdx].confidence = f_conf
        shm.finish_a_frame()
    
    # find confirmed_tid
    for fIdx in range(doneFIdx + 1, FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        shm.data.frames[frameIdx].reid = -1
        shm.data.frames[frameIdx].confidence = f_conf
        for pIdx in personIdx:
            if shm.data.people[pIdx].tid == confirmed_tid:
                shm.data.frames[frameIdx].reid = pIdx
                shm.data.frames[frameIdx].confidence = t_conf
                break
        shm.finish_a_frame()
        
    shm.finish_process()
    
def personReid_topdb(shm, processOrder, nextPid, gpu_idx):
    myPid = 'topdbReid'
    top_db_engine, top_db_cfg = config_for_topdb( root_path=root_path, query_image_path=query_image_path, gpu_idx=gpu_idx)
    run_top_db_test(engine=top_db_engine, cfg=top_db_cfg, 
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path,
                    shm=shm, processOrder=processOrder, myPid=myPid, nextPid=nextPid,
                    query_image_path=query_image_path)
    # 지금 reidRslt에서 확진자가 없는 경우(-1)는 나오지 않는다. (reid 정확성 문제 때문에)
    
def personReid_la_transformer(shm, processOrder, nextPid, calculation_mode, gpu_idx):
    myPid = 'laReid'
    model, data_transforms = config_la_transformer(root_path, gpu_idx, gpu_usage_check=False)
    run_la_transformer(model=model, data_transforms=data_transforms,
                    root_path=root_path, query_image_path=query_image_path,
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path, 
                    shm=shm, processOrder=processOrder, myPid=myPid, nextPid=nextPid,
                    calculation_mode=calculation_mode,
                    debug_enable=False,
                    debug_logging_file_path=root_path+"/la_trans_log.txt")

def set_cuda_visible_devices(gpu_idx):
    '''
        사용 가능한 GPU device id 환경변수 설정.
        Topdb에서는 CUDA_VISIBLE_DEVICES 환경변수에 gpu_idx 이하의 device id가 모두 포함이 되어있어야 한다.
    '''
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        
    enable_gpu = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    for g_idx in range(0, gpu_idx+1):
        if str(g_idx) not in enable_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] += ","+str(g_idx)
   
def runPersonReid(shm, processOrder, nextPid, select_reid_model, gpu_idx=0): 
    set_cuda_visible_devices(gpu_idx)
    
    if select_reid_model == 'topdb':
        personReid_topdb(shm, processOrder, nextPid, gpu_idx)
    elif select_reid_model == 'la':
        calculation_mode = 'custom' # 이제 무조건 custom만 사용 가능 # 'custom' or 'original'
        personReid_la_transformer(shm, processOrder, nextPid, calculation_mode, gpu_idx)
    elif select_reid_model == 'fake':
        fakeReid(shm, processOrder, nextPid)
    elif select_reid_model == 'fake2' : 
        fakeReid2(shm, processOrder, nextPid)
    else:
        print("Plz Select PersonReid model")
        sys.exit()