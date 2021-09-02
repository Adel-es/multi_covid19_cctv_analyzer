
import os, sys
# 상위 디렉토리 절대 경로 추가
# ~/covid19_cctv_analyzer
root_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(root_path + "/personReid/top-dropblock")
sys.path.append(root_path + "/personReid/LA_Transformer")

from main import config_for_topdb, run_top_db_test
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

def fakeReid(shm, processOrder, nextPid):
    myPid = 'fakeReid'
    shm.init_process(processOrder, myPid, nextPid)
    
    # select confirmed case randomly
    FRAME_NUM = end_frame - start_frame + 1
    doneFIdx = FRAME_NUM - 1
    for fIdx in range(FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        if len(personIdx) > 0:
            random_pIdx = random.choice(personIdx)
            confirmed_tid = shm.data.people[random_pIdx].tid
            shm.data.frames[frameIdx].reid = random_pIdx
            shm.finish_a_frame()
            doneFIdx = fIdx
            break
        shm.data.frames[frameIdx].reid = -1
        shm.finish_a_frame()
    
    # find confirmed_tid
    for fIdx in range(doneFIdx + 1, FRAME_NUM):
        frameIdx, personIdx = shm.get_ready_to_read()
        shm.data.frames[frameIdx].reid = -1
        for pIdx in personIdx:
            if shm.data.people[pIdx].tid == confirmed_tid:
                shm.data.frames[frameIdx].reid = pIdx
                break
        shm.finish_a_frame()
        
    shm.finish_process()
    
def personReid_topdb(shm, processOrder, nextPid):
    # CUDA_VISIBLE_DEVICES를 0으로 설정하지 않으면 topdb 돌릴 때 아래와 같은 err가 뜬다
    # TypeError: forward() missing 1 required positional argument: 'x'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    top_db_engine, top_db_cfg = config_for_topdb( root_path , query_image_path=query_image_path)
    myPid = 'topdbReid'
    run_top_db_test(engine=top_db_engine, cfg=top_db_cfg, 
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path,
                    shm=shm, processOrder=processOrder, myPid=myPid, nextPid=nextPid,
                    query_image_path=query_image_path)
    # 지금 reidRslt에서 확진자가 없는 경우(-1)는 나오지 않는다. (reid 정확성 문제 때문에)
    
def personReid_la_transformer(shm, processOrder, nextPid):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    calculation_mode = 'original' # 'custom' or 'original'
    myPid = 'laReid'
    
    model, data_transforms = config_la_transformer(root_path)
    run_la_transformer(model=model, data_transforms=data_transforms,
                    root_path=root_path, query_image_path=query_image_path,
                    start_frame=start_frame, end_frame=end_frame,
                    input_video_path=input_video_path, output_video_path=output_video_path, 
                    shm=shm, processOrder=processOrder, myPid=myPid, nextPid=nextPid,
                    calculation_mode=calculation_mode,
                    debug_enable=False,
                    debug_logging_file_path=root_path+"/la_trans_log.txt")

def runPersonReid(shm, processOrder, nextPid, select_reid_model): 
    if select_reid_model == 'topdb':
        personReid_topdb(shm, processOrder, nextPid)
    elif select_reid_model == 'la':
        personReid_la_transformer(shm, processOrder, nextPid)
    elif select_reid_model == 'fake':
        fakeReid(shm, processOrder, nextPid)
    else:
        print("Plz Select PersonReid model")
        sys.exit()