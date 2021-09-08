import file_io

videoName = "08_14_2020_1_1.mp4"
# Create shm_file_path based on runInfo.input_video_path
shm_file_path = file_io.getShmFilePath(videoName) 
shm = file_io.convertShmFileToJsonObject(shm_file_path)
# Create gTruth_file_path based on runInfo.input_video_path
gTruth_file_path = file_io.getGTruthFilePath(videoName) 
gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
        
print("shm: {}\n".format(shm))

P2_tid = 1
shm_P2 = []
for aFramePeople in shm['people']:
    for person in aFramePeople:
        if person['tid'] == P2_tid:
            shm_P2.append(person)
print("shm_P2: {}\n".format(shm_P2))

start_frame = shm["start_frame"]
end_frame = shm["end_frame"]
print("gTruth_P2: {}".format(gTruth["P2"][start_frame:end_frame+1]))