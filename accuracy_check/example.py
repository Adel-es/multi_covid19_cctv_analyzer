import file_io

videoName = "08_14_2020_1_1.mp4"
# Create shm_file_path based on runInfo.input_video_path
output_file_path = file_io.getOutputFilePath(videoName) 
systemOutput = file_io.convertOutputFileToJsonObject(output_file_path)
# Create gTruth_file_path based on runInfo.input_video_path
gTruth_file_path = file_io.getGTruthFilePath(videoName) 
gTruth = file_io.convertGTruthFileToJsonObject(gTruth_file_path)
        
print("systemOutput: {}\n".format(systemOutput))

P2_tid = 1
system_P2 = []
for aFramePeople in systemOutput['people']:
    for person in aFramePeople:
        if person['tid'] == P2_tid:
            system_P2.append(person)
print("system_P2: {}\n".format(system_P2))

start_frame = systemOutput["start_frame"]
end_frame = systemOutput["end_frame"]
print("gTruth_P2: {}".format(gTruth["P2"][start_frame:end_frame+1]))