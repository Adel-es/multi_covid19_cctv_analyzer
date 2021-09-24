#! /bin/sh

# soource accuracy_check/run_mask.sh
conda activate track-reid-mask
which python 

accFile=accuracy_check/run_accuracy.py
configFile=configs/runInfo.py

if [ $# -ne 1 ]; then
    echo "script need 1 parameter : outputFileName"
    exit 1
fi
resultFile=$1

echo "WITHOUT MASK VOTING" >> $resultFile
echo "" >> $resultFile
for i in {1..4}
do
    echo "===========mask accucracy check #$i ================"
    sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    sed -i "s/TEST_REID =.*/TEST_REID = False/" $accFile
    sed -i "s/TEST_MASK =.*/TEST_MASK = True/" $accFile
    sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile

    inputVideo="accuracy_check\/test_video_data\/08_14_2020_${i}_1.mp4"
    inputCommand="s/input_video_path =.*/input_video_path = '${inputVideo}'/"
    sed -i "${inputCommand}" $configFile
    sed -i "s/use_mask_voting =.*/use_mask_voting = False/" $configFile

    tmpFile="tmp${i}"
    python accuracy_check/run_accuracy.py > "${tmpFile}"
    status=$?
    echo "Status : ${status}"
    if [ $status -eq 0 ]
    then
        echo "Success... in ${i}"
    else
        echo "Failure... in ${i}"
        break
    fi
    echo "=== result of ${inputVideo} ===" >> $resultFile
    awk '/average precision : /' "${tmpFile}" >> $resultFile
    awk '/average recall : /' "${tmpFile}" >> $resultFile 
    awk '/F1-score : /' "${tmpFile}" >> $resultFile
    rm "${tmpFile}"
done


echo "WITH MASK VOTING" >> $resultFile
echo "" >> $resultFile
for i in {1..4}
do
    echo "===========mask accucracy check #$i ================"
    sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    sed -i "s/TEST_REID =.*/TEST_REID = False/" $accFile
    sed -i "s/TEST_MASK =.*/TEST_MASK = True/" $accFile
    sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile

    inputVideo="accuracy_check\/test_video_data\/08_14_2020_${i}_1.mp4"
    inputCommand="s/input_video_path =.*/input_video_path = '${inputVideo}'/"
    sed -i "${inputCommand}" $configFile
    sed -i "s/use_mask_voting =.*/use_mask_voting = True/" $configFile

    tmpFile="tmp${i}"
    python accuracy_check/run_accuracy.py > "${tmpFile}"
    status=$?
    echo "Status : ${status}"
    if [ $status -eq 0 ]
    then
        echo "Success... in ${i}"
    else
        echo "Failure... in ${i}"
        break
    fi
    echo "=== result of ${inputVideo} ===" >> $resultFile
    awk '/average precision : /' "${tmpFile}" >> $resultFile
    awk '/average recall : /' "${tmpFile}" >> $resultFile 
    awk '/F1-score : /' "${tmpFile}" >> $resultFile
    rm "${tmpFile}"
done