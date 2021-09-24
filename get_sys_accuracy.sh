#! /bin/sh

accFile=accuracy_check/run_accuracy.py
configFile=configs/runInfo.py
resultPath=result/
modelPath=personReid/top-dropblock/weight/
modelName=model.pth.tar-400

startFrame=631
endFrame=3571
for model in "gb" "both" "own"
do
	echo "Copy model"
	rm $modelPath$modelName
	echo "cp $modelPath${model}/$modelName $modelPath"
	cp $modelPath${model}/$modelName $modelPath
	pwd
	for i in {1..8}
		do
    			echo "===========reid accucracy check #$i ================"
    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = True/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${i}'/" $accFile
			
			sed -i "s/start_frame =.*/start_frame = ${startFrame}/" $configFile
			sed -i "s/end_frame =.*/end_frame = ${endFrame}/" $configFile
			queryRefImage="proj/data/input/test_pictures/${i}_*"
			queryDestPath="proj/data/input/query/"
			
			rm proj/data/input/query/*
			#rm "$queryDestPath*"
			cp $queryRefImage $queryDestPath

    			python accuracy_check/run_accuracy.py > $resultPath"reid_acc_${model}_P${i}.txt"
		done
done
