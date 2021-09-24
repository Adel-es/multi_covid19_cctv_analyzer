#! /bin/sh

accFile=accuracy_check/run_accuracy.py
configFile=configs/runInfo.py

resultPath=final_result/
modelPath=personReid/top-dropblock/weight/
modelName=model.pth.tar-400

startFrame=631
endFrame=3571
for videonum in {1..4}
do
	for querynum in {1..8}
		do
    			echo "============= Reid accucracy check #$querynum ================="
			echo "============= run_accuracy check ======================="
    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = True/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${querynum}'/" $accFile
    			sed -n "/TEST_BBOX =.*/p" $accFile
    			sed -n "/TEST_REID =.*/p" $accFile
    			sed -n "/TEST_MASK =.*/p" $accFile
    			sed -n "/TEST_SYSTEM =.*/p" $accFile
    			sed -n "/WRITE_VIDEO =.*/p" $accFile
			sed -n "/QUERY_GROUND_TRUTH =.*/p" $accFile
			
			echo "============= Image file copy check #$querynum================="
			queryRefImage=proj/data/input/final_test_pictures/${querynum}_*
			queryDestPath=proj/data/input/query/
			rm proj/data/input/query/*
			cp $queryRefImage $queryDestPath

			ls $queryDestPath

			echo "=================== Video file check ===================="
			videoInputPath=proj/data/input
			ls $videoInputPath/08_14_2020_${videonum}_1.mp4

			echo "=================== runInfo check ====================="
			sed -i "s/start_frame *=.*/start_frame = ${startFrame}/" $configFile
			sed -i "s/end_frame *=.*/end_frame = ${endFrame}/" $configFile
			sed -n "/^start_frame/p" $configFile
			sed -n "/^end_frame/p" $configFile

			for voteFlag in "True" "False"
			do
				sed -i "s/use_reid_voting *=.*/use_reid_voting = $voteFlag/" $configFile
				sed -n "/use_reid_voting *=.*/p" $configFile
			done

			echo "=================== TopDB check ====================="
			topdbFile=personReid/top-dropblock/top_dropblock.py
			sed -n "/ *threshold = .*/p" $topdbFile
			echo " * model use: no"
		
#    			python accuracy_check/run_accuracy.py > $resultPath"reid_acc_${model}_P${i}.txt"
		done
done
