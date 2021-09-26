#! /bin/sh

accFile=accuracy_check/run_accuracy.py
configFile=configs/runInfo.py

resultPath=final_result/
modelPath=personReid/top-dropblock/weight/
modelName=model.pth.tar-400

#startFrame=631
#endFrame=3571
for videonum in {1..4}
do
	for querynum in {1..8}
	do
		for voteFlag in "True" "False"
		do
    			echo "============= Reid accucracy check #video[$videonum] #q[$querynum] #vote[$voteFlag] ================="
			resultFile=$resultPath"reid_vid$videonum""_q$querynum""_vt$voteFlag"".txt"

			echo "============= Video file check" >> $resultFile
			videoInputPath=proj/data/input
			ls $videoInputPath/08_14_2020_${videonum}_1.mp4
			echo " - Input Video name: 08_14_2020_${videonum}_1.mp4" >> $resultFile


			echo "============= run_accuracy check" >> $resultFile

    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = True/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${querynum}'/" $accFile
    			sed -n "/TEST_BBOX =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_REID =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_MASK =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_SYSTEM =.*/p" $accFile >> $resultFile

    			sed -n "/WRITE_VIDEO =.*/p" $accFile >> $resultFile

			sed -n "/QUERY_GROUND_TRUTH =.*/p" $accFile >> $resultFile

			
			echo "============= Query image check"

			queryRefImage=proj/data/input/final_test_pictures/${querynum}_*
			queryDestPath=proj/data/input/query/
			rm proj/data/input/query/*
			cp $queryRefImage $queryDestPath

			ls $queryDestPath

			echo "============= runInfo check" >> $resultFile

			#sed -i "s/start_frame *=.*/start_frame = ${startFrame}/" $configFile
			#sed -i "s/end_frame *=.*/end_frame = ${endFrame}/" $configFile
			sed -n "/^start_frame/p" $configFile >> $resultFile

			sed -n "/^end_frame/p" $configFile >> $resultFile


			sed -i "s/use_mask_voting *=.*/use_mask_voting = False/" $configFile
			sed -n "/use_mask_voting *=.*/p" $configFile >> $resultFile

			sed -i "s/use_reid_voting *=.*/use_reid_voting = $voteFlag/" $configFile
			sed -n "/use_reid_voting *=.*/p" $configFile >> $resultFile


			echo "============= TopDB check"
			topdbFile=personReid/top-dropblock/top_dropblock.py
			sed -n "/ *threshold = .*/p" $topdbFile >> $resultFile

			echo " model use: no" >> $resultFile

		
    			python accuracy_check/run_accuracy.py >> $resultFile

		done
		for voteFlag in "True" "False"
		do
    			echo "============= System accucracy check #video[$videonum] #q[$querynum] #vote[$voteFlag] ================="
			resultFile=$resultPath"system_vid$videonum""_q$querynum""_vt$voteFlag"".txt"
			echo "============= Query image check"

			queryRefImage=proj/data/input/final_test_pictures/${querynum}_*
			queryDestPath=proj/data/input/query/
			rm proj/data/input/query/*
			cp $queryRefImage $queryDestPath

			ls $queryDestPath


			echo "============= Video file check" >> $resultFile
			videoInputPath=proj/data/input
			ls $videoInputPath/08_14_2020_${videonum}_1.mp4
			echo " - Input Video name: 08_14_2020_${videonum}_1.mp4" >> $resultFile


			echo "============= run_accuracy check" >> $resultFile

    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = False/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = True/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${querynum}'/" $accFile
    			sed -n "/TEST_BBOX =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_REID =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_MASK =.*/p" $accFile >> $resultFile

    			sed -n "/TEST_SYSTEM =.*/p" $accFile >> $resultFile

    			sed -n "/WRITE_VIDEO =.*/p" $accFile >> $resultFile

			sed -n "/QUERY_GROUND_TRUTH =.*/p" $accFile >> $resultFile

			
			echo "============= runInfo check" >> $resultFile

			#sed -i "s/start_frame *=.*/start_frame = ${startFrame}/" $configFile
			#sed -i "s/end_frame *=.*/end_frame = ${endFrame}/" $configFile
			sed -n "/^start_frame/p" $configFile >> $resultFile
			sed -n "/^end_frame/p" $configFile >> $resultFile

			sed -i "s/use_mask_voting *=.*/use_mask_voting = $voteFlag/" $configFile
			sed -n "/use_mask_voting *=.*/p" $configFile >> $resultFile

			sed -i "s/use_reid_voting *=.*/use_reid_voting = $voteFlag/" $configFile
			sed -n "/use_reid_voting *=.*/p" $configFile >> $resultFile


			echo "============= TopDB check"
			topdbFile=personReid/top-dropblock/top_dropblock.py
			sed -n "/ *threshold = .*/p" $topdbFile >> $resultFile

			echo " model use: no" >> $resultFile

    			python accuracy_check/run_accuracy.py >> $resultFile

		done

	done
done
