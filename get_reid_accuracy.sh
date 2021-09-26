#! /bin/sh

# 주의할 점
# 1. runInfo에서 start_frame = 0, end_frame = -1로 되어 있는지 확인하기
# 2. 아래 변수들의 경로가 잘 설정되었는지 확인하기 (accFile, configFile, topdbFile, videoInputPath 경로 확인)
# 3. 결과 파일이 저장될 디렉토리를 생성하기 (resultPath에 생성한 디렉토리의 경로 적기)
# 4. 아래 영상 번호, query 번호의 범위가 잘 맞는지 확인하기 (25, 28, 93, 95 line 확인)
# 5. 되도록 nohup으로 실행하기. nohup.out파일에 system 정확도 결과가 출력으로 남기 때문. 그냥 쉘에서 돌리면 결과가 날아갈 수 있음.
# 6. 실행 경로는 우리 메인 레포임 (multi_covid19~/)
# 7. (optional) resultFile의 이름이 괜찮은지 확인하기

accFile=accuracy_check/run_accuracy.py
configFile=configs/runInfo.py
topdbFile=personReid/top-dropblock/top_dropblock.py
videoInputPath=proj/data/input 				# 입력 영상이 있는 경로
queryRefPath=proj/data/input/final_test_pictures/ 	# 모든 query 이미지를 모아둔 디렉토리 경로 (6, 7번 사람의 이미지를 모두 여기에 모아둬야 함) 
							# 	--> 여기에 있던 이미지 중, 시스템 분석할 때마다 해당 query의 이미지가 queryDestPath로 복사됨.
queryDestPath=proj/data/input/query/			# 시스템 분석할 때 사용할 query 이미지가 위치하는 디렉토리 경로

resultPath=test_result/ 				# 결과 파일을 저장할 디렉토리 경로 설정 (디렉토리 생성해야 함)
modelPath=personReid/top-dropblock/weight/
modelName=model.pth.tar-400

for videonum in {1..2}
do
	# 우선 system에 대해 query 6, 7번과 voting T/F의 정확도를 모두 돌린다.
	for querynum in {6..7}
	do
		for voteFlag in "True" "False"
		do
    			echo "============= System accucracy check #video[$videonum] #q[$querynum] #vote[$voteFlag] ================="
			resultFile=$resultPath"system_vid$videonum""_q$querynum""_vt$voteFlag"".txt"

			echo "============= Video file check" >> $resultFile
			ls $videoInputPath/08_14_2020_${videonum}_1.mp4
			echo " - Input Video name: 08_14_2020_${videonum}_1.mp4" >> $resultFile


			echo "============= run_accuracy check" >> $resultFile
			# 변수에 각 값을 설정
    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = False/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = True/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${querynum}'/" $accFile
			
			# 변수 확인용 출력
    			sed -n "/TEST_BBOX =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_REID =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_MASK =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_SYSTEM =.*/p" $accFile >> $resultFile
    			sed -n "/WRITE_VIDEO =.*/p" $accFile >> $resultFile
			sed -n "/QUERY_GROUND_TRUTH =.*/p" $accFile >> $resultFile

			echo "============= Query image check"
			queryRefImage=$queryRefPath${querynum}_*

			rm $queryDestPath*
			cp $queryRefImage $queryDestPath

			ls $queryDestPath

			echo "============= runInfo check" >> $resultFile

			# 변수 확인용 출력
			sed -n "/^start_frame/p" $configFile >> $resultFile
			sed -n "/^end_frame/p" $configFile >> $resultFile

			# use_mask_voting flag를 False로 설정
			sed -i "s/use_mask_voting *=.*/use_mask_voting = False/" $configFile
			sed -n "/use_mask_voting *=.*/p" $configFile >> $resultFile

			# use_reid_voting flag 설정
			sed -i "s/use_reid_voting *=.*/use_reid_voting = $voteFlag/" $configFile
			sed -n "/use_reid_voting *=.*/p" $configFile >> $resultFile

			echo "============= TopDB check"
			# threshold값 확인
			sed -n "/ *threshold = .*/p" $topdbFile >> $resultFile
			# weight를 사용하지 않는다는 걸 명시
			echo " model use: no" >> $resultFile

		
    			python accuracy_check/run_accuracy.py >> $resultFile

		done
	done
done

# 다음으로 reid에 대해 query 6, 7번과 voting T/F의 정확도를 모두 돌린다.
for videonum in {1..2} 
do
	for querynum in {6..7}
	do
		for voteFlag in "True" "False"
		do
    			echo "============= Reid accucracy check #video[$videonum] #q[$querynum] #vote[$voteFlag] ================="
			resultFile=$resultPath"reid_vid$videonum""_q$querynum""_vt$voteFlag"".txt"

			echo "============= Video file check" >> $resultFile
			ls $videoInputPath/08_14_2020_${videonum}_1.mp4
			echo " - Input Video name: 08_14_2020_${videonum}_1.mp4" >> $resultFile


			echo "============= run_accuracy check" >> $resultFile
			# 변수에 각 값을 설정
    			sed -i "s/TEST_BBOX =.*/TEST_BBOX = False/" $accFile
    			sed -i "s/TEST_REID =.*/TEST_REID = True/" $accFile
    			sed -i "s/TEST_MASK =.*/TEST_MASK = False/" $accFile
    			sed -i "s/TEST_SYSTEM =.*/TEST_SYSTEM = False/" $accFile
    			sed -i "s/WRITE_VIDEO =.*/WRITE_VIDEO = False/" $accFile
			sed -i "s/QUERY_GROUND_TRUTH =.*/QUERY_GROUND_TRUTH = 'P${querynum}'/" $accFile
			
			# 변수 확인용 출력
    			sed -n "/TEST_BBOX =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_REID =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_MASK =.*/p" $accFile >> $resultFile
    			sed -n "/TEST_SYSTEM =.*/p" $accFile >> $resultFile
    			sed -n "/WRITE_VIDEO =.*/p" $accFile >> $resultFile
			sed -n "/QUERY_GROUND_TRUTH =.*/p" $accFile >> $resultFile

			echo "============= Query image check"
			queryRefImage=$queryRefPath${querynum}_*

			rm $queryDestPath*
			cp $queryRefImage $queryDestPath

			ls $queryDestPath

			echo "============= runInfo check" >> $resultFile

			# 변수 확인용 출력
			sed -n "/^start_frame/p" $configFile >> $resultFile
			sed -n "/^end_frame/p" $configFile >> $resultFile

			# use_mask_voting flag를 False로 설정
			sed -i "s/use_mask_voting *=.*/use_mask_voting = False/" $configFile
			sed -n "/use_mask_voting *=.*/p" $configFile >> $resultFile

			# use_reid_voting flag 설정
			sed -i "s/use_reid_voting *=.*/use_reid_voting = $voteFlag/" $configFile
			sed -n "/use_reid_voting *=.*/p" $configFile >> $resultFile

			echo "============= TopDB check"
			# threshold값 확인
			sed -n "/ *threshold = .*/p" $topdbFile >> $resultFile
			# weight를 사용하지 않는다는 걸 명시
			echo " model use: no" >> $resultFile

		
    			python accuracy_check/run_accuracy.py >> $resultFile

		done
	done
done
