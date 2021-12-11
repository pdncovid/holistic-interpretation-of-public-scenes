rm -rf ./temp/*
mkdir ./temp
mkdir ./temp/detections
mkdir ./temp/groundtruths
mkdir ./temp/plots


python eval.py --testType maskAgnostic
cd ../submodules/Object-Detection-Metrics.git/
python pascalvoc.py -gtformat xyrb -t 0.00 -gt ../../eval/temp/groundtruths/ -det ../../eval/temp/detections/ -sp ../../eval/temp/plots/


echo "PROGRAM END"