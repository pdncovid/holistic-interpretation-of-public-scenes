rm -rf ./temp/*
mkdir ./temp
mkdir ./temp/detections
mkdir ./temp/groundtruths
python eval.py

cd ../submodules/Object-Detection-Metrics.git/
python pascalvoc.py -gtformat xyrb  -gt ../../eval/temp/groundtruths/ -det ../../eval/temp/detections/
echo "PROGRAM END"