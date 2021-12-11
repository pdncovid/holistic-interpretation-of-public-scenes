rm -rf ./temp/*
mkdir ./temp
mkdir ./temp/detections
mkdir ./temp/groundtruths
mkdir ./temp/plots
python eval.py

cd ../submodules/Object-Detection-Metrics.git/
python pascalvoc.py -gtformat xyrb -t 0.5 -gt ../../eval/temp/groundtruths/ -det ../../eval/temp/detections/ -sp ../../eval/temp/plots/
echo "PROGRAM END"