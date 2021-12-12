rm -rf ./temp/*
mkdir ./temp
mkdir ./temp/detection-results
mkdir ./temp/ground-truth


python eval.py --testType maskAgnostic


cd ../submodules/mAP/
rm -rf input/*
cp -r ../../eval/temp/detection-results/ ./input/
cp -r ../../eval/temp/ground-truth/ ./input/

python main.py -np -na


# cd ../submodules/Object-Detection-Metrics.git/
# python pascalvoc.py -gtformat xyrb -t 0.00 -gt ../../eval/temp/groundtruths/ -det ../../eval/temp/detections/ -sp ../../eval/temp/plots/


echo "PROGRAM END"