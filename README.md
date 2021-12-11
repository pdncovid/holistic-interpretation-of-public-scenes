# People Interaction Graph


[Read more](https://covid.eng.pdn.ac.lk/research#vision) 

## Datasets and results

[data](./data) folder contains neural network outputs and graphs for different videos.

## Quick start

The yolo human and handshake detection output files can be converted to the graph by running the following code.
```
python Scheduler.py -sg data/vid-01-graph.json --nnout_yolo data/vid-01-yolo.txt --nnout_handshake data/vid-01-handshake.json --timeSeriesLength 2006
```


## Visualization
```
python Visualize.py -i data/vid-01-graph.json -p 3 --onlyDetectedTime True --outputPrefix plot-figure-name --onlyDetectedTime True

python Visualize.py -i data/vid-01-graph.json -p 3 --onlyDetectedTime True --outputPrefix plot-figure-name --interpolateUndetected True
```


## Evaluation
```
cd eval
./eval.sh
```