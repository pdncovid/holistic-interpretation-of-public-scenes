#!/bin/bash
for i in {1..3}
do
	for j in {1..4}
	do
		SS="python Visualize.py -i data/vid-0"$i"-graph.json -p "$j" --outputPrefix visualizations/plot --onlyDetectedTime True --interpolateUndetected True"
		echo $SS
		eval $SS
	done      
done
