# NN outputs

This folder is going to contain triplets of files for all the videos. The three files are as following,

1. **vid-01.txt**: This file should contain the location of the video file. An example is given below. The first line should contain the server name and the second line should contain the directory.
```
/storage/datasets/covid.eng.pdn.ac.lk/ltl/main-gate/04-01-2021.mp4
kepler.ce.pdn.ac.lk
```
2. **vid-01-yolo.txt**: This file is processed by [NNHandler_yolo.py](../NNHandler_yolo,py)
```
Frame #:  1
Tracker ID: 1, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (230, 127, 301, 355)
Tracker ID: 2, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (184, 42, 234, 260)
Tracker ID: 3, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (330, 46, 385, 243)
Tracker ID: 4, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (452, 133, 515, 351)
FPS: 9.08
Frame #:  2
Tracker ID: 1, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (230, 127, 302, 355)
Tracker ID: 2, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (184, 42, 234, 260)
Tracker ID: 3, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (330, 47, 385, 242)
Tracker ID: 4, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (452, 133, 514, 351)
FPS: 8.81
```
3. **vid-01-handshake.json** This file is processed by [NNHandler_handshake.py](../NNHandler_handshake.py)
```
{
	"noFrames" : 10,
	"0" : {

		"noBboxes" : 2,
		"Bboxes" :
			[
				{"y1": 0, "x1": 0, "y2": 0, "x2": 0, "conf": 90},
				{"y1": 0, "x1": 0, "y2": 0, "x2": 0, "conf": 90}
			]
		}

}
```

## Graph Structure

4. **vid-01-graph.json** This file is processed by [Graph.py](../Graph.py)

Consider an example graph where 
- we have only 1 person and 4 video frames.
- The person is sized (300,50) and paced on leftTop=(100,201) rightBottom=(201,251)
- S/he moves 6 pixels in the x and y directions each within the 4 frames. 
- The person is properly detected in all the frames except the second one.
- @suren3141 or @jameelhassan complete the part about the handshake ;-)

```
{
	"N" : 1,
	"frames" : 4,
	"nodes" : [
		{
			"xMin" : [100,102,104,106],
			"yMin" : [201,203,205,206],
			"xMax" : [400,402,404,406],
			"yMax" : [251,253,255,256],
			"detection" : [true, false, true, true],
			"handshake" : [
				{
					"person" : null,
					"confidence" : null
				}
			] 
		},

		{
			#Second person here
		}
	]
}
```
