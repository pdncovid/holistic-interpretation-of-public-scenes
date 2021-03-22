# NN outputs

This folder is going to contain triplets of files for all the videos. The three files are as following,

1. **vid-01.txt**: This file should contain the location of the video file. An example is given below. The first line should contain the server name and the second line should contain the directory.
```
/storage/datasets/covid.eng.pdn.ac.lk/ltl/main-gate/04-01-2021.mp4
kepler.ce.pdn.ac.lk
```
2. **vid-01-yolo.json**
```
{
    "N": 1,
    "nodes": [
        {
            "xMin": [
                0,
                302.5,
                302.0,
            ],
            "xMax": [
                0,
                1080,
                1080,
            ],
            "yMin": [
                0,
                302.5,
                302.0,
            ],
            "yMax": [
                0,
                1080,
                1080,
            ],

		    "detection": [
                false,
                true,
                true,
            ],
        }
    ]
}
```
3. **vid-01-handshake.json**
```
{
	"noFrames" : 10,
	"0" : {

		"noBboxes" : 2,
		"Bboxes" :
			[
				{"x1": 0, "x2": 0, "y1": 0, "y2": 0, "conf": 90},
				{"x1": 0, "x2": 0, "y1": 0, "y2": 0, "conf": 90}
			]
		}

}
```