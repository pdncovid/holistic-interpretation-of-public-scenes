# People Interaction Graph

## Abstract

The COVID-19 pandemic has caused an unprecedented global public health crisis. Given its inherent nature, social distancing measures are proposed as the primary strategies to curb the spread of this pandemic. Therefore, identifying situations where these protocols are violated, has implications for curtailing the spread of the disease and promoting a sustainable lifestyle. This paper proposes a novel computer vision-based system to analyze CCTV footage to provide a threat level assessment of COVID-19 spread. The system strives to holistically capture and interpret the information content of CCTV footage spanning multiple frames to recognize instances of various violations of social distancing protocols, across time and space, as well as identification of group behaviors. This functionality is achieved primarily by utilizing a temporal graph-based structure to represent the information of the CCTV footage and a strategy to holistically interpret the graph and quantify the threat level of the given scene. The individual components are tested and validated on a range of scenarios and the complete system is tested against human expert opinion. The results reflect the dependence of the threat level on people, their physical proximity, interactions, protective clothing, and group dynamics. The system performance has an accuracy of 76%, thus enabling a deployable threat monitoring system in cities, to permit normalcy and sustainability in the society.


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

## Publications
This repository contains the codebase for

Gihan Jayatilaka\*, Jameel Hassan\*, Suren Sritharan\*, Janith Bandara Senananayaka, Harshana Weligampola, Roshan Godaliyadda, Parakrama Ekanayake, Vijitha Herath,Janaka Ekanayake, Samath Dharmaratne, 2021. **Holistic Interpretation of Public Scenes Using Computer Vision and Temporal Graphs to Identify Social Distancing Violations**. *arXiv preprint*.

\[[Preprint (PDF arXiv:2112.06428)](https://arxiv.org/pdf/2112.06428)\]

\* Equally contributing authors.


You may cite this work as
```
@misc{holistic-interpretation-of-public-scenes-2021,
      title={Holistic Interpretation of Public Scenes Using Computer Vision and Temporal Graphs to Identify Social Distancing Violations},
      author={Gihan Jayatilaka and Jameel Hassan and Suren Sritharan and Janith Bandara Senananayaka and Harshana Weligampola and Roshan Godaliyadda and Parakrama Ekanayake and Vijitha Herath and Janaka Ekanayake and Samath Dharmaratne},
      year={2021},
      eprint={2112.06428},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Another conference paper generated out of this work is

Jameel Hassan, Suren Sritharan, Gihan Jayatilaka, Roshan Godaliyadda, Parakrama Ekanayake, Vijitha Herath, Janaka Ekanayake, 2021. **Hands Off: A Handshake Interaction Detection and Localization Model for COVID-19 Threat Control**. In *2019 14th Conference on Industrial and Information Systems (ICIIS) (pp. 260-265). IEEE*.

\[[Preprint (PDF arXiv:2110.0957)](https://arxiv.org/pdf/2110.09571.pdf), [Presentation (PDF)](https://gihan.me/projects/covid/iciis-2021-presentation.pdf), [Presentation (Youtube)](https://youtu.be/oLd0oU_Tiyg)\]