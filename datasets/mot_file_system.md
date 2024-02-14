# multi-view data
All COCO classes, not just relevant 

# MOT17 and MOT20
## annotations
<frame_number>, <track_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence_score>, <class_id>, <visibility_score>
- <confidence_score>
    - DET: Indicates how confident the detector is that this instance is a pedestrian.
    - GT: It acts as a flag whether the entry is to be considered (1) or ignored (0).
- <class_id>
    - GT: Indicates the type of object annotated
- <visibility_score>
    - GT: Visibility ratio, a number between 0 and 1 that says how much of that object is visible. Can be due
to occlusion and due to image border cropping.

|class|id|
|:---:|:---:|
|Pedestrian| 1|
|Person on vehicle| 2|
|Car| 3|
|Bicycle| 4|
|Motorbike| 5|
|Non motorized vehicle| 6|
|Static person| 7|
|Distractor| 8|
|Occluder| 9|
|Occluder on the ground| 10|
|Occluder full| 11|
|Reflection| 12|
|Crowd| 13|

## folder structure
```
mot
├── annotations -> COCO annotation files; automatically generated from mot annotations; one class - pedestrian 
│   ├── test.json
│   ├── train_half.json
│   ├── train.json
│   └── val_half.json
├── test
│   ├── MOT17-01-DPM -> root folder for a sequence
│   │   ├── det -> detection annotations, no track ids
│   │   ├── img1 -> images in the video sequence
│   │   └── seqinfo.ini -> sequence metadata
│   ├── MOT17-01-FRCNN
│   ├── MOT17-01-SDP
│   ├── MOT17-03-DPM
│   ├── MOT17-03-FRCNN
│   ├── MOT17-03-SDP
│   ├── MOT17-06-DPM
│   ├── MOT17-06-FRCNN
│   ├── MOT17-06-SDP
│   ├── MOT17-07-DPM
│   ├── MOT17-07-FRCNN
│   ├── MOT17-07-SDP
│   ├── MOT17-08-DPM
│   ├── MOT17-08-FRCNN
│   ├── MOT17-08-SDP
│   ├── MOT17-12-DPM
│   ├── MOT17-12-FRCNN
│   ├── MOT17-12-SDP
│   ├── MOT17-14-DPM
│   ├── MOT17-14-FRCNN
│   └── MOT17-14-SDP
└── train
    ├── MOT17-02-DPM
    │   ├── det -> detection annotations, no track ids
    │   ├── gt -> detection + tracking annotations
    │   ├── img1
    │   └── seqinfo.ini
    ├── MOT17-02-FRCNN
    ├── MOT17-02-SDP
    ├── MOT17-04-DPM
    ├── MOT17-04-FRCNN
    ├── MOT17-04-SDP
    ├── MOT17-05-DPM
    ├── MOT17-05-FRCNN
    ├── MOT17-05-SDP
    ├── MOT17-09-DPM
    ├── MOT17-09-FRCNN
    ├── MOT17-09-SDP
    ├── MOT17-10-DPM
    ├── MOT17-10-FRCNN
    ├── MOT17-10-SDP
    ├── MOT17-11-DPM
    ├── MOT17-11-FRCNN
    ├── MOT17-11-SDP
    ├── MOT17-13-DPM
    ├── MOT17-13-FRCNN
    └── MOT17-13-SDP
```