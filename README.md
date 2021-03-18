# 41X-simulation
Simulation For Object detection for occupancy system

For running code, Please switch to `master` branch.

### Installations required:

Both requires:
- Python Installation at [Python3.8](https://www.python.org/downloads/release/python-388/)

Mask RCNN
- `pip3 install tensorflow==2.4.1`
- `pip3 install pixellib`

SSD
- `pip3 install aspose-imaging-cloud`

Live Capture (Mask RCNN only for now)
- `pip3 install selenium`
- NOTE: Might need to install Chrome webdriver for newer versions of Google Chrome. The one in the folder uses v88.x

Postgres SQL DB storage
- `brew install postgresql`
- `pip install psycopg2`
- `pip install psycopg2-binary`
- NOTE: if the first fails try second one and check if psycopg2 is installed by following Terminal commands:
```
python
# type python in terminal
>>> import psycopg2
>>> # if it errors then unsuccessful else successful
```

Visualization
- `brew install grafana`
- use the [quick start guide](https://grafana.com/docs/grafana/latest/getting-started/getting-started/)

### How to Run Mask RCNN:

1. Download [Mask RCNN model](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5)

2. Put model on same level as `object-detection-combined.py`

3. `python3 object-detection-combined.py` on Terminal. NOTE: Here frames DO NOT get stored in `./detection/frames/`.

4. OPTIONAL: In the code change the video path to the place where your video to use your own video.

### How to Run SSD:

1. `python3 object-detection.py` on Terminal

2. Frames will be generated and put inside the `./detection/frames/` directory. Frames are the individual video slices.

### Outputs:

1. Find detected images in `./detection/output/` directory

### Observations:

1. SSD: Fast but low accuracy

2. Mask RCNN: Slow but high accuracy

3. We should go with Mask RCNN even though it is not fast because of much higher precision.


