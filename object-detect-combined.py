import time
import pixellib
from pixellib.instance import instance_segmentation

import cv2
import math
import numpy as np

tic = time.perf_counter()
segment_image = instance_segmentation(infer_speed="average") # infer_speed = "average"
segment_image.load_model("mask_rcnn_coco.h5")
target_classes = segment_image.select_target_classes(person=True)
toc = time.perf_counter()
print(f"INITIALIZING TIME: {toc - tic:0.4f} seconds")


#################### Setting up the file ################

videoFile = "./detection/stctrim.mp4"
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()

#################### Setting up parameters ################

seconds = 2
fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = round(fps * seconds)

#################### Initiate Process ################
max_count = 0
prev_max = 0
avg = 0.0

while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()
    # print(frameId, multiplier)
    
    if frameId % multiplier == 0:
        
        src_img = "./detection/frames/frame%d.jpg" % frameId
        # cv2.imwrite(src_img, image)

        ######################### DETECTION ################################
        tic = time.perf_counter()
        segmask, output = segment_image.segmentFrame(image, 
                                show_bboxes = True, 
                                segment_target_classes=target_classes, 
                                extract_segmented_objects= False, 
                                save_extracted_objects=False,
                                output_image_name = "./detection/output/{}-person-only.jpg".format(frameId)
                                )
        toc = time.perf_counter()

        bbox = len(segmask['rois'])
        avg = (avg + bbox) // 2
        print("{} BBOX COORDINATES:".format(frameId), bbox)
        print("{} CLASS IDS:".format(frameId), len(segmask['class_ids']))

        print(f"{frameId} RUNNING AVG: {avg} ")
        print(f"{frameId} TIME: {toc - tic:0.4f} seconds")

        max_count = max(max_count, bbox)

        if prev_max != max_count:
            print("MAX COUNT CHANGED:", max_count)
            prev_max = max_count            

vidcap.release()
print ("Complete")
print("MAX COUNT:", max_count)
print(f"RUNNING AVG: {avg} ")
