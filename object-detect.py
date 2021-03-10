import os
import time
import shutil

import asposeimagingcloud.models.requests as requests

from imaging_base import ImagingBase

from asposeimagingcloud import ImagingApi


class ObjectDetectionImage(ImagingBase):
    """ObjectDetection image example"""
    def __init__(self, imaging_api):
        ImagingBase.__init__(self, imaging_api)
        self._print_header('Object detection image example:')

    def _get_sample_image_file_name(self):
        return 'new-orleans-3.jpg'

    def detect_objects_image_from_request_body(self, img_name, output_image_name):
        """detected object on an image that is passed in a request stream"""
        print('detected object on an image that is passed in a request stream')

        method = 'ssd'
        threshold = 50
        includeLabel = True
        includeScore = True
        allowedLabels = "person"
        blockedLabels = ""
        input_stream = img_name # os.path.join("./detection/", img_name)
        
        outPath = output_img_name
        storage = None  # We are using default Cloud Storage

        request = requests.CreateObjectBoundsRequest(input_stream, method, threshold,
                                               includeLabel, includeScore, allowedLabels, blockedLabels, outPath, storage)

        # print('Call CreateObjectBoundsRequest with params: method: {0}, threshold: {1}, includeLabel: {2}, includeScore: {3}'.format(method, threshold, includeLabel, includeScore))

        detectedObjectsList = self._imaging_api.create_object_bounds(request)
        person_count = len(detectedObjectsList.detected_objects)
        print('objects detected: {0}'.format(person_count) )
        # print(detectedObjectsList)

        return person_count

    def visualize_detect_objects_image_from_request_body(self, img_name, output_image_name):
        """Visualize detected object on an image that is passed in a request stream."""
        print('Visualize detected object on an image that is passed in a request stream')

        def write_to_file(file_path, file_name):
            shutil.copy(file_name, file_path)
            
            print('Image ' + ' is saved to ' + os.path.dirname(file_path))


        method = 'ssd'
        threshold = 50
        includeLabel = True
        includeScore = True
        allowedLabels = "person"
        blockedLabels = "dog"
        color = None
        input_stream = img_name
        outPath = None
        storage = None  # We are using default Cloud Storage

        request = requests.CreateVisualObjectBoundsRequest(input_stream, method, threshold,
                                               includeLabel, includeScore, allowedLabels, blockedLabels, color, outPath, storage)

        print('Call CreateVisualObjectBoundsRequest with params: method: {0}, threshold: {1}, includeLabel: {2}, includeScore: {3}, color: {4}'.format(method, threshold, includeLabel, includeScore, color))

        updated_image = self._imaging_api.create_visual_object_bounds(request)
        
        # self._save_updated_sample_image_to_output(updated_image, False, "jpg")
        write_to_file(output_image_name, updated_image)
        print()

        return


def split_images():
    import cv2
    vidcap = cv2.VideoCapture('./detection/mall-video-1.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        if count % 100000 == 0:
            cv2.imwrite("./detection/frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('{}. Read a new frame: '.format(count), success)
        count += 1

def split_images_2(videoFile):
    import cv2
    import math
    import numpy as np

    #################### Setting up the file ################
    
    vidcap = cv2.VideoCapture(videoFile)
    success,image = vidcap.read()

    #################### Setting up parameters ################

    seconds = 2
    fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    multiplier = round(fps * seconds)

    #################### Initiate Process ################

    while success:
        frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        success, image = vidcap.read()
        # print(frameId, multiplier)
        src_img = "./detection/frames/frame%d.jpg" % frameId
        if frameId % multiplier == 0:
            cv2.imwrite(src_img, image)
            yield src_img
    vidcap.release()
    print ("Complete")

    return ''

# split_images()


api = "asposeimagingcloudexamples.imaging_examples --clientSecret=c7b1307121677aee924c7c78e03f56fc --clientId=d3ea372d-dd02-4f91-aa8b-7c14fe20ad74"
imaging_api = ImagingApi('c7b1307121677aee924c7c78e03f56fc', 'd3ea372d-dd02-4f91-aa8b-7c14fe20ad74')
object_detect = ObjectDetectionImage(imaging_api)

videoFile = "./detection/stctrim.mp4"

img_src = './detection/frames/frame60.jpg'

def detectObjects(img_src):
    max_count = 0
    prev_max = 0
    avg = 0.0
    for img in split_images_2(img_src):
        print("Detecting: ", img)
        img_dest = "./detection/output/detected-"+os.path.basename(img)

        tic = time.perf_counter()
        curr_count = object_detect.detect_objects_image_from_request_body(img, img_dest)
        toc = time.perf_counter()

        avg = (avg + curr_count) // 2

        max_count = max(max_count, curr_count)

        if prev_max != max_count:
            print("MAX COUNT CHANGED:", max_count)
            prev_max = max_count
        print(f"RUNNING AVG: {avg} ")
        print(f"{img} TIME: {toc - tic:0.4f} seconds")

    print("MAX COUNT:", max_count)
    print(f"RUNNING AVG: {avg} ")

def viewObjects(img_src, single_img = False):
    # count = 0

    def detect_image(img):
        img_dest = "./detection/output/detected-"+os.path.basename(img)

        tic = time.perf_counter()
        curr_count = object_detect.visualize_detect_objects_image_from_request_body(img, img_dest)
        toc = time.perf_counter()
        print(f"{img} TIME: {toc - tic:0.4f} seconds")

    if single_img:
        print("Detecting: ", img_src)
        detect_image(img_src)
        return

    for img in split_images_2(img_src):
        print("Detecting: ", img)
        detect_image(img)

        # count += 1

        # if count == 3: # detecting detected images give same results :(
        #     break
        # else:
        #     images.append(img_dest)

viewObjects(videoFile, False) # sliced video detection
# viewObjects(img_src, True) # single image detection
