import os

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

    def detect_objects_image_from_request_body(self):
        """detected object on an image that is passed in a request stream"""
        print('detected object on an image that is passed in a request stream')

        method = 'ssd'
        threshold = 50
        includeLabel = True
        includeScore = True
        allowedLabels = "person"
        blockedLabels = "dog"
        input_stream = os.path.join("./detection/", self._get_sample_image_file_name())
        outPath = None
        storage = None  # We are using default Cloud Storage

        request = requests.CreateObjectBoundsRequest(input_stream, method, threshold,
                                               includeLabel, includeScore, allowedLabels, blockedLabels, outPath, storage)

        print('Call CreateObjectBoundsRequest with params: method: {0}, threshold: {1}, includeLabel: {2}, includeScore: {3}'.format(method, threshold, includeLabel, includeScore))

        detectedObjectsList = self._imaging_api.create_object_bounds(request)
        
        print('objects detected: {0}'.format(len(detectedObjectsList.detected_objects)))
        print(detectedObjectsList)

    def detect_objects_image_from_request_body_bbox(self):
        import shutil
        """detected object on an image that is passed in a request stream"""
        print('detected object on an image that is passed in a request stream')

        method = 'ssd'
        threshold = 50
        includeLabel = True
        includeScore = True
        allowedLabels = "person"
        blockedLabels = "dog"
        input_stream = os.path.join("./detection/", self._get_sample_image_file_name())
        outPath = None
        storage = None  # We are using default Cloud Storage

        request = requests.CreateObjectBoundsRequest(input_stream, method, threshold,
                                               includeLabel, includeScore, allowedLabels, blockedLabels, outPath, storage)

        print('Call CreateObjectBoundsRequest with params: method: {0}, threshold: {1}, includeLabel: {2}, includeScore: {3}'.format(method, threshold, includeLabel, includeScore))

        detectedObjectsList = self._imaging_api.create_object_bounds(request)
        print('objects detected: {0}'.format(len(detectedObjectsList.detected_objects)))
        print(detectedObjectsList)

        # updated_image = self._imaging_api.create_visual_object_bounds(request)
        # image_name = 'detected-image.jpg'
        # path = os.path.abspath(os.path.join('./detection/', image_name))
        # shutil.copy(updated_image, path)
        # print('Image ' + image_name + ' is saved to ' + os.path.dirname(path))
        # print()

        

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

def split_images_2():
    import cv2
    import math
    import numpy as np

    #################### Setting up the file ################
    videoFile = "./detection/mall-video-1.mp4"
    vidcap = cv2.VideoCapture(videoFile)
    success,image = vidcap.read()

    #################### Setting up parameters ################

    seconds = 3
    fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    multiplier = round(fps * seconds)

    #################### Initiate Process ################

    while success:
        frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        success, image = vidcap.read()
        print(frameId, multiplier)
        if frameId % multiplier == 0:
            cv2.imwrite("./detection/frames/frame%d.jpg" % frameId, image)

    vidcap.release()
    print ("Complete")


# split_images()
split_images_2()

# api = "asposeimagingcloudexamples.imaging_examples --clientSecret=c7b1307121677aee924c7c78e03f56fc --clientId=d3ea372d-dd02-4f91-aa8b-7c14fe20ad74"
# imaging_api = ImagingApi('c7b1307121677aee924c7c78e03f56fc', 'd3ea372d-dd02-4f91-aa8b-7c14fe20ad74')
# object_detect = ObjectDetectionImage(imaging_api)
# object_detect.detect_objects_image_from_request_body_bbox()
