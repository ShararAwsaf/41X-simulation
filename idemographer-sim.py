import time
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg

import cv2
import math
import numpy as np

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


import psycopg2
import sys
import os

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

DATA_PATH = os.path.join('.', "detection")
STC_VIDEO_FILE = os.path.join(DATA_PATH, "stctrim.mp4")
IMAGE_FORMAT = "png"
OUTPUT_PATH = os.path.join(DATA_PATH, "output")
WEBDRIVER_PATH = os.path.join("drivers", "chromedriver")

RCNN_MODEL_H5 = os.path.join("models", "mask_rcnn_coco.h5")
VIDEO_DETECT_DELAY = 1
NUMBER_OF_SAMPLES = 5

MODEL = None
BLUR_MODEL = None

def setup_db():

    connection, cursor = None, None
    logger.info("Connecting to Database...")
    try:
        start_time = time.perf_counter()
        connection = psycopg2.connect(user="postgres",
                                    password="pgpass",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="postgres")
        cursor = connection.cursor()
        finish_time = time.perf_counter()

        logger.info(f"Database Connection Time: {finish_time - start_time:0.4f} seconds")
        
    except (Exception, psycopg2.Error) as error:
        logger.info("Failed to Establish connection", error)

    return connection, cursor


def insert_to_occupancy_table(connection, cursor, loc, cam, area, occ):
    try:

        postgres_insert_query = """ INSERT INTO occupancy_table (LOCATION, CAMERA, AREA, OCCUPANCY) VALUES (%s,%s,%s,%s)"""
        record_to_insert = (loc, cam, area, occ)
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()

        row_count = cursor.rowcount
        logger.info("Record inserted successfully into occupancy_table table")
        logger.info(f"Total rows: {row_count}")

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into occupancy_table", error)

    
def initialize_model():
    start_time = time.perf_counter()

    segment_image = instance_segmentation(infer_speed="average")
    segment_image.load_model(RCNN_MODEL_H5)

    finish_time = time.perf_counter()

    logger.info(f"Initialization Time: {finish_time - start_time:0.4f} seconds")
    
    return segment_image


def video_detect(videoFile): 
    vidcap = cv2.VideoCapture(videoFile)
    is_successful, image = vidcap.read()

    capture_period_seconds = 2
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = round(fps * capture_period_seconds)

    max_count = 0
    prev_max = 0
    running_avg = 0.0

    while is_successful:
        frameId = int(round(vidcap.get(1))) # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough

        is_successful, image = vidcap.read()

        if frameId % total_frames == 0:
            src_img = os.path.join("detection", "frames", f"frame{frameId}.jpg")
            start_time = time.perf_counter()

            segmask, output = MODEL.segmentFrame(image, 
                                    show_bboxes = True, 
                                    segment_target_classes=MODEL.select_target_classes(person=True), 
                                    extract_segmented_objects= False, 
                                    save_extracted_objects=False,
                                    output_image_name = os.path.join(OUTPUT_PATH, f"{frameId}-person-only.jpg")
                                    )

            finish_time = time.perf_counter()

            bounding_box = len(segmask['rois'])

            running_avg = (running_avg + bounding_box) // 2

            logger.info(f"Frame ID:{frameId} Bounding Box Coordinates: {bounding_box}")
            logger.info(f"Frame ID:{frameId} Class IDs: {len(segmask['class_ids'])}")

            logger.info(f"Frame ID:{frameId} Running Average: {running_avg}")
            logger.info(f"Frame ID:{frameId} Compute Time: {finish_time - start_time:0.4f} seconds")

            max_count = max(max_count, bounding_box)

            if prev_max <= max_count:
                logger.info(f"New maximum count: {max_count}")
                prev_max = max_count

    vidcap.release()

    logger.info("Finished")
    logger.info(f"Max Count: {max_count}")


def initialize_model_blur():
    change_bg = alter_bg(model_type = "pb")
    change_bg.load_pascalvoc_model("./models/xception_pascalvoc.pb")
    return change_bg

def image_blur(image_path, output_path=OUTPUT_PATH, tag=None):
    if not tag:
        tag = "blur-"
    out_path = os.path.join(output_path, tag+os.path.basename(image_path))
    print(f"Changing {image_path} Background and storing at {out_path}")
    # image_path = './detection/stc-1.png'
    # output_path = "./detection/output/stc-1-blur.png"
    BLUR_MODEL.blur_bg(image_path, low=True, detect="person", output_image_name=out_path )
    

def image_detect(image_path, output_path=OUTPUT_PATH):
    target_classes = MODEL.select_target_classes(person=True)
    
    start_time = time.perf_counter()

    segmask, output = MODEL.segmentImage(image_path, 
                            show_bboxes = True, 
                            segment_target_classes=target_classes, 
                            extract_segmented_objects= False, 
                            save_extracted_objects=False,
                            output_image_name = os.path.join(output_path, os.path.basename(image_path))
                            )

    finish_time = time.perf_counter()

    bounding_box = len(segmask['rois'])

    logger.info(f"Bounding Box Coordinates: {bounding_box}")
    logger.info(f"Class IDs: {len(segmask['class_ids'])}")
    logger.info(f"Compute Time: {finish_time - start_time:0.4f} seconds")

    return bounding_box


def live_detect_earth_cam(url, tag):
    
    options = Options()
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(WEBDRIVER_PATH, chrome_options=options)
    driver.get(url)
    # We need this sleep to wait for the ad to complete
    button_clicked = False
    
    while not button_clicked:
        try:
            button = driver.find_element_by_class_name('fullScreenBtn')
            button.click()
            button_clicked = True
        except Exception as e:
            logger.info(f"Button pressing failed : {e}")

    logger.info(f"Capturing from {url}...")

    running_avg = 0.0
    num_samples = NUMBER_OF_SAMPLES
    live_capture_path = os.path.join(DATA_PATH, "live")
    live_output_path = os.path.join(OUTPUT_PATH, "live")

    for i in range(0, num_samples):
        image_path = os.path.join(live_capture_path, f"{tag}-{i}.{IMAGE_FORMAT}")
        driver.save_screenshot(image_path)
        
        curr_count = image_detect(image_path, live_output_path)
        running_avg = (running_avg + curr_count) // 2

        logger.info(f"Running Average: {running_avg}")

        yield curr_count

        # time.sleep(VIDEO_DETECT_DELAY)

    driver.close()


def live_detect_insecam(url, tag):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')

    driver = webdriver.Chrome(WEBDRIVER_PATH, chrome_options=options)
    driver.get(url)

    # Wait for ads and use full screen
    time.sleep(15)
    button = driver.find_element_by_id('playIcon')
    button.click()
    time.sleep(1)
    button = driver.find_element_by_css_selector("[title*='Full screen']")
    button.click()
    time.sleep(2)

    logger.info(f"Capturing from {url}...")

    running_avg = 0.0

    for i in range(0, num_samples):
        image_path = f"{live_capture_path}-{i}.png"
        driver.save_screenshot(image_path)
        
        curr_count = image_detect(image_path)
        running_avg = (running_avg + curr_count) // 2

        logger.info(f"Running Average: {running_avg}")

        time.sleep(VIDEO_DETECT_DELAY)

    driver.close()

def driver():
    detection_type = "video"
    # conn, cur = setup_db()

    if len(sys.argv) > 1:
        detection_type = sys.argv[1]

    logger.info(f"Using detection type: {detection_type}")
    
    if detection_type == "image":
        image = "sample_1.jpeg"
        if len(sys.argv) > 2:
            image = sys.argv[2]

        sample_image = os.path.join(DATA_PATH, image)
        image_detect(sample_image)
    elif detection_type == "video":
        
        video = STC_VIDEO_FILE
        if len(sys.argv) > 2:
            video = sys.argv[2]

        sample_video = os.path.join(DATA_PATH, video)
        video_detect(sample_video)

    elif detection_type == "earthcam":
        if len(sys.argv) != 6:
            logger.info("Please provide location, camera name, area, earthcam URL in order when using earthcam detection type")
        else:            
            location = sys.argv[2]
            cam_num = sys.argv[3]
            area = sys.argv[4]
            tag =  f"{location}-{cam_num}".replace(" ", "_").lower()

            # url = "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=bourbonstreet"
            url = sys.argv[5]

            connection, cursor = setup_db()
            for o in live_detect_earth_cam(url, tag):
                insert_to_occupancy_table(connection, cursor, loc=location, cam=cam_num, area=area, occ=o)
                

    elif detection_type == "insecam":
        pass
        location = 'Toronto'
        cam_num = 'Cam1'
        tag =  f"{location}-{cam_num}".replace(" ", "_").lower()
        url = "http://www.insecam.org/en/view/237860/"

        live_detect_insecam(url, tag)
    else:
        logger.info("Unknown detection type provided. Please use one of (image|video|earthcam)")


if __name__ == "__main__":

    # BLUR_MODEL = initialize_model_blur()
    # image_blur(sys.argv[1])
    MODEL = initialize_model()
    driver()
