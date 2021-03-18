import time
import pixellib
from pixellib.instance import instance_segmentation

import cv2
import math
import numpy as np

from selenium import webdriver

import psycopg2
def setup_db():
    connection, cursor = None, None
    try:
        connection = psycopg2.connect(user="postgres",
                                    password="pgpass",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="postgres")
        cursor = connection.cursor()

        
    except (Exception, psycopg2.Error) as error:
        print("Failed to Establish connection", error)

    return connection, cursor

def insert_to_occupancy_table(connection, cursor, loc, cam, area, occ):
    try:

        postgres_insert_query = """ INSERT INTO occupancy_table (LOCATION, CAMERA, AREA, OCCUPANCY) VALUES (%s,%s,%s,%s)"""
        record_to_insert = (loc, cam, area, occ)
        cursor.execute(postgres_insert_query, record_to_insert)

        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into mobile table")

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into occupancy_table", error)


def initialize_model():
    tic = time.perf_counter()
    segment_image = instance_segmentation(infer_speed="average") # infer_speed = "average"
    segment_image.load_model("mask_rcnn_coco.h5")
    toc = time.perf_counter()
    print(f"INITIALIZING TIME: {toc - tic:0.4f} seconds")
    return segment_image

segment_image = initialize_model()

def video_detect(videoFile): 

    #################### Setting up the file ################

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

def image_detect(path, image, frmt, output_path):
    print(f"STARTING DETECTION: {image}...")
    target_classes = segment_image.select_target_classes(person=True)
    
    tic = time.perf_counter()
    segmask, output = segment_image.segmentImage(path+image+frmt, 
                            show_bboxes = True, 
                            segment_target_classes=target_classes, 
                            extract_segmented_objects= False, 
                            save_extracted_objects=False,
                            output_image_name = "{}{}-person-only.jpg".format(output_path, image)
                            )
    toc = time.perf_counter()
    bbox = len(segmask['rois'])

    print("{} BBOX COORDINATES:".format(image), bbox)
    print("{} CLASS IDS:".format(image), len(segmask['class_ids']))
    print(f"{image} TIME: {toc - tic:0.4f} seconds")

    return bbox

def live_detect_earth_cam(website, tag, delay=0):

    driver = webdriver.Chrome('./chromedriver')
    driver.get(website)
    time.sleep(20)
    
    button = driver.find_element_by_class_name('fullScreenBtn')
    button.click()
    

    print(f"CAPTURING IMAGES FROM: {website}.....")
    count = 0
    avg = 0.0
    while count <= 5:
        count += 1
        path = "./detection/live/"
        image = '{}-{}'.format(tag, count)
        frmt = ".png"
        output = './detection/output/live/'
        driver.save_screenshot(path+image+frmt)
        
        curr_count = image_detect(path, image, frmt, output)
        yield curr_count
        
        avg = (avg + curr_count)//2
        print(f"RUNNING AVERAGE: {avg}")
        time.sleep(delay)

    driver.close()


def live_detect_insecam(website, tag, delay=0):
    driver = webdriver.Chrome('./chromedriver')
    driver.get(website)
    time.sleep(15)
    button = driver.find_element_by_id('playIcon')
    button.click()
    time.sleep(1)
    button = driver.find_element_by_css_selector("[title*='Full screen']")
    button.click()
    time.sleep(2)

    print(f"CAPTURING IMAGES FROM: {website}.....")
    count = 0
    avg = 0.0
    while count <= 5:
        count += 1
        path = "./detection/live/"
        image = '{}-{}'.format(tag, count)
        frmt = ".png"
        output = './detection/output/live/'
        driver.save_screenshot(path+image+frmt)
        
        
        curr_count = image_detect(path, image, frmt, output)
        avg = (avg + curr_count)//2
        print(f"RUNNING AVERAGE: {avg}")
        time.sleep(delay)

videoFile = "./detection/stctrim.mp4"
path = "./detection/"
image = 'pub-pic'
frmt = ".png"
output = './detection/output/'
# image_detect(path, image, frmt, output)
# video_detect(videoFile)

# EXPERIMENT 1: New Orleans (use Cats Meow and The Bourbon Street View)
# url = "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeow2"
url = "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=bourbonstreet"
tag = "new-orleans"

# EXPERIMENT 2: Key West Florida
# url = "https://www.earthcam.com/usa/florida/keywest/?cam=irishkevins"
# tag = 'florida'

# EXPERIMENT 3: Times Square
# url = "https://www.earthcam.com/usa/newyork/timessquare/?cam=tsstreet"
# tag = "nyc"


conn, cur = setup_db()

for o in live_detect_earth_cam(url, tag):
    insert_to_occupancy_table(conn, cur, loc='New Orleans', cam='Cam1', area=400, occ=o)



# insecam_url = "http://128.206.113.98/#view"
# tag = "russia-mall"
# live_detect_insecam(insecam_url, tag)