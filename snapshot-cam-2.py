from selenium import webdriver
from selenium_video import VideoRecorder
import time

driver = webdriver.Chrome('./chromedriver')
driver.get("https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeow2")
time.sleep(20)
button = driver.find_element_by_class_name('fullScreenBtn')
button.click()

def screen_shot(output_image_path):
    driver.save_screenshot(output_image_path)


output = './detection/z-new-orleans.png'



driver.close()