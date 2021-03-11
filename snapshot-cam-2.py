from selenium import webdriver
import time

driver = webdriver.Chrome('./chromedriver')
driver.get("https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeow2")

button = driver.find_element_by_class_name('fullScreenBtn')
button.click()
time.sleep(20)
driver.save_screenshot('./detection/z-new-orleans.png')
# driver.close()