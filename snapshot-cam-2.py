from selenium import webdriver
import time

driver = webdriver.Chrome('./chromedriver')
driver.get("https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeow2")
time.sleep(20)

button = driver.find_element_by_class_name('snapshot-dome-btn-icon')
button.click()
time.sleep(10)
button = driver.find_element_by_css_selector('button.downloadBtn')
button.click()
driver.close()