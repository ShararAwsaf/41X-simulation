from selenium import webdriver
import time

driver = webdriver.Chrome('./chromedriver')
driver.get("http://128.206.113.98/#view")
time.sleep(15)
button = driver.find_element_by_id('playIcon')
button.click()

time.sleep(7)
button = driver.find_element_by_class_name('lvc-btn')
button.click()
time.sleep(2)
driver.close()