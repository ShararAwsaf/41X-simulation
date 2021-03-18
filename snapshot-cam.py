from selenium import webdriver
import time

driver = webdriver.Chrome('./chromedriver')
driver.get("http://128.206.113.98/#view")
time.sleep(15)
button = driver.find_element_by_id('playIcon')
button.click()
time.sleep(1)
button = driver.find_element_by_css_selector("[title*='Full screen']")
button.click()
time.sleep(2)

# time.sleep(2)
# button = driver.find_element_by_class_name('lvc-btn')
# button.click()
# time.sleep(2)

def screen_shot(output_image_path):
    driver.save_screenshot(output_image_path)

output = './detection/z-russia-mall.png'
screen_shot(output)
driver.close()