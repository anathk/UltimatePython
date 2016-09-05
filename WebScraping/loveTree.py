from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time
import os
import pyautogui



url = 'http://lover.renren.com/600171362/lovetree?ref=love_tab'
driver = webdriver.Chrome()
# driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")

if os.name == 'posix':
    driver = webdriver.PhantomJS(executable_path='/opt/phantomjs/bin/phantomjs')
# driver.set_window_size(200, 300)
driver.get(url)

try:
    username_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'email')))
    password_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'password')))
    submit_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//input[@type="submit" and @value="登录"]')))
    username_element.send_keys('test')
    password_element.send_keys('test')
    submit_element.click()
    # print(driver.page_source)

    loveTree_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, '情侣树')))
    loveTree_element.click()
    time.sleep(3)
    print('start to locate')
    water_location = pyautogui.locateOnScreen('./img/water.png')
    print(water_location)
    flower_location = pyautogui.locateOnScreen('./img/flower.png')
    print(flower_location)

    pyautogui.click(pyautogui.center(water_location))
    time.sleep(3)
    pyautogui.click(pyautogui.center(water_location))
    time.sleep(3)
    pyautogui.click(pyautogui.center(flower_location))
    time.sleep(3)
    pyautogui.click(pyautogui.center(flower_location))
except Exception as e:
    print(e)
finally:
    driver.close()
    driver.quit()
